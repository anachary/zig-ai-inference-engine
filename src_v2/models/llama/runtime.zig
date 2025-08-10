const std = @import("std");
const ir = @import("../../core/ir.zig");
const cfgmod = @import("config.zig");
const regs = @import("../../core/registries.zig");
const ggmlh = @import("../../runtime/ggml_helpers.zig");
const weights = @import("../../runtime/weight_store.zig");
const norm = @import("../../ops/normalization.zig");
const mm = @import("../../ops/matmul.zig");

pub const ProfileMode = enum { off, text, json };
var g_profile_mode: ProfileMode = .text;

pub fn setProfileMode(mode: ProfileMode) void {
    g_profile_mode = mode;
}

// Weight cache for dequantized weights
const WeightCache = struct {
    data: std.StringHashMap([]f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) WeightCache {
        return .{
            .data = std.StringHashMap([]f32).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *WeightCache) void {
        var it = self.data.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.data.deinit();
    }

    pub fn getOrDequantize(self: *WeightCache, store: *weights.WeightStore, view: weights.TensorView, quantized_size: usize) ![]f32 {
        // Check cache first - this is the key optimization!
        if (self.data.get(view.name)) |cached| {
            std.io.getStdOut().writer().print("ok: cache_hit {s}\n", .{view.name}) catch {};
            return cached;
        }

        // Dequantize and cache - only happens once per weight!
        const dequantized_size = try ggmlh.calcDequantizedLen(view.ggml_type_id, view.shape);
        var data = try self.allocator.alloc(f32, dequantized_size);
        std.io.getStdOut().writer().print("ok: dequantizing {s} size={d}\n", .{view.name, dequantized_size}) catch {};
        try store.dequantizeInto(view, quantized_size, data, self.allocator);
        
        // Cache the dequantized weights for future use
        const name_copy = try self.allocator.dupe(u8, view.name);
        try self.data.put(name_copy, data);
        
        return data;
    }
};

pub const LlamaRuntime = struct {
    allocator: std.mem.Allocator,
    model: *const ir.ModelDescriptor,
    config: cfgmod.LlamaConfig,
    store: weights.WeightStore,
    weight_cache: WeightCache,

    // Snapshot of tensor list to avoid external mutation of len/ptr
    tensors: []ir.TensorMeta = &[_]ir.TensorMeta{},

    // Liveness/canary
    is_alive: bool = false,
    canary: u64 = 0,

    // KV cache buffers
    k_cache: []f32 = &[_]f32{},
    v_cache: []f32 = &[_]f32{},
    pos: usize = 0,

    // Pre-allocated working buffers (reused across all layers)
    working_buffers: WorkingBuffers,

    const WorkingBuffers = struct {
        // Attention buffers
        q: []f32,
        k: []f32,
        v: []f32,
        attn_out: []f32,
        
        // FFN buffers
        ffn_in: []f32,
        gate: []f32,
        up: []f32,
        down: []f32,
        
        // Norm buffers
        gamma: []f32,
        fgamma: []f32,
        
        // Weight buffers (reused for different weights)
        weight_buffer: []f32,

        pub fn init(allocator: std.mem.Allocator, config: cfgmod.LlamaConfig) !WorkingBuffers {
            const hidden_dim = config.hidden_dim;
            const ffn_dim = config.ffn_dim;
            const num_heads = config.num_heads;
            const num_kv_heads = config.num_kv_heads;
            const head_dim = config.head_dim;
            
            // Calculate max weight buffer size needed
            const max_weight_size = @max(
                hidden_dim * hidden_dim, // attention weights
                hidden_dim * ffn_dim,    // FFN weights
                ffn_dim * hidden_dim     // FFN output weights
            );

            return .{
                .q = try allocator.alloc(f32, num_heads * head_dim),
                .k = try allocator.alloc(f32, num_kv_heads * head_dim),
                .v = try allocator.alloc(f32, num_kv_heads * head_dim),
                .attn_out = try allocator.alloc(f32, num_heads * head_dim),
                .ffn_in = try allocator.alloc(f32, hidden_dim),
                .gate = try allocator.alloc(f32, ffn_dim),
                .up = try allocator.alloc(f32, ffn_dim),
                .down = try allocator.alloc(f32, hidden_dim),
                .gamma = try allocator.alloc(f32, hidden_dim),
                .fgamma = try allocator.alloc(f32, hidden_dim),
                .weight_buffer = try allocator.alloc(f32, max_weight_size),
            };
        }

        pub fn deinit(self: *WorkingBuffers, allocator: std.mem.Allocator) void {
            allocator.free(self.q);
            allocator.free(self.k);
            allocator.free(self.v);
            allocator.free(self.attn_out);
            allocator.free(self.ffn_in);
            allocator.free(self.gate);
            allocator.free(self.up);
            allocator.free(self.down);
            allocator.free(self.gamma);
            allocator.free(self.fgamma);
            allocator.free(self.weight_buffer);
        }
    };

    pub fn init(allocator: std.mem.Allocator, model: *const ir.ModelDescriptor, model_path: []const u8, data_offset: usize) !*LlamaRuntime {
        var self = try allocator.create(LlamaRuntime);
        self.allocator = allocator;
        // store a pointer to the session-owned model (stable lifetime)
        self.model = model;
        self.config = try cfgmod.build(allocator, model);
        
        // Allocate KV caches based on config
        if (self.config.context_length == 0 or self.config.head_dim == 0 or self.config.num_kv_heads == 0 or self.config.num_layers == 0) {
            return error.InvalidConfig;
        }
        const kv_elems_per_layer = self.config.num_kv_heads * self.config.context_length * self.config.head_dim;
        const total_layers = self.config.num_layers;
        const total_elems = kv_elems_per_layer * total_layers;
        self.k_cache = try allocator.alloc(f32, total_elems);
        
        // Snapshot tensors list by value to protect against external mutation of ArrayList length
        if (self.model.tensors.items.len > 0) {
            self.tensors = try allocator.alloc(ir.TensorMeta, self.model.tensors.items.len);
            std.mem.copy(ir.TensorMeta, self.tensors, self.model.tensors.items);
        } else {
            self.tensors = &[_]ir.TensorMeta{};
        }

        self.v_cache = try allocator.alloc(f32, total_elems);
        @memset(self.k_cache, 0);
        @memset(self.v_cache, 0);

        self.store = try weights.WeightStore.open(self.allocator, model_path, data_offset);
        self.weight_cache = WeightCache.init(self.allocator);
        self.working_buffers = try WorkingBuffers.init(self.allocator, self.config);
        
        // CRITICAL OPTIMIZATION: Pre-load all weights at startup
        // This eliminates repeated dequantization during inference
        std.debug.print("INFO: Pre-loading all weights (one-time cost)...\n", .{});
        const t0 = std.time.nanoTimestamp();
        
        // Pre-load embedding weights
        const emb_meta = self.tensors[self.config.token_embd];
        _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = emb_meta.name, .shape = emb_meta.shape, .ggml_type_id = emb_meta.ggml_type_id, .offset = emb_meta.offset }, try ggmlh.calcQuantizedSize(emb_meta.ggml_type_id, emb_meta.shape));
        
        // Pre-load all layer weights (sequential for now - working version)
        var layer: usize = 0;
        while (layer < self.config.num_layers) : (layer += 1) {
            // Attention weights
            const an_meta = self.tensors[self.config.attn_norm[layer]];
            _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = an_meta.name, .shape = an_meta.shape, .ggml_type_id = an_meta.ggml_type_id, .offset = an_meta.offset }, try ggmlh.calcQuantizedSize(an_meta.ggml_type_id, an_meta.shape));
            
            const wq_meta = self.tensors[self.config.wq[layer]];
            _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = wq_meta.name, .shape = wq_meta.shape, .ggml_type_id = wq_meta.ggml_type_id, .offset = wq_meta.offset }, try ggmlh.calcQuantizedSize(wq_meta.ggml_type_id, wq_meta.shape));
            
            const wk_meta = self.tensors[self.config.wk[layer]];
            _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = wk_meta.name, .shape = wk_meta.shape, .ggml_type_id = wk_meta.ggml_type_id, .offset = wk_meta.offset }, try ggmlh.calcQuantizedSize(wk_meta.ggml_type_id, wk_meta.shape));
            
            const wv_meta = self.tensors[self.config.wv[layer]];
            _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = wv_meta.name, .shape = wv_meta.shape, .ggml_type_id = wv_meta.ggml_type_id, .offset = wv_meta.offset }, try ggmlh.calcQuantizedSize(wv_meta.ggml_type_id, wv_meta.shape));
            
            const wo_meta = self.tensors[self.config.wo[layer]];
            _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = wo_meta.name, .shape = wo_meta.shape, .ggml_type_id = wo_meta.ggml_type_id, .offset = wo_meta.offset }, try ggmlh.calcQuantizedSize(wo_meta.ggml_type_id, wo_meta.shape));
            
            // FFN weights
            const fn_meta = self.tensors[self.config.ffn_norm[layer]];
            _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = fn_meta.name, .shape = fn_meta.shape, .ggml_type_id = fn_meta.ggml_type_id, .offset = fn_meta.offset }, try ggmlh.calcQuantizedSize(fn_meta.ggml_type_id, fn_meta.shape));
            
            const w_gate = self.tensors[self.config.ffn_gate[layer]];
            _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = w_gate.name, .shape = w_gate.shape, .ggml_type_id = w_gate.ggml_type_id, .offset = w_gate.offset }, try ggmlh.calcQuantizedSize(w_gate.ggml_type_id, w_gate.shape));
            
            const w_up = self.tensors[self.config.ffn_up[layer]];
            _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = w_up.name, .shape = w_up.shape, .ggml_type_id = w_up.ggml_type_id, .offset = w_up.offset }, try ggmlh.calcQuantizedSize(w_up.ggml_type_id, w_up.shape));
            
            const w_down = self.tensors[self.config.ffn_down[layer]];
            _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = w_down.name, .shape = w_down.shape, .ggml_type_id = w_down.ggml_type_id, .offset = w_down.offset }, try ggmlh.calcQuantizedSize(w_down.ggml_type_id, w_down.shape));
        }
        
        // Pre-load output weights
        const out_meta = self.tensors[self.config.output_weight];
        _ = try self.weight_cache.getOrDequantize(&self.store, .{ .name = out_meta.name, .shape = out_meta.shape, .ggml_type_id = out_meta.ggml_type_id, .offset = out_meta.offset }, try ggmlh.calcQuantizedSize(out_meta.ggml_type_id, out_meta.shape));
        
        const t1 = std.time.nanoTimestamp();
        const load_ms = @divTrunc(t1 - t0, 1_000_000);
        std.debug.print("INFO: Weight pre-loading completed in {d}ms\n", .{load_ms});
        std.debug.print("INFO: All weights are now cached and ready for fast inference\n", .{});
        
        self.pos = 0;
        // mark alive and set a simple canary derived from model_path and data_offset
        self.is_alive = true;
        var h: u64 = 1469598103934665603; // FNV-1a offset basis
        for (model_path) |b| {
            h ^= @as(u64, @intCast(b));
            h *%= 1099511628211;
        }
        self.canary = h ^ @as(u64, @intCast(data_offset));
        std.debug.print("INFO: runtime tensors_len={} vocab={}\n", .{ self.model.tensors.items.len, self.model.tokenizer.vocab_size });

        return self;
    }

    pub fn deinit(self: *LlamaRuntime) void {
        self.store.close();
        self.weight_cache.deinit();
        self.working_buffers.deinit(self.allocator);
        
        // Free config arrays
        self.allocator.free(self.config.attn_norm);
        self.allocator.free(self.config.wq);
        self.allocator.free(self.config.wk);
        self.allocator.free(self.config.wv);
        self.allocator.free(self.config.wo);
        self.allocator.free(self.config.ffn_norm);
        self.allocator.free(self.config.ffn_gate);
        self.allocator.free(self.config.ffn_up);
        self.allocator.free(self.config.ffn_down);
        self.allocator.destroy(self);
    }

    // Full single-step forward with KV cache and registries
    pub fn forward(self: *LlamaRuntime, tokens: []const u32, out_logits: []f32) !void {
        std.io.getStdOut().writer().print("ok: fwd_start\n", .{}) catch {};
        std.io.getStdOut().writer().print("ok: emb_before\n", .{}) catch {};

        if (tokens.len == 0) return error.EmptyTokens;
        const t = tokens.len - 1;

        // 1) Embed current token
        std.debug.assert(self.is_alive);
        std.debug.assert(@intFromPtr(self.model) != 0);
        // Sanity guard: check runtime snapshot, not the external ArrayList
        const tcount = self.tensors.len;
        std.io.getStdOut().writer().print("ok: tensors_len_snapshot={d} model_len={d}\n", .{ tcount, self.model.tensors.items.len }) catch {};
        if (tcount < 10 or tcount > 1_000_000) return error.InvalidModel;

        std.io.getStdOut().writer().print("ok: emb_idx idx={d} len={d}\n", .{ self.config.token_embd, tcount }) catch {};
        if (self.config.token_embd >= self.tensors.len) return error.InvalidConfig;
        const emb_meta = self.tensors[self.config.token_embd];
        std.io.getStdOut().writer().print("ok: emb_shape len={d} d0={d} d1={d}\n", .{
            emb_meta.shape.len,
            if (emb_meta.shape.len >= 1) emb_meta.shape[0] else 0,
            if (emb_meta.shape.len >= 2) emb_meta.shape[1] else 0,
        }) catch {};
        std.io.getStdOut().writer().print("ok: vocab={d}\n", .{self.model.tokenizer.vocab_size}) catch {};
        const hidden_dim = self.config.hidden_dim;

        // Embedding matrix may be (vocab x hidden) or (hidden x vocab). Extract an appropriate slice.
        var vocab: usize = 0;
        var orient_vh = false; // true if (vocab x hidden)
        if (emb_meta.shape.len >= 2 and emb_meta.shape[0] == self.model.tokenizer.vocab_size and emb_meta.shape[1] == hidden_dim) {
            vocab = emb_meta.shape[0];
            orient_vh = true;
        } else if (emb_meta.shape.len >= 2 and emb_meta.shape[1] == self.model.tokenizer.vocab_size and emb_meta.shape[0] == hidden_dim) {
            vocab = emb_meta.shape[1];
            orient_vh = false;
        } else {
            // Fallback: use the larger dimension as vocab, smaller as hidden
            if (emb_meta.shape.len >= 2) {
                if (emb_meta.shape[0] > emb_meta.shape[1]) {
                    vocab = emb_meta.shape[0];
                    orient_vh = true;
                } else {
                    vocab = emb_meta.shape[1];
                    orient_vh = false;
                }
            } else {
                // Breadcrumb: embedding start
                std.io.getStdOut().writer().print("ok: emb\n", .{}) catch {};

                return error.ShapeMismatch;
            }
        }
        if (tokens[t] >= vocab) return error.TokenOutOfRange;
        std.io.getStdOut().writer().print("ok: emb_orient hv={} vocab={d} hid={d} tok={d}\n", .{ !orient_vh, vocab, hidden_dim, tokens[t] }) catch {};

        // Load the embedding vector for the current token efficiently
        var x = try self.allocator.alloc(f32, hidden_dim);
        defer self.allocator.free(x);
        if (orient_vh) {
            // (vocab x hidden): fast path — one row
            // For now, read one column as if matrix is (hidden x vocab); this works because we only support f32/f16 partial reads
            std.io.getStdOut().writer().print("ok: emb_loaded\n", .{}) catch {};

            try self.store.readEmbeddingColumn(.{ .name = emb_meta.name, .shape = emb_meta.shape, .ggml_type_id = emb_meta.ggml_type_id, .offset = emb_meta.offset }, tokens[t], hidden_dim, x, self.allocator);
        } else {
            // (hidden x vocab): read one column directly
            try self.store.readEmbeddingColumn(.{ .name = emb_meta.name, .shape = emb_meta.shape, .ggml_type_id = emb_meta.ggml_type_id, .offset = emb_meta.offset }, tokens[t], hidden_dim, x, self.allocator);
        }

        // Resolve algorithms
        const rms = regs.getNorm("rmsnorm") orelse return error.Unimplemented;
        const rope = regs.getRope("rope:llama") orelse return error.Unimplemented;
        const attn = regs.getAttention("attn:cpu:causal") orelse return error.Unimplemented;
        const act = regs.getActivation("act:swiglu") orelse return error.Unimplemented;

        // Layer loop - using pre-allocated buffers
        var layer: usize = 0;
        while (layer < self.config.num_layers) : (layer += 1) {
            const t0 = std.time.nanoTimestamp();
            var tp = t0; // prev
            var ms_deq_gamma: i128 = 0;
            var ms_deq_wq: i128 = 0;
            var ms_deq_wk: i128 = 0;
            var ms_deq_wv: i128 = 0;
            var ms_mm_q: i128 = 0;
            var ms_mm_k: i128 = 0;
            var ms_mm_v: i128 = 0;
            var ms_rope: i128 = 0;
            var ms_cache: i128 = 0;
            var ms_attn: i128 = 0;
            var ms_deq_wo: i128 = 0;
            var ms_mm_wo: i128 = 0;
            var ms_resid1: i128 = 0;
            var ms_deq_fgamma: i128 = 0;
            var ms_scale_fgamma: i128 = 0;
            var ms_deq_wg: i128 = 0;
            var ms_deq_wu: i128 = 0;
            var ms_deq_wd: i128 = 0;
            var ms_mm_wg: i128 = 0;
            var ms_mm_wu: i128 = 0;
            var ms_act: i128 = 0;
            var ms_mm_wd: i128 = 0;
            var ms_resid2: i128 = 0;

            // attn norm gamma - using cached weights
            const an_meta = self.tensors[self.config.attn_norm[layer]];
            const gamma = try self.weight_cache.getOrDequantize(&self.store, .{ .name = an_meta.name, .shape = an_meta.shape, .ggml_type_id = an_meta.ggml_type_id, .offset = an_meta.offset }, try ggmlh.calcQuantizedSize(an_meta.ggml_type_id, an_meta.shape));
            var tnow = std.time.nanoTimestamp();
            ms_deq_gamma = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            rms.apply(x, 1e-6);
            var i: usize = 0;
            while (i < hidden_dim) : (i += 1) x[i] *= gamma[i];
            tnow = std.time.nanoTimestamp();
            ms_scale_fgamma = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;

            // QKV projections - using cached weights
            const wq_meta = self.tensors[self.config.wq[layer]];
            const wk_meta = self.tensors[self.config.wk[layer]];
            const wv_meta = self.tensors[self.config.wv[layer]];
            
            const wq = try self.weight_cache.getOrDequantize(&self.store, .{ .name = wq_meta.name, .shape = wq_meta.shape, .ggml_type_id = wq_meta.ggml_type_id, .offset = wq_meta.offset }, try ggmlh.calcQuantizedSize(wq_meta.ggml_type_id, wq_meta.shape));
            const wk = try self.weight_cache.getOrDequantize(&self.store, .{ .name = wk_meta.name, .shape = wk_meta.shape, .ggml_type_id = wk_meta.ggml_type_id, .offset = wk_meta.offset }, try ggmlh.calcQuantizedSize(wk_meta.ggml_type_id, wk_meta.shape));
            std.io.getStdOut().writer().print("ok: kv layer={d}\n", .{layer}) catch {};
            const wv = try self.weight_cache.getOrDequantize(&self.store, .{ .name = wv_meta.name, .shape = wv_meta.shape, .ggml_type_id = wv_meta.ggml_type_id, .offset = wv_meta.offset }, try ggmlh.calcQuantizedSize(wv_meta.ggml_type_id, wv_meta.shape));
            
            tnow = std.time.nanoTimestamp();
            ms_deq_wq = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            tnow = std.time.nanoTimestamp();
            ms_deq_wk = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            tnow = std.time.nanoTimestamp();
            ms_deq_wv = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            
            // FUSED QKV PROJECTION - Maximum performance!
            const q_out = self.config.num_heads * self.config.head_dim;
            const kv_out = self.config.num_kv_heads * self.config.head_dim;
            mm.fusedQKVProjection(
                x, wq, wk, wv,
                self.working_buffers.q, self.working_buffers.k, self.working_buffers.v,
                hidden_dim, q_out, kv_out
            );
            tnow = std.time.nanoTimestamp();
            ms_mm_q = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            ms_mm_k = ms_mm_q; // Fused operation
            ms_mm_v = ms_mm_q; // Fused operation
            tp = tnow;

            // RoPE per head and GQA/MQA mapping
            var h: usize = 0;
            while (h < self.config.num_heads) : (h += 1) {
                const qh = self.working_buffers.q[h * self.config.head_dim .. (h + 1) * self.config.head_dim];
                const kh = self.working_buffers.k[(h % self.config.num_kv_heads) * self.config.head_dim .. ((h % self.config.num_kv_heads) + 1) * self.config.head_dim];
                rope.apply(qh, kh, self.config.head_dim, t, self.model.rope.theta);
            }
            tnow = std.time.nanoTimestamp();
            ms_rope = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;

            // Append K/V to cache
            const layer_stride = self.config.num_kv_heads * self.config.context_length * self.config.head_dim;
            const step_stride = self.config.num_kv_heads * self.config.head_dim;
            const base = layer * layer_stride + t * step_stride;
            @memcpy(self.k_cache[base .. base + step_stride], self.working_buffers.k);
            @memcpy(self.v_cache[base .. base + step_stride], self.working_buffers.v);
            tnow = std.time.nanoTimestamp();
            ms_cache = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;

            // Attention using cache up to t
            const k_slice = self.k_cache[layer * layer_stride .. layer * layer_stride + (t + 1) * step_stride];
            const v_slice = self.v_cache[layer * layer_stride .. layer * layer_stride + (t + 1) * step_stride];
            attn.compute(self.config.num_heads, self.config.num_kv_heads, self.config.head_dim, t, self.working_buffers.q, k_slice, v_slice, self.working_buffers.attn_out);
            tnow = std.time.nanoTimestamp();
            ms_attn = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;

            // Wo and residual - using cached weights
            const wo_meta = self.tensors[self.config.wo[layer]];
            const wo = try self.weight_cache.getOrDequantize(&self.store, .{ .name = wo_meta.name, .shape = wo_meta.shape, .ggml_type_id = wo_meta.ggml_type_id, .offset = wo_meta.offset }, try ggmlh.calcQuantizedSize(wo_meta.ggml_type_id, wo_meta.shape));
            tnow = std.time.nanoTimestamp();
            ms_deq_wo = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            if (wo_meta.shape.len >= 2 and wo_meta.shape[0] == hidden_dim and wo_meta.shape[1] == hidden_dim) {
                mm.matmulF32(self.working_buffers.attn_out, wo, 1, hidden_dim, hidden_dim, self.working_buffers.ffn_in);
            } else if (wo_meta.shape.len >= 2 and wo_meta.shape[1] == hidden_dim and wo_meta.shape[0] == hidden_dim) {
                mm.matmulF32_Bt(self.working_buffers.attn_out, wo, 1, hidden_dim, hidden_dim, self.working_buffers.ffn_in);
            } else return error.ShapeMismatch;
            tnow = std.time.nanoTimestamp();
            ms_mm_wo = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            var r: usize = 0;
            while (r < hidden_dim) : (r += 1) x[r] += self.working_buffers.ffn_in[r];
            tnow = std.time.nanoTimestamp();
            ms_resid1 = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;

            // FFN block - using cached weights
            const fn_meta = self.tensors[self.config.ffn_norm[layer]];
            const fgamma = try self.weight_cache.getOrDequantize(&self.store, .{ .name = fn_meta.name, .shape = fn_meta.shape, .ggml_type_id = fn_meta.ggml_type_id, .offset = fn_meta.offset }, try ggmlh.calcQuantizedSize(fn_meta.ggml_type_id, fn_meta.shape));
            tnow = std.time.nanoTimestamp();
            ms_deq_fgamma = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            rms.apply(x, 1e-6);
            var ii: usize = 0;
            while (ii < hidden_dim) : (ii += 1) x[ii] *= fgamma[ii];
            tnow = std.time.nanoTimestamp();
            ms_scale_fgamma = ms_scale_fgamma + @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            
            const w_gate = self.tensors[self.config.ffn_gate[layer]];
            const w_up = self.tensors[self.config.ffn_up[layer]];
            const w_down = self.tensors[self.config.ffn_down[layer]];
            
            const wg = try self.weight_cache.getOrDequantize(&self.store, .{ .name = w_gate.name, .shape = w_gate.shape, .ggml_type_id = w_gate.ggml_type_id, .offset = w_gate.offset }, try ggmlh.calcQuantizedSize(w_gate.ggml_type_id, w_gate.shape));
            tnow = std.time.nanoTimestamp();
            ms_deq_wg = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            const wu = try self.weight_cache.getOrDequantize(&self.store, .{ .name = w_up.name, .shape = w_up.shape, .ggml_type_id = w_up.ggml_type_id, .offset = w_up.offset }, try ggmlh.calcQuantizedSize(w_up.ggml_type_id, w_up.shape));
            tnow = std.time.nanoTimestamp();
            ms_deq_wu = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            const wd = try self.weight_cache.getOrDequantize(&self.store, .{ .name = w_down.name, .shape = w_down.shape, .ggml_type_id = w_down.ggml_type_id, .offset = w_down.offset }, try ggmlh.calcQuantizedSize(w_down.ggml_type_id, w_down.shape));
            tnow = std.time.nanoTimestamp();
            ms_deq_wd = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            
            mm.matmulF32(x, wg, 1, hidden_dim, self.config.ffn_dim, self.working_buffers.gate);
            tnow = std.time.nanoTimestamp();
            ms_mm_wg = @as(i128, @divTrunc(tnow - tp, 1_000_000));
            tp = tnow;
            mm.matmulF32(x, wu, 1, hidden_dim, self.config.ffn_dim, self.working_buffers.up);
            tnow = std.time.nanoTimestamp();
            ms_mm_wu = @divTrunc(tnow - tp, 1_000_000);
            tp = tnow;
            act.swiglu(self.working_buffers.up, self.working_buffers.gate);
            tnow = std.time.nanoTimestamp();
            ms_act = @divTrunc(tnow - tp, 1_000_000);
            tp = tnow;
            
            if (w_down.shape.len >= 2 and w_down.shape[0] == self.config.ffn_dim and w_down.shape[1] == hidden_dim) {
                mm.matmulF32(self.working_buffers.up, wd, 1, self.config.ffn_dim, hidden_dim, self.working_buffers.down);
            } else if (w_down.shape.len >= 2 and w_down.shape[1] == self.config.ffn_dim and w_down.shape[0] == hidden_dim) {
                mm.matmulF32_Bt(self.working_buffers.up, wd, 1, self.config.ffn_dim, hidden_dim, self.working_buffers.down);
            } else return error.ShapeMismatch;
            tnow = std.time.nanoTimestamp();
            ms_mm_wd = @divTrunc(tnow - tp, 1_000_000);
            tp = tnow;
            var rr: usize = 0;
            while (rr < hidden_dim) : (rr += 1) x[rr] += self.working_buffers.down[rr];
            tnow = std.time.nanoTimestamp();
            ms_resid2 = @divTrunc(tnow - tp, 1_000_000);

            const t1 = std.time.nanoTimestamp();
            const ms_total = @divTrunc(t1 - t0, 1_000_000);
            switch (g_profile_mode) {
                .off => {},
                .text => {
                    std.io.getStdOut().writer().print(
                        "ok: layer={d} ms total={d} deq_gamma={d} scale_gamma={d} deq_wq/wk/wv={d}/{d}/{d} mm_q/k/v={d}/{d}/{d} rope={d} cache={d} attn={d} deq_wo={d} mm_wo={d} resid1={d} deq_wg/wu/wd={d}/{d}/{d} mm_wg/wu/wd={d}/{d}/{d} act={d} mm_wd={d} resid2={d}\n",
                        .{ layer, ms_total, ms_deq_gamma, ms_scale_fgamma, ms_deq_wq, ms_deq_wk, ms_deq_wv, ms_mm_q, ms_mm_k, ms_mm_v, ms_rope, ms_cache, ms_attn, ms_deq_wo, ms_mm_wo, ms_resid1, ms_deq_wg, ms_deq_wu, ms_deq_wd, ms_mm_wg, ms_mm_wu, ms_mm_wd, ms_act, ms_mm_wd, ms_resid2 },
                    ) catch {};
                },
                .json => {
                    std.io.getStdOut().writer().print(
                        "ok_json: {{\"layer\":{d},\"ms_total\":{d},\"ms\":{{\"deq_gamma\":{d},\"scale_gamma\":{d},\"deq_wq\":{d},\"deq_wk\":{d},\"deq_wv\":{d},\"mm_q\":{d},\"mm_k\":{d},\"mm_v\":{d},\"rope\":{d},\"cache\":{d},\"attn\":{d},\"deq_wo\":{d},\"mm_wo\":{d},\"resid1\":{d},\"deq_wg\":{d},\"deq_wu\":{d},\"deq_wd\":{d},\"mm_wg\":{d},\"mm_wu\":{d},\"mm_wd\":{d},\"act\":{d},\"resid2\":{d}}}}}\n",
                        .{ layer, ms_total, ms_deq_gamma, ms_scale_fgamma, ms_deq_wq, ms_deq_wk, ms_deq_wv, ms_mm_q, ms_mm_k, ms_mm_v, ms_rope, ms_cache, ms_attn, ms_deq_wo, ms_mm_wo, ms_resid1, ms_deq_wg, ms_deq_wu, ms_deq_wd, ms_mm_wg, ms_mm_wu, ms_mm_wd, ms_act, ms_resid2 },
                    ) catch {};
                },
            }
        }

        // Final logits - using cached weights
        const out_meta = self.tensors[self.config.output_weight];
        const out_w = try self.weight_cache.getOrDequantize(&self.store, .{ .name = out_meta.name, .shape = out_meta.shape, .ggml_type_id = out_meta.ggml_type_id, .offset = out_meta.offset }, try ggmlh.calcQuantizedSize(out_meta.ggml_type_id, out_meta.shape));
        
        // Determine vocab size and orientation
        var vocab_out: usize = 0;
        if (out_meta.shape.len >= 2 and out_meta.shape[0] == hidden_dim) {
            vocab_out = out_meta.shape[1];
            if (out_logits.len < vocab_out) return error.ShapeMismatch;
            mm.matmulF32(x, out_w, 1, hidden_dim, vocab_out, out_logits);
        } else if (out_meta.shape.len >= 2 and out_meta.shape[1] == hidden_dim) {
            vocab_out = out_meta.shape[0];
            if (out_logits.len < vocab_out) return error.ShapeMismatch;
            mm.matmulF32_Bt(x, out_w, 1, hidden_dim, vocab_out, out_logits);
        } else return error.ShapeMismatch;
    }
};

const std = @import("std");
const ir = @import("../../core/ir.zig");

pub const TensorRef = struct {
    name: []const u8,
    idx: usize,
};

pub const LlamaConfig = struct {
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    context_length: usize,

    token_embd: usize,
    output_weight: usize,

    // per-layer indices into md.tensors
    attn_norm: []usize,
    wq: []usize,
    wk: []usize,
    wv: []usize,
    wo: []usize,
    ffn_norm: []usize,
    ffn_gate: []usize,
    ffn_up: []usize,
    ffn_down: []usize,
};

pub fn build(allocator: std.mem.Allocator, md: *const ir.ModelDescriptor) !LlamaConfig {
    var cfg = LlamaConfig{
        .num_layers = md.num_layers,
        .num_heads = md.num_heads,
        .num_kv_heads = md.num_kv_heads,
        .head_dim = md.head_dim,
        .hidden_dim = md.hidden_dim,
        .ffn_dim = md.ffn_dim,
        .context_length = md.context_length,
        .token_embd = std.math.maxInt(usize),
        .output_weight = std.math.maxInt(usize),
        .attn_norm = &[_]usize{},
        .wq = &[_]usize{},
        .wk = &[_]usize{},
        .wv = &[_]usize{},
        .wo = &[_]usize{},
        .ffn_norm = &[_]usize{},
        .ffn_gate = &[_]usize{},
        .ffn_up = &[_]usize{},
        .ffn_down = &[_]usize{},
    };

    // allocate arrays
    cfg.attn_norm = try allocator.alloc(usize, md.num_layers);
    cfg.wq = try allocator.alloc(usize, md.num_layers);
    cfg.wk = try allocator.alloc(usize, md.num_layers);
    cfg.wv = try allocator.alloc(usize, md.num_layers);
    cfg.wo = try allocator.alloc(usize, md.num_layers);
    cfg.ffn_norm = try allocator.alloc(usize, md.num_layers);
    cfg.ffn_gate = try allocator.alloc(usize, md.num_layers);
    cfg.ffn_up = try allocator.alloc(usize, md.num_layers);
    cfg.ffn_down = try allocator.alloc(usize, md.num_layers);
    // initialize arrays with sentinel values
    var ii: usize = 0;
    while (ii < md.num_layers) : (ii += 1) {
        cfg.attn_norm[ii] = std.math.maxInt(usize);
        cfg.wq[ii] = std.math.maxInt(usize);
        cfg.wk[ii] = std.math.maxInt(usize);
        cfg.wv[ii] = std.math.maxInt(usize);
        cfg.wo[ii] = std.math.maxInt(usize);
        cfg.ffn_norm[ii] = std.math.maxInt(usize);
        cfg.ffn_gate[ii] = std.math.maxInt(usize);
        cfg.ffn_up[ii] = std.math.maxInt(usize);
        cfg.ffn_down[ii] = std.math.maxInt(usize);
    }

    // scan tensor names
    for (md.tensors.items, 0..) |t, idx| {
        if (std.mem.eql(u8, t.name, "token_embd.weight") or std.mem.eql(u8, t.name, "tok_embeddings.weight")) {
            cfg.token_embd = idx;
        } else if (std.mem.eql(u8, t.name, "output.weight")) {
            cfg.output_weight = idx;
        } else {
            // layer patterns
            var layer: usize = 0;
            var matched = false;
            while (layer < md.num_layers) : (layer += 1) {
                var buf: [64]u8 = undefined;
                if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_norm.weight", .{layer})) |key| {
                    if (std.mem.eql(u8, t.name, key)) {
                        cfg.attn_norm[layer] = idx;
                        matched = true;
                        break;
                    }
                } else |_| {}
                if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_q.weight", .{layer})) |key| {
                    if (std.mem.eql(u8, t.name, key)) {
                        cfg.wq[layer] = idx;
                        matched = true;
                        break;
                    }
                } else |_| {}
                if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_k.weight", .{layer})) |key| {
                    if (std.mem.eql(u8, t.name, key)) {
                        cfg.wk[layer] = idx;
                        matched = true;
                        break;
                    }
                } else |_| {}
                if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_v.weight", .{layer})) |key| {
                    if (std.mem.eql(u8, t.name, key)) {
                        cfg.wv[layer] = idx;
                        matched = true;
                        break;
                    }
                } else |_| {}
                if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_output.weight", .{layer})) |key| {
                    if (std.mem.eql(u8, t.name, key)) {
                        cfg.wo[layer] = idx;
                        matched = true;
                        break;
                    }
                } else |_| {}
                if (std.fmt.bufPrint(buf[0..], "blk.{d}.ffn_norm.weight", .{layer})) |key| {
                    if (std.mem.eql(u8, t.name, key)) {
                        cfg.ffn_norm[layer] = idx;
                        matched = true;
                        break;
                    }
                } else |_| {}
                if (std.fmt.bufPrint(buf[0..], "blk.{d}.ffn_gate.weight", .{layer})) |key| {
                    if (std.mem.eql(u8, t.name, key)) {
                        cfg.ffn_gate[layer] = idx;
                        matched = true;
                        break;
                    }
                } else |_| {}
                if (std.fmt.bufPrint(buf[0..], "blk.{d}.ffn_up.weight", .{layer})) |key| {
                    if (std.mem.eql(u8, t.name, key)) {
                        cfg.ffn_up[layer] = idx;
                        matched = true;
                        break;
                    }
                } else |_| {}
                if (std.fmt.bufPrint(buf[0..], "blk.{d}.ffn_down.weight", .{layer})) |key| {
                    if (std.mem.eql(u8, t.name, key)) {
                        cfg.ffn_down[layer] = idx;
                        matched = true;
                        break;
                    }
                } else |_| {}
            }
        }
    }
    // Fallback: detect token embedding and output weight by shape if names did not match
    if (cfg.token_embd == std.math.maxInt(usize)) {
        var idx_best: ?usize = null;
        for (md.tensors.items, 0..) |t, idx| {
            if (t.shape.len >= 2) {
                const a = t.shape[0];
                const b = t.shape[1];
                if ((a == md.tokenizer.vocab_size and b == md.hidden_dim) or (b == md.tokenizer.vocab_size and a == md.hidden_dim)) {
                    idx_best = idx;
                    break;
                }
            }
        }
        if (idx_best) |i| cfg.token_embd = i;
    }
    if (cfg.output_weight == std.math.maxInt(usize)) {
        var idx_best: ?usize = null;
        for (md.tensors.items, 0..) |t, idx| {
            if (t.shape.len >= 2) {
                const a = t.shape[0];
                const b = t.shape[1];
                if ((a == md.hidden_dim and b == md.tokenizer.vocab_size) or (b == md.hidden_dim and a == md.tokenizer.vocab_size)) {
                    idx_best = idx;
                    break;
                }
            }
        }
        if (idx_best) |i| cfg.output_weight = i;
    }

    // Diagnostics: report which tensors were selected
    if (cfg.token_embd != std.math.maxInt(usize)) {
        const te = md.tensors.items[cfg.token_embd];
        std.debug.print("INFO: token_embd -> idx={}, name={s}, shape_len={}, shape={any}, ggml_type_id={}\n", .{ cfg.token_embd, te.name, te.shape.len, te.shape, te.ggml_type_id });
    } else {
        std.debug.print("WARN: token_embd not found by name or fallback\n", .{});
    }
    if (cfg.output_weight != std.math.maxInt(usize)) {
        const ow = md.tensors.items[cfg.output_weight];
        std.debug.print("INFO: output_weight -> idx={}, name={s}, shape_len={}, shape={any}\n", .{ cfg.output_weight, ow.name, ow.shape.len, ow.shape });
    } else {
        std.debug.print("WARN: output_weight not found by name or fallback\n", .{});
    }

    // final validation after scanning all tensors
    if (cfg.token_embd == std.math.maxInt(usize) or cfg.output_weight == std.math.maxInt(usize))
        return error.MissingCoreWeights;
    var layer_check: usize = 0;
    while (layer_check < md.num_layers) : (layer_check += 1) {
        if (cfg.attn_norm[layer_check] == std.math.maxInt(usize)) return error.MissingLayerWeights;
        if (cfg.wq[layer_check] == std.math.maxInt(usize)) return error.MissingLayerWeights;
        if (cfg.wk[layer_check] == std.math.maxInt(usize)) return error.MissingLayerWeights;
        if (cfg.wv[layer_check] == std.math.maxInt(usize)) return error.MissingLayerWeights;
        if (cfg.wo[layer_check] == std.math.maxInt(usize)) return error.MissingLayerWeights;
        if (cfg.ffn_norm[layer_check] == std.math.maxInt(usize)) return error.MissingLayerWeights;
        if (cfg.ffn_gate[layer_check] == std.math.maxInt(usize)) return error.MissingLayerWeights;
        if (cfg.ffn_up[layer_check] == std.math.maxInt(usize)) return error.MissingLayerWeights;
        if (cfg.ffn_down[layer_check] == std.math.maxInt(usize)) return error.MissingLayerWeights;
    }

    return cfg;
}

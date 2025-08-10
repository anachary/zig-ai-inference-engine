const std = @import("std");
const core = @import("../core/api.zig");
const tok = @import("../tokenizers/gguf_vocab.zig");
const tmpl = @import("templates/mod.zig");
const sampling_factory = @import("../runtime/sampling/factory.zig");
const regs = @import("../core/registries.zig");
const types = @import("../core/types.zig");

pub const ChatSettings = struct {
    max_context: usize = 4096,
    max_tokens: usize = 256,
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.95,
    seed: u64 = 0,
};

pub const ChatSession = struct {
    allocator: std.mem.Allocator,
    rt: core.RuntimeSession,
    settings: ChatSettings,
    // Keep tokenized history to reuse KV cache; we append per turn
    ctx_tokens: []u32,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, settings: ChatSettings) !ChatSession {
        var rt = try core.loadModel(allocator, model_path);
        return .{ .allocator = allocator, .rt = rt, .settings = settings, .ctx_tokens = &[_]u32{} };
    }

    pub fn deinit(self: *ChatSession) void {
        // Free context tokens and runtime session resources
        if (self.ctx_tokens.len > 0) self.allocator.free(self.ctx_tokens);
        core.deinit(&self.rt, self.allocator);
    }

    fn ensure_room(self: *ChatSession, more: usize) void {
        // Simple truncation: if context would exceed max, drop from the front
        if (self.ctx_tokens.len + more > self.settings.max_context) {
            const overflow = self.ctx_tokens.len + more - self.settings.max_context;
            const keep = self.ctx_tokens.len - overflow;
            if (keep > 0) {
                // shift tokens left by overflow
                std.mem.copy(u32, self.ctx_tokens[0..keep], self.ctx_tokens[overflow..self.ctx_tokens.len]);
                self.ctx_tokens = self.ctx_tokens[0..keep];
            } else {
                self.ctx_tokens = self.ctx_tokens[0..0];
            }
        }
    }

    pub fn appendUser(self: *ChatSession, system_opt: ?[]const u8, user_text: []const u8) !void {
        // Build tokens for a new user turn per LLaMA‑2‑Chat template, without assistant reply
        var msgs = [_]tmpl.llama2.Message{
            .{ .role = .user, .content = user_text },
        };
        // Insert system once if this is the very first turn
        const sys: ?[]const u8 = if (self.ctx_tokens.len == 0) system_opt else null;
        const new_tokens = try tmpl.llama2.buildTokens(self.allocator, sys, &msgs);
        defer self.allocator.free(new_tokens);
        self.ensure_room(new_tokens.len);
        // Append to context
        const old_len = self.ctx_tokens.len;
        const total = old_len + new_tokens.len;
        var new_buf = try self.allocator.alloc(u32, total);
        if (old_len > 0) std.mem.copy(u32, new_buf[0..old_len], self.ctx_tokens);
        std.mem.copy(u32, new_buf[old_len..], new_tokens);
        if (old_len > 0) self.allocator.free(self.ctx_tokens);
        self.ctx_tokens = new_buf;
    }

    pub fn generate(self: *ChatSession, writer: anytype) !void {
        var gpa = self.allocator;

        // Seed context if empty using BOS when available
        if (self.ctx_tokens.len == 0) {
            if (tok.getBosId()) |bos| {
                const seeded = try gpa.alloc(u32, 1);
                seeded[0] = bos;
                self.ctx_tokens = seeded;
            } else return error.EmptyTokens;
        }

        // Allocate logits using the actual output weight shape if present
        var vocab_out: usize = self.rt.model.tokenizer.vocab_size;
        // Try to find "output.weight" tensor to get precise vocab size
        for (self.rt.model.tensors.items) |t| {
            if (std.mem.eql(u8, t.name, "output.weight")) {
                if (t.shape.len >= 2) vocab_out = t.shape[1];
                break;
            }
        }
        var logits = try gpa.alloc(f32, vocab_out);
        defer gpa.free(logits);

        // Build sampling pipeline from settings
        const params: types.SamplingParams = .{
            .temperature = self.settings.temperature,
            .top_k = self.settings.top_k,
            .top_p = self.settings.top_p,
            .seed = self.settings.seed,
        };
        const pipeline = try sampling_factory.buildPipeline(self.allocator, params);
        defer self.allocator.free(pipeline.transforms);
        var step: usize = 0;
        while (step < self.settings.max_tokens) : (step += 1) {
            if (core.forward(&self.rt, self.ctx_tokens, logits)) |_| {} else |err| {
                std.debug.print("ERROR: forward failed: {any}\n", .{err});
                return err;
            }
            // Apply transforms
            var sctx: regs.SamplerCtx = .{ .params = params, .recent = self.ctx_tokens, .rng = null };
            for (pipeline.transforms) |t| t.apply(logits, sctx);
            // Softmax for multinomial (harmless for greedy)
            var maxv: f32 = -3.4e38;
            for (logits) |v| {
                if (v > maxv) maxv = v;
            }
            var sum: f32 = 0;
            for (logits, 0..) |v, i| {
                const e = @exp(v - maxv);
                logits[i] = e;
                sum += e;
            }
            for (logits, 0..) |v, i| logits[i] = v / sum;
            // Select next token
            const best = pipeline.selector.select(logits, sctx);
            // Append token to context
            self.ensure_room(1);
            const resized = try gpa.realloc(self.ctx_tokens, self.ctx_tokens.len + 1);
            self.ctx_tokens = resized;
            self.ctx_tokens[self.ctx_tokens.len - 1] = best;
            // Stream detokenized piece
            const piece = try tok.detokenize(gpa, self.ctx_tokens[self.ctx_tokens.len - 1 .. self.ctx_tokens.len]);
            defer gpa.free(piece);
            try writer.writeAll(piece);
            // Stop if EOS known and matched
            if (tok.getEosId()) |eos| {
                if (best == eos) break;
            }
        }
    }
};

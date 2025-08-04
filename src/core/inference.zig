const std = @import("std");
const Model = @import("model.zig").Model;
const Tokenizer = @import("tokenizer.zig");
const TokenId = Tokenizer.TokenId;
const Tensor = @import("tensor.zig").DynamicTensor;

/// Inference configuration
pub const InferenceConfig = struct {
    max_tokens: u32 = 512,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repeat_penalty: f32 = 1.1,
    seed: ?u64 = null,

    pub fn init() InferenceConfig {
        return InferenceConfig{};
    }
};

/// Generation result
pub const GenerationResult = struct {
    tokens: []TokenId,
    text: []u8,
    logits: ?[]f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GenerationResult) void {
        self.allocator.free(self.tokens);
        self.allocator.free(self.text);
        if (self.logits) |logits| {
            self.allocator.free(logits);
        }
    }
};

/// Inference context for maintaining state
pub const InferenceContext = struct {
    kv_cache: ?[]Tensor,
    position: u32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) InferenceContext {
        return InferenceContext{
            .kv_cache = null,
            .position = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *InferenceContext) void {
        if (self.kv_cache) |cache| {
            for (cache) |*tensor| {
                tensor.deinit();
            }
            self.allocator.free(cache);
        }
    }

    pub fn reset(self: *InferenceContext) void {
        self.position = 0;
        // Keep KV cache allocated but reset position
    }
};

/// Universal inference interface
pub const Inference = struct {
    model: *Model,
    tokenizer: *Tokenizer.Tokenizer,
    config: InferenceConfig,
    context: InferenceContext,
    allocator: std.mem.Allocator,

    // Virtual function table for architecture-specific operations
    vtable: *const VTable,
    impl: *anyopaque,

    pub const VTable = struct {
        deinit: *const fn (impl: *anyopaque, allocator: std.mem.Allocator) void,
        forward: *const fn (impl: *anyopaque, tokens: []const TokenId, context: *InferenceContext) anyerror![]f32,
        sample: *const fn (impl: *anyopaque, logits: []const f32, config: *const InferenceConfig) anyerror!TokenId,
    };

    pub fn init(allocator: std.mem.Allocator, model: *Model, tokenizer: *Tokenizer.Tokenizer, config: InferenceConfig, vtable: *const VTable, impl: *anyopaque) Inference {
        return Inference{
            .model = model,
            .tokenizer = tokenizer,
            .config = config,
            .context = InferenceContext.init(allocator),
            .allocator = allocator,
            .vtable = vtable,
            .impl = impl,
        };
    }

    pub fn deinit(self: *Inference) void {
        self.context.deinit();
        self.vtable.deinit(self.impl, self.allocator);
    }

    /// Generate text from prompt
    pub fn generate(self: *Inference, prompt: []const u8) !GenerationResult {
        // Tokenize prompt
        var prompt_tokens = try self.tokenizer.encodeWithSpecial(prompt, true, false);
        defer prompt_tokens.deinit();

        // Prepare result storage
        var all_tokens = std.ArrayList(TokenId).init(self.allocator);
        defer all_tokens.deinit();

        // Add prompt tokens
        try all_tokens.appendSlice(prompt_tokens.tokens);

        // Reset context for new generation
        self.context.reset();

        // Generate tokens one by one
        var generated: u32 = 0;
        while (generated < self.config.max_tokens) {
            // Get current sequence
            const current_tokens = all_tokens.items;

            // Forward pass
            const logits = try self.vtable.forward(self.impl, current_tokens, &self.context);
            defer self.allocator.free(logits);

            // Sample next token
            const next_token = try self.vtable.sample(self.impl, logits, &self.config);

            // Check for EOS
            if (self.tokenizer.special_tokens.eos != null and next_token == self.tokenizer.special_tokens.eos.?) {
                break;
            }

            // Add token to sequence
            try all_tokens.append(next_token);
            generated += 1;
        }

        // Decode to text
        const text = try self.tokenizer.decode(all_tokens.items);

        // Create result
        return GenerationResult{
            .tokens = try all_tokens.toOwnedSlice(),
            .text = text,
            .logits = null,
            .allocator = self.allocator,
        };
    }

    /// Generate single token (for streaming)
    pub fn generateNext(self: *Inference, tokens: []const TokenId) !TokenId {
        const logits = try self.vtable.forward(self.impl, tokens, &self.context);
        defer self.allocator.free(logits);

        return self.vtable.sample(self.impl, logits, &self.config);
    }

    /// Get logits for tokens without sampling
    pub fn getLogits(self: *Inference, tokens: []const TokenId) ![]f32 {
        return self.vtable.forward(self.impl, tokens, &self.context);
    }

    /// Update configuration
    pub fn updateConfig(self: *Inference, new_config: InferenceConfig) void {
        self.config = new_config;
    }

    /// Reset inference state
    pub fn reset(self: *Inference) void {
        self.context.reset();
    }
};

/// Sampling utilities
pub const Sampling = struct {
    /// Apply temperature scaling
    pub fn applyTemperature(logits: []f32, temperature: f32) void {
        if (temperature <= 0.0) return;

        for (logits) |*logit| {
            logit.* /= temperature;
        }
    }

    /// Apply top-k filtering
    pub fn applyTopK(allocator: std.mem.Allocator, logits: []f32, k: u32) !void {
        if (k >= logits.len) return;

        // Create indices array
        var indices = try allocator.alloc(u32, logits.len);
        defer allocator.free(indices);

        for (indices, 0..) |*idx, i| {
            idx.* = @intCast(i);
        }

        // Sort by logits (descending)
        std.sort.heap(u32, indices, logits, struct {
            fn lessThan(context: []f32, a: u32, b: u32) bool {
                return context[a] > context[b];
            }
        }.lessThan);

        // Zero out logits beyond top-k
        for (indices[k..]) |idx| {
            logits[idx] = -std.math.inf(f32);
        }
    }

    /// Apply top-p (nucleus) filtering
    pub fn applyTopP(allocator: std.mem.Allocator, logits: []f32, p: f32) !void {
        if (p >= 1.0) return;

        // Convert to probabilities
        const max_logit = std.mem.max(f32, logits);
        var sum: f32 = 0.0;

        for (logits) |*logit| {
            logit.* = @exp(logit.* - max_logit);
            sum += logit.*;
        }

        for (logits) |*logit| {
            logit.* /= sum;
        }

        // Create sorted indices
        var indices = try allocator.alloc(u32, logits.len);
        defer allocator.free(indices);

        for (indices, 0..) |*idx, i| {
            idx.* = @intCast(i);
        }

        std.sort.heap(u32, indices, logits, struct {
            fn lessThan(context: []f32, a: u32, b: u32) bool {
                return context[a] > context[b];
            }
        }.lessThan);

        // Find cutoff point
        var cumulative: f32 = 0.0;
        var cutoff: usize = logits.len;

        for (indices, 0..) |idx, i| {
            cumulative += logits[idx];
            if (cumulative >= p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out logits beyond cutoff
        for (indices[cutoff..]) |idx| {
            logits[idx] = 0.0;
        }

        // Convert back to log space
        for (logits) |*logit| {
            if (logit.* > 0.0) {
                logit.* = @log(logit.*);
            } else {
                logit.* = -std.math.inf(f32);
            }
        }
    }

    /// Multinomial sampling
    pub fn sampleMultinomial(logits: []const f32, rng: std.Random) u32 {
        // Convert to probabilities
        const max_logit = std.mem.max(f32, logits);
        var probs = std.ArrayList(f32).init(std.heap.page_allocator);
        defer probs.deinit();

        var sum: f32 = 0.0;
        for (logits) |logit| {
            const prob = @exp(logit - max_logit);
            probs.append(prob) catch unreachable;
            sum += prob;
        }

        // Normalize
        for (probs.items) |*prob| {
            prob.* /= sum;
        }

        // Sample
        const r = rng.float(f32);
        var cumulative: f32 = 0.0;

        for (probs.items, 0..) |prob, i| {
            cumulative += prob;
            if (r <= cumulative) {
                return @intCast(i);
            }
        }

        return @intCast(probs.items.len - 1);
    }
};

test "inference config" {
    const testing = std.testing;

    var config = InferenceConfig.init();
    config.temperature = 0.8;
    config.max_tokens = 256;

    try testing.expect(config.temperature == 0.8);
    try testing.expect(config.max_tokens == 256);
}

test "sampling temperature" {
    const testing = std.testing;

    var logits = [_]f32{ 1.0, 2.0, 3.0 };
    Sampling.applyTemperature(&logits, 2.0);

    try testing.expect(logits[0] == 0.5);
    try testing.expect(logits[1] == 1.0);
    try testing.expect(logits[2] == 1.5);
}

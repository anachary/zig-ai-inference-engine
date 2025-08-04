const std = @import("std");

/// Complete model architecture definitions
/// Contains full model implementations like Transformer, GPT, LLaMA, etc.

// Model architectures
pub const transformer = @import("transformer.zig");
pub const qwen = @import("qwen.zig");

// Model configuration types
pub const ModelConfig = struct {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f32 = 1e-5,
    dropout: f32 = 0.0,

    // Architecture-specific parameters
    rope_theta: f32 = 10000.0,
    rope_scaling: ?RopeScaling = null,
    attention_bias: bool = false,
    mlp_bias: bool = false,
    tie_word_embeddings: bool = false,
};

pub const RopeScaling = struct {
    type: enum { linear, dynamic },
    factor: f32,
};

pub const ModelType = enum {
    transformer,
    gpt2,
    gpt_neox,
    llama,
    llama2,
    qwen,
    qwen2,
    mistral,
    mixtral,

    pub fn getDefaultConfig(self: ModelType) ModelConfig {
        return switch (self) {
            .transformer => ModelConfig{
                .vocab_size = 50257,
                .hidden_size = 768,
                .num_layers = 12,
                .num_heads = 12,
                .intermediate_size = 3072,
                .max_position_embeddings = 1024,
            },
            .gpt2 => ModelConfig{
                .vocab_size = 50257,
                .hidden_size = 768,
                .num_layers = 12,
                .num_heads = 12,
                .intermediate_size = 3072,
                .max_position_embeddings = 1024,
            },
            .llama, .llama2 => ModelConfig{
                .vocab_size = 32000,
                .hidden_size = 4096,
                .num_layers = 32,
                .num_heads = 32,
                .intermediate_size = 11008,
                .max_position_embeddings = 2048,
                .rope_theta = 10000.0,
            },
            .qwen, .qwen2 => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 896,
                .num_layers = 24,
                .num_heads = 14,
                .intermediate_size = 4864,
                .max_position_embeddings = 32768,
                .rope_theta = 1000000.0,
            },
            .mistral => ModelConfig{
                .vocab_size = 32000,
                .hidden_size = 4096,
                .num_layers = 32,
                .num_heads = 32,
                .intermediate_size = 14336,
                .max_position_embeddings = 32768,
                .rope_theta = 10000.0,
            },
            else => ModelConfig{
                .vocab_size = 50257,
                .hidden_size = 768,
                .num_layers = 12,
                .num_heads = 12,
                .intermediate_size = 3072,
                .max_position_embeddings = 1024,
            },
        };
    }
};

/// Base model interface that all models implement
pub const Model = struct {
    const Self = @This();

    forward_fn: *const fn (self: *anyopaque, input_ids: []const u32, output: []f32, allocator: std.mem.Allocator) anyerror!void,
    deinit_fn: *const fn (self: *anyopaque, allocator: std.mem.Allocator) void,
    get_config_fn: *const fn (self: *anyopaque) ModelConfig,
    impl: *anyopaque,

    pub fn forward(self: *Self, input_ids: []const u32, output: []f32, allocator: std.mem.Allocator) !void {
        return self.forward_fn(self.impl, input_ids, output, allocator);
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.deinit_fn(self.impl, allocator);
    }

    pub fn getConfig(self: *Self) ModelConfig {
        return self.get_config_fn(self.impl);
    }
};

/// Create a model from any implementation that follows the model interface
pub fn createModel(allocator: std.mem.Allocator, implementation: anytype) !Model {
    const T = @TypeOf(implementation);
    const impl_ptr = try allocator.create(T);
    impl_ptr.* = implementation;

    const vtable = struct {
        fn forward(impl: *anyopaque, input_ids: []const u32, output: []f32, alloc: std.mem.Allocator) anyerror!void {
            const self: *T = @ptrCast(@alignCast(impl));
            return self.forward(input_ids, output, alloc);
        }

        fn deinit(impl: *anyopaque, alloc: std.mem.Allocator) void {
            const self: *T = @ptrCast(@alignCast(impl));
            self.deinit();
            alloc.destroy(self);
        }

        fn getConfig(impl: *anyopaque) ModelConfig {
            const self: *T = @ptrCast(@alignCast(impl));
            return self.getConfig();
        }
    };

    return Model{
        .forward_fn = vtable.forward,
        .deinit_fn = vtable.deinit,
        .get_config_fn = vtable.getConfig,
        .impl = impl_ptr,
    };
}

test "models module" {
    const testing = std.testing;

    // Test default configs
    const gpt2_config = ModelType.gpt2.getDefaultConfig();
    try testing.expect(gpt2_config.vocab_size == 50257);
    try testing.expect(gpt2_config.hidden_size == 768);

    const qwen2_config = ModelType.qwen2.getDefaultConfig();
    try testing.expect(qwen2_config.vocab_size == 151936);
    try testing.expect(qwen2_config.hidden_size == 896);
}

const std = @import("std");
const ModelConfig = @import("mod.zig").ModelConfig;
const TransformerModel = @import("transformer.zig").TransformerModel;

/// Qwen model implementation
/// Specific configuration and optimizations for Qwen/Qwen2 models
pub const QwenModel = struct {
    transformer: TransformerModel,

    pub fn init(allocator: std.mem.Allocator, model_type: QwenType) !QwenModel {
        const config = model_type.getConfig();
        const transformer = try TransformerModel.init(allocator, config);

        return QwenModel{
            .transformer = transformer,
        };
    }

    pub fn deinit(self: *QwenModel) void {
        self.transformer.deinit();
    }

    pub fn loadWeights(self: *QwenModel, model: *@import("../core/model.zig").Model) !void {
        return self.transformer.loadWeights(model);
    }

    pub fn forward(self: *QwenModel, input_ids: []const u32, output: []f32, allocator: std.mem.Allocator) !void {
        return self.transformer.forward(input_ids, output, allocator);
    }

    pub fn getConfig(self: *QwenModel) ModelConfig {
        return self.transformer.getConfig();
    }
};

pub const QwenType = enum {
    qwen_0_5b,
    qwen_1_8b,
    qwen_4b,
    qwen_7b,
    qwen_14b,
    qwen_72b,
    qwen2_0_5b,
    qwen2_1_5b,
    qwen2_7b,
    qwen2_72b,

    pub fn getConfig(self: QwenType) ModelConfig {
        return switch (self) {
            .qwen_0_5b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 1024,
                .num_layers = 24,
                .num_heads = 16,
                .intermediate_size = 2816,
                .max_position_embeddings = 8192,
                .rope_theta = 10000.0,
            },
            .qwen_1_8b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 2048,
                .num_layers = 24,
                .num_heads = 16,
                .intermediate_size = 5504,
                .max_position_embeddings = 8192,
                .rope_theta = 10000.0,
            },
            .qwen_4b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 2560,
                .num_layers = 40,
                .num_heads = 20,
                .intermediate_size = 6912,
                .max_position_embeddings = 8192,
                .rope_theta = 10000.0,
            },
            .qwen_7b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 4096,
                .num_layers = 32,
                .num_heads = 32,
                .intermediate_size = 11008,
                .max_position_embeddings = 8192,
                .rope_theta = 10000.0,
            },
            .qwen_14b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 5120,
                .num_layers = 40,
                .num_heads = 40,
                .intermediate_size = 13696,
                .max_position_embeddings = 8192,
                .rope_theta = 10000.0,
            },
            .qwen_72b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 8192,
                .num_layers = 80,
                .num_heads = 64,
                .intermediate_size = 24576,
                .max_position_embeddings = 8192,
                .rope_theta = 10000.0,
            },
            .qwen2_0_5b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 896,
                .num_layers = 24,
                .num_heads = 14,
                .intermediate_size = 4864,
                .max_position_embeddings = 32768,
                .rope_theta = 1000000.0,
            },
            .qwen2_1_5b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 1536,
                .num_layers = 28,
                .num_heads = 12,
                .intermediate_size = 8960,
                .max_position_embeddings = 32768,
                .rope_theta = 1000000.0,
            },
            .qwen2_7b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 3584,
                .num_layers = 28,
                .num_heads = 28,
                .intermediate_size = 18944,
                .max_position_embeddings = 32768,
                .rope_theta = 1000000.0,
            },
            .qwen2_72b => ModelConfig{
                .vocab_size = 151936,
                .hidden_size = 8192,
                .num_layers = 80,
                .num_heads = 64,
                .intermediate_size = 29568,
                .max_position_embeddings = 32768,
                .rope_theta = 1000000.0,
            },
        };
    }

    pub fn fromModelName(model_name: []const u8) ?QwenType {
        if (std.mem.indexOf(u8, model_name, "Qwen2-0.5B") != null) return .qwen2_0_5b;
        if (std.mem.indexOf(u8, model_name, "Qwen2-1.5B") != null) return .qwen2_1_5b;
        if (std.mem.indexOf(u8, model_name, "Qwen2-7B") != null) return .qwen2_7b;
        if (std.mem.indexOf(u8, model_name, "Qwen2-72B") != null) return .qwen2_72b;
        if (std.mem.indexOf(u8, model_name, "Qwen-0.5B") != null) return .qwen_0_5b;
        if (std.mem.indexOf(u8, model_name, "Qwen-1.8B") != null) return .qwen_1_8b;
        if (std.mem.indexOf(u8, model_name, "Qwen-4B") != null) return .qwen_4b;
        if (std.mem.indexOf(u8, model_name, "Qwen-7B") != null) return .qwen_7b;
        if (std.mem.indexOf(u8, model_name, "Qwen-14B") != null) return .qwen_14b;
        if (std.mem.indexOf(u8, model_name, "Qwen-72B") != null) return .qwen_72b;
        return null;
    }

    pub fn name(self: QwenType) []const u8 {
        return switch (self) {
            .qwen_0_5b => "Qwen-0.5B",
            .qwen_1_8b => "Qwen-1.8B",
            .qwen_4b => "Qwen-4B",
            .qwen_7b => "Qwen-7B",
            .qwen_14b => "Qwen-14B",
            .qwen_72b => "Qwen-72B",
            .qwen2_0_5b => "Qwen2-0.5B",
            .qwen2_1_5b => "Qwen2-1.5B",
            .qwen2_7b => "Qwen2-7B",
            .qwen2_72b => "Qwen2-72B",
        };
    }
};

test "qwen model configs" {
    const testing = std.testing;

    // Test Qwen2-0.5B config (our main test model)
    const qwen2_0_5b_config = QwenType.qwen2_0_5b.getConfig();
    try testing.expect(qwen2_0_5b_config.vocab_size == 151936);
    try testing.expect(qwen2_0_5b_config.hidden_size == 896);
    try testing.expect(qwen2_0_5b_config.num_layers == 24);
    try testing.expect(qwen2_0_5b_config.num_heads == 14);
    try testing.expect(qwen2_0_5b_config.intermediate_size == 4864);
    try testing.expect(qwen2_0_5b_config.max_position_embeddings == 32768);
    try testing.expect(qwen2_0_5b_config.rope_theta == 1000000.0);

    // Test model name detection
    try testing.expect(QwenType.fromModelName("Qwen2-0.5B-Instruct-Q4_K_M.gguf") == .qwen2_0_5b);
    try testing.expect(QwenType.fromModelName("Qwen2-7B-Chat") == .qwen2_7b);
    try testing.expect(QwenType.fromModelName("unknown-model") == null);
}

test "qwen model creation" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var qwen = try QwenModel.init(allocator, .qwen2_0_5b);
    defer qwen.deinit();

    const config = qwen.getConfig();
    try testing.expect(config.vocab_size == 151936);
    try testing.expect(config.hidden_size == 896);
}

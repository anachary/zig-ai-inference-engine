const std = @import("std");
const lib = @import("zig-ai-inference");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧠 Testing Real LLM Model Integration", .{});
    std.log.info("=====================================", .{});

    // Initialize text generator
    var text_generator = lib.llm.TextGenerator.init(allocator);
    defer text_generator.deinit();

    // Test different model types
    const test_models = [_][]const u8{
        "distilgpt2",
        "gpt2-small", 
        "phi2",
        "tinyllama",
    };

    for (test_models) |model_name| {
        std.log.info("", .{});
        std.log.info("🔄 Testing model: {s}", .{model_name});
        std.log.info("----------------------------", .{});

        // Try to load the model
        text_generator.loadModel(model_name) catch |err| {
            std.log.err("❌ Failed to load {s}: {}", .{ model_name, err });
            continue;
        };

        // Test generation with different prompts
        const test_prompts = [_][]const u8{
            "What is artificial intelligence?",
            "Explain machine learning in simple terms",
            "Write a short story about a robot",
            "How does neural network work?",
        };

        const config = lib.llm.GenerationConfig{
            .max_tokens = 100,
            .temperature = 0.7,
            .top_p = 0.9,
        };

        for (test_prompts) |prompt| {
            std.log.info("", .{});
            std.log.info("💬 Prompt: \"{s}\"", .{prompt});
            
            const response = text_generator.generate(prompt, config) catch |err| {
                std.log.err("❌ Generation failed: {}", .{err});
                continue;
            };
            defer allocator.free(response);

            std.log.info("🤖 Response: {s}", .{response});
        }

        // Test model information
        std.log.info("", .{});
        std.log.info("📊 Model Information:", .{});
        std.log.info("   Type: {s}", .{@tagName(text_generator.model_type)});
        std.log.info("   Vocab Size: {d}", .{text_generator.model_config.vocab_size});
        std.log.info("   Max Sequence: {d}", .{text_generator.model_config.max_sequence_length});
        std.log.info("   Embedding Dim: {d}", .{text_generator.model_config.embedding_dim});
    }

    std.log.info("", .{});
    std.log.info("✅ Real LLM testing completed!", .{});
}

const std = @import("std");
const lib = @import("zig-ai-inference");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 Real LLM Model Demo - Zig AI Inference Engine", .{});
    std.log.info("================================================", .{});

    // Get command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_name: []const u8 = "distilgpt2"; // Default model
    var interactive_mode = false;

    // Parse arguments
    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i < args.len) {
                model_name = args[i];
            }
        } else if (std.mem.eql(u8, arg, "--interactive")) {
            interactive_mode = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            printHelp();
            return;
        }
        i += 1;
    }

    std.log.info("🎯 Target model: {s}", .{model_name});
    std.log.info("🔧 Interactive mode: {}", .{interactive_mode});

    // Initialize model downloader
    var downloader = lib.models.ModelDownloader.init(allocator);

    // Show available models
    std.log.info("", .{});
    downloader.listOfficialModels();

    // Initialize text generator
    var text_generator = lib.llm.TextGenerator.init(allocator);
    defer text_generator.deinit();

    // Load the specified model
    std.log.info("", .{});
    std.log.info("🔄 Loading model: {s}", .{model_name});
    
    text_generator.loadModel(model_name) catch |err| {
        std.log.err("❌ Failed to load model: {}", .{err});
        std.log.info("💡 Try one of the available models listed above", .{});
        return;
    };

    std.log.info("✅ Model loaded successfully!", .{});

    // Show model information
    std.log.info("", .{});
    std.log.info("📊 Model Information:", .{});
    std.log.info("   Type: {s}", .{@tagName(text_generator.model_type)});
    std.log.info("   Vocabulary Size: {d}", .{text_generator.model_config.vocab_size});
    std.log.info("   Max Sequence Length: {d}", .{text_generator.model_config.max_sequence_length});
    std.log.info("   Embedding Dimension: {d}", .{text_generator.model_config.embedding_dim});
    std.log.info("   Memory Requirement: ~{d}MB", .{downloader.getModelMemoryRequirement(model_name)});

    const config = lib.llm.GenerationConfig{
        .max_tokens = 150,
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 50,
    };

    if (interactive_mode) {
        try runInteractiveMode(&text_generator, config, allocator);
    } else {
        try runDemoMode(&text_generator, config, allocator);
    }
}

fn printHelp() void {
    std.log.info("🧠 Real LLM Model Demo", .{});
    std.log.info("Usage: zig run examples/real_llm_demo.zig -- [options]", .{});
    std.log.info("", .{});
    std.log.info("Options:", .{});
    std.log.info("  --model <name>     Model to use (distilgpt2, gpt2-small, phi2, tinyllama)", .{});
    std.log.info("  --interactive      Run in interactive mode", .{});
    std.log.info("  --help             Show this help", .{});
    std.log.info("", .{});
    std.log.info("Examples:", .{});
    std.log.info("  zig run examples/real_llm_demo.zig -- --model distilgpt2", .{});
    std.log.info("  zig run examples/real_llm_demo.zig -- --model phi2 --interactive", .{});
}

fn runDemoMode(text_generator: *lib.llm.TextGenerator, config: lib.llm.GenerationConfig, allocator: std.mem.Allocator) !void {
    std.log.info("", .{});
    std.log.info("🎭 Running Demo Mode", .{});
    std.log.info("===================", .{});

    const demo_prompts = [_][]const u8{
        "What is the future of artificial intelligence?",
        "Explain quantum computing in simple terms",
        "Write a short poem about technology",
        "How do neural networks learn?",
        "What are the benefits of edge AI?",
    };

    for (demo_prompts, 0..) |prompt, i| {
        std.log.info("", .{});
        std.log.info("📝 Demo {d}/5: {s}", .{ i + 1, prompt });
        std.log.info("---", .{});

        const response = text_generator.generate(prompt, config) catch |err| {
            std.log.err("❌ Generation failed: {}", .{err});
            continue;
        };
        defer allocator.free(response);

        std.log.info("🤖 {s}", .{response});
        
        // Small delay for readability
        std.time.sleep(1_000_000_000); // 1 second
    }
}

fn runInteractiveMode(text_generator: *lib.llm.TextGenerator, config: lib.llm.GenerationConfig, allocator: std.mem.Allocator) !void {
    std.log.info("", .{});
    std.log.info("💬 Interactive Mode - Type 'quit' to exit", .{});
    std.log.info("=========================================", .{});

    const stdin = std.io.getStdIn().reader();
    
    while (true) {
        std.log.info("", .{});
        std.log.info("You: ", .{});
        
        // Read user input
        var input_buffer: [1024]u8 = undefined;
        if (try stdin.readUntilDelimiterOrEof(input_buffer[0..], '\n')) |input| {
            const trimmed_input = std.mem.trim(u8, input, " \t\r\n");
            
            if (std.mem.eql(u8, trimmed_input, "quit") or std.mem.eql(u8, trimmed_input, "exit")) {
                std.log.info("👋 Goodbye!", .{});
                break;
            }
            
            if (trimmed_input.len == 0) continue;
            
            // Generate response
            const response = text_generator.generate(trimmed_input, config) catch |err| {
                std.log.err("❌ Generation failed: {}", .{err});
                continue;
            };
            defer allocator.free(response);
            
            std.log.info("🤖 {s}", .{response});
        }
    }
}

const std = @import("std");
const zig_ai = @import("zig-ai-platform");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Zig AI Platform Demo ===\n", .{});
    std.debug.print("Version: {s}\n\n", .{zig_ai.version.string});

    // Test format detection
    std.debug.print("1. Testing format detection:\n", .{});

    const test_files = [_][]const u8{
        "model.gguf",
        "model.onnx",
        "model.safetensors",
        "model.pth",
        "unknown.xyz",
    };

    for (test_files) |filename| {
        const format = zig_ai.formats.registry.ModelFormat.fromExtension(std.fs.path.extension(filename));
        std.debug.print("  {s} -> {s}\n", .{ filename, format.toString() });
    }

    // Test tokenizer creation
    std.debug.print("\n2. Testing BPE tokenizer:\n", .{});

    var tokenizer = zig_ai.tokenizers.bpe.create(allocator, "dummy") catch |err| {
        std.debug.print("Error creating tokenizer: {}\n", .{err});
        return;
    };
    defer tokenizer.deinit();

    std.debug.print("  Tokenizer created successfully!\n", .{});
    std.debug.print("  Vocabulary size: {}\n", .{tokenizer.vocab_size});
    std.debug.print("  Special tokens: BOS={?}, EOS={?}, UNK={?}\n", .{
        tokenizer.special_tokens.bos,
        tokenizer.special_tokens.eos,
        tokenizer.special_tokens.unk,
    });

    // Test tokenization
    const test_texts = [_][]const u8{
        "hello",
        "world",
        "hello world!",
    };

    std.debug.print("\n3. Testing tokenization:\n", .{});
    for (test_texts) |text| {
        var result = tokenizer.encode(text) catch |err| {
            std.debug.print("  Error tokenizing '{s}': {}\n", .{ text, err });
            continue;
        };
        defer result.deinit();

        std.debug.print("  '{s}' -> {} tokens: [", .{ text, result.tokens.len });
        for (result.tokens, 0..) |token, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{}", .{token});
        }
        std.debug.print("]\n", .{});

        // Test decoding
        const decoded = tokenizer.decode(result.tokens) catch |err| {
            std.debug.print("    Error decoding: {}\n", .{err});
            continue;
        };
        defer allocator.free(decoded);

        std.debug.print("    Decoded back to: '{s}'\n", .{decoded});
    }

    // Test inference configuration
    std.debug.print("\n4. Testing inference configuration:\n", .{});

    var config = zig_ai.core.InferenceConfig.init();
    config.temperature = 0.8;
    config.max_tokens = 256;
    config.top_p = 0.95;
    config.top_k = 50;

    std.debug.print("  Temperature: {d:.2}\n", .{config.temperature});
    std.debug.print("  Max tokens: {}\n", .{config.max_tokens});
    std.debug.print("  Top-p: {d:.2}\n", .{config.top_p});
    std.debug.print("  Top-k: {}\n", .{config.top_k});

    // Test format capabilities
    std.debug.print("\n5. Testing format capabilities:\n", .{});

    const formats = zig_ai.formats.registry.getSupportedFormats();
    for (formats) |format| {
        const caps = zig_ai.formats.registry.getCapabilities(format);
        std.debug.print("  {s}:\n", .{format.toString()});
        std.debug.print("    Streaming: {}\n", .{caps.supports_streaming});
        std.debug.print("    Quantization: {}\n", .{caps.supports_quantization});
        std.debug.print("    Metadata: {}\n", .{caps.supports_metadata});
    }

    std.debug.print("\n=== Demo completed successfully! ===\n", .{});
    std.debug.print("\nTo test with a real model:\n", .{});
    std.debug.print("1. Download a GGUF model file\n", .{});
    std.debug.print("2. Place it in the models/ directory\n", .{});
    std.debug.print("3. Run: zig build run -- --model models/your-model.gguf\n", .{});
}

const std = @import("std");
const lib = @import("src/lib.zig");

const CliError = error{
    InvalidArguments,
    ModelNotFound,
    InferenceFailed,
};

const CliConfig = struct {
    model_path: []const u8 = "models/squeezenet.onnx",
    input_text: ?[]const u8 = null,
    help: bool = false,
};

fn printHelp() void {
    std.log.info("🧠 Zig AI Inference Engine CLI", .{});
    std.log.info("===============================", .{});
    std.log.info("", .{});
    std.log.info("Usage: zig-ai [options]", .{});
    std.log.info("", .{});
    std.log.info("Options:", .{});
    std.log.info("  --model <path>     Path to ONNX model file (default: models/squeezenet.onnx)", .{});
    std.log.info("  --input <text>     Input text for inference", .{});
    std.log.info("  --help             Show this help message", .{});
    std.log.info("", .{});
    std.log.info("Examples:", .{});
    std.log.info("  zig-ai --model models/squeezenet.onnx", .{});
    std.log.info("  zig-ai --model models/distilgpt2.onnx --input \"Hello world\"", .{});
    std.log.info("", .{});
    std.log.info("Available models:", .{});
    std.log.info("  - models/squeezenet.onnx (Image classification, working)", .{});
    std.log.info("  - models/distilgpt2.onnx (Text generation, needs advanced parser)", .{});
    std.log.info("  - models/gpt2_fp16.onnx (Text generation, needs advanced parser)", .{});
}

fn parseArgs(allocator: std.mem.Allocator, args: [][]const u8) !CliConfig {
    _ = allocator; // Not used in this simple implementation
    var config = CliConfig{};

    var i: usize = 1; // Skip program name
    while (i < args.len) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            config.help = true;
        } else if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return CliError.InvalidArguments;
            config.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--input")) {
            i += 1;
            if (i >= args.len) return CliError.InvalidArguments;
            config.input_text = args[i];
        } else {
            std.log.err("Unknown argument: {s}", .{arg});
            return CliError.InvalidArguments;
        }

        i += 1;
    }

    return config;
}

fn runInference(allocator: std.mem.Allocator, config: CliConfig) !void {
    std.log.info("🚀 Starting Zig AI Inference Engine", .{});
    std.log.info("===================================", .{});
    std.log.info("", .{});

    // Initialize the inference engine
    std.log.info("🔧 Initializing inference engine...", .{});
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 1024,
        .num_threads = null,
        .enable_profiling = false,
        .tensor_pool_size = 100,
    });
    defer engine.deinit();
    std.log.info("✅ Engine initialized successfully", .{});
    std.log.info("", .{});

    // Load the model
    std.log.info("📥 Loading model: {s}", .{config.model_path});
    const model_result = engine.loadModel(config.model_path);
    if (model_result) |_| {
        std.log.info("✅ Model loaded successfully!", .{});
        std.log.info("", .{});

        // Display model information
        std.log.info("📊 Model Information:", .{});
        std.log.info("   📁 Path: {s}", .{config.model_path});
        std.log.info("   🧠 Format: ONNX", .{});
        std.log.info("   ✅ Status: Ready for inference", .{});
        std.log.info("", .{});

        if (config.input_text) |input| {
            std.log.info("🔍 Running inference with input: \"{s}\"", .{input});
            std.log.info("", .{});

            // TODO: Implement actual inference with input
            // For now, just show that the model is loaded and ready
            std.log.info("⚠️  Note: Actual inference execution is not yet implemented.", .{});
            std.log.info("   The model is successfully loaded and parsed.", .{});
            std.log.info("   Next step: Implement tensor input/output handling.", .{});
        } else {
            std.log.info("ℹ️  Model loaded successfully. Use --input to run inference.", .{});
        }

        std.log.info("", .{});
        std.log.info("🎉 Inference engine test completed successfully!", .{});
    } else |err| {
        std.log.err("❌ Failed to load model: {}", .{err});
        std.log.err("", .{});
        std.log.err("💡 Troubleshooting:", .{});
        std.log.err("   - Check if the model file exists", .{});
        std.log.err("   - Ensure the model is in ONNX format", .{});
        std.log.err("   - For text models, advanced protobuf features may be needed", .{});
        return err;
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse arguments
    const config = parseArgs(allocator, args) catch |err| {
        switch (err) {
            CliError.InvalidArguments => {
                std.log.err("❌ Invalid arguments. Use --help for usage information.", .{});
                return;
            },
            else => return err,
        }
    };

    // Show help if requested
    if (config.help) {
        printHelp();
        return;
    }

    // Run inference
    try runInference(allocator, config);
}

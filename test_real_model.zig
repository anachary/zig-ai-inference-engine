const std = @import("std");
const onnx_parser = @import("zig-onnx-parser");
const inference_engine = @import("zig-inference-engine");
const tensor_core = @import("zig-tensor-core");

/// Test real ONNX model loading and inference
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧪 Testing Real ONNX Model Loading and Inference", .{});
    std.log.info("=" ** 50, .{});

    // Test model paths
    const test_models = [_][]const u8{
        "models/minimal_test.onnx",
    };

    var successful_tests: usize = 0;

    for (test_models) |model_path| {
        std.log.info("\n🔍 Testing model: {s}", .{model_path});

        if (testModelLoading(allocator, model_path)) {
            std.log.info("✅ {s} - SUCCESS", .{model_path});
            successful_tests += 1;
        } else |err| {
            std.log.err("❌ {s} - FAILED: {any}", .{ model_path, err });
        }
    }

    std.log.info("\n📊 Test Results: {}/{} models loaded successfully", .{ successful_tests, test_models.len });

    if (successful_tests > 0) {
        std.log.info("🎉 Real ONNX model loading is working!", .{});
        std.log.info("🚀 Ready for real inference implementation", .{});
    } else {
        std.log.err("❌ No models loaded successfully", .{});
        std.log.info("💡 Try running: python download_test_model.py", .{});
    }
}

/// Test loading a specific ONNX model
fn testModelLoading(allocator: std.mem.Allocator, model_path: []const u8) !void {
    // Check if file exists
    const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
        switch (err) {
            error.FileNotFound => {
                std.log.warn("📁 File not found: {s}", .{model_path});
                std.log.info("💡 Run: python download_test_model.py", .{});
                return err;
            },
            else => return err,
        }
    };
    defer file.close();

    const file_size = try file.getEndPos();
    std.log.info("📊 File size: {d:.2} MB", .{@as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0)});

    // Initialize ONNX parser
    var parser = onnx_parser.Parser.init(allocator);

    // Parse the model
    std.log.info("🔍 Parsing ONNX model...", .{});
    var model = parser.parseFile(model_path) catch |err| {
        std.log.err("❌ ONNX parsing failed: {any}", .{err});
        return err;
    };
    defer model.deinit();

    // Get model metadata
    const metadata = model.getMetadata();
    std.log.info("📋 Model Info:", .{});
    std.log.info("   Name: {s}", .{metadata.name});
    std.log.info("   Version: {s}", .{metadata.version});
    std.log.info("   Format: {any}", .{metadata.format});
    std.log.info("   Inputs: {}", .{metadata.input_count});
    std.log.info("   Outputs: {}", .{metadata.output_count});
    std.log.info("   Parameters: {}", .{metadata.parameter_count});

    // Validate model
    model.validate() catch |err| {
        std.log.err("❌ Model validation failed: {any}", .{err});
        return err;
    };
    std.log.info("✅ Model validation passed", .{});

    // Check model structure
    std.log.info("🔍 Analyzing model structure...", .{});
    const inputs = model.getInputs();
    const outputs = model.getOutputs();

    std.log.info("✅ Model has {} inputs and {} outputs", .{ inputs.len, outputs.len });

    // Show input/output info
    for (inputs, 0..) |input, i| {
        if (i >= 3) break; // Show first 3 inputs
        std.log.info("   Input {}: {s}", .{ i, input.name });
    }

    for (outputs, 0..) |output, i| {
        if (i >= 3) break; // Show first 3 outputs
        std.log.info("   Output {}: {s}", .{ i, output.name });
    }

    // Test inference engine integration
    std.log.info("🔍 Testing inference engine integration...", .{});
    try testInferenceEngineIntegration(allocator, &model);

    std.log.info("✅ Model loading test completed successfully", .{});
}

/// Test integration with inference engine
fn testInferenceEngineIntegration(allocator: std.mem.Allocator, model: *const onnx_parser.Model) !void {
    // Initialize inference engine
    const engine_config = inference_engine.Config{
        .device_type = .cpu,
        .num_threads = 2,
        .enable_gpu = false,
        .optimization_level = .balanced,
        .memory_limit_mb = 1024,
    };

    var engine = try inference_engine.Engine.init(allocator, engine_config);
    defer engine.deinit();

    // Create model interface
    const model_interface = createRealONNXModelInterface(model);

    // Load model into engine
    try engine.loadModel(@as(*anyopaque, @ptrCast(@constCast(model))), model_interface);

    // Verify engine state
    const stats = engine.getStats();
    if (!stats.model_loaded) {
        return error.ModelNotLoaded;
    }

    std.log.info("✅ Model loaded into inference engine", .{});
    std.log.info("📊 Engine stats: {} total inferences", .{stats.total_inferences});
}

/// Create a real ONNX model interface for the inference engine
fn createRealONNXModelInterface(model: *const onnx_parser.Model) inference_engine.ModelInterface {
    _ = model; // Will be used in the implementation

    const model_impl = inference_engine.ModelImpl{
        .validateFn = realONNXValidate,
        .getMetadataFn = realONNXGetMetadata,
        .freeFn = realONNXFree,
    };

    return inference_engine.ModelInterface{
        .ctx = undefined,
        .impl = &model_impl,
    };
}

/// Real ONNX model validation function
fn realONNXValidate(ctx: *anyopaque, model_ptr: *anyopaque) !void {
    _ = ctx;
    const model = @as(*const onnx_parser.Model, @ptrCast(@alignCast(model_ptr)));

    // Validate the real ONNX model
    try model.validate();
    std.log.info("✅ Real ONNX model validation successful", .{});
}

/// Real ONNX model metadata function
fn realONNXGetMetadata(ctx: *anyopaque) !inference_engine.ModelMetadata {
    _ = ctx;
    // Return basic metadata for now
    return inference_engine.ModelMetadata{
        .name = "Real ONNX Model",
        .input_count = 1,
        .output_count = 1,
    };
}

/// Real ONNX model cleanup function
fn realONNXFree(ctx: *anyopaque, model_ptr: *anyopaque) void {
    _ = ctx;
    _ = model_ptr;
    // Model cleanup is handled by the caller
    std.log.info("🧹 Real ONNX model cleanup called", .{});
}

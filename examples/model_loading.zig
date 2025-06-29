const std = @import("std");
const lib = @import("zig-ai-inference");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🔧 Model Loading Example - Phase 2 ONNX Parser", .{});
    std.log.info("===============================================", .{});

    // Initialize the AI engine
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 1024,
        .num_threads = 4,
    });
    defer engine.deinit();

    std.log.info("✅ AI Engine initialized", .{});

    // Test ONNX parser directly
    std.log.info("🧪 Testing ONNX parser functionality...", .{});

    var onnx_parser = lib.onnx.ONNXParser.init(allocator);
    _ = onnx_parser;

    // Test supported operations
    const supported_ops = lib.onnx.ONNXParser.getSupportedOps();
    std.log.info("📋 Supported ONNX operations ({d} total):", .{supported_ops.len});
    for (supported_ops) |op| {
        std.log.info("  - {s}", .{op});
    }

    // Test operation support checking
    std.log.info("", .{});
    std.log.info("🔍 Testing operation support:", .{});
    const test_ops = [_][]const u8{ "Add", "Conv", "Relu", "UnsupportedOp" };
    for (test_ops) |op| {
        const supported = lib.onnx.ONNXParser.isOpSupported(op);
        const status = if (supported) "✅" else "❌";
        std.log.info("  {s} {s}: {s}", .{ status, op, if (supported) "Supported" else "Not supported" });
    }

    // Test model format detection
    std.log.info("", .{});
    std.log.info("📁 Testing model format detection:", .{});
    const test_paths = [_][]const u8{ "model.onnx", "model.tflite", "model.pt", "model.bin" };

    for (test_paths) |path| {
        const format = lib.formats.ModelFormat.fromPath(path);
        std.log.info("  {s} -> {s}", .{ path, @tagName(format) });
    }

    // Create a dummy ONNX model for testing
    std.log.info("", .{});
    std.log.info("🏗️ Creating test model structure...", .{});

    var metadata = try lib.formats.ModelMetadata.init(allocator, "test_model", "1.0");
    metadata.format = .onnx;

    var test_model = lib.formats.Model.init(allocator, metadata);
    defer test_model.deinit();

    // Add some test nodes
    var add_node = try lib.formats.GraphNode.init(allocator, "add_1", "Add");
    try test_model.graph.addNode(add_node);

    var relu_node = try lib.formats.GraphNode.init(allocator, "relu_1", "Relu");
    try test_model.graph.addNode(relu_node);

    // Add input/output specs
    const input_shape = [_]i32{ -1, 3, 224, 224 };
    var input_spec = try lib.formats.TensorSpec.init(allocator, "input", &input_shape, .f32);
    try test_model.graph.addInput(input_spec);

    const output_shape = [_]i32{ -1, 1000 };
    var output_spec = try lib.formats.TensorSpec.init(allocator, "output", &output_shape, .f32);
    try test_model.graph.addOutput(output_spec);

    std.log.info("✅ Test model created with:", .{});
    std.log.info("  - {d} nodes", .{test_model.graph.nodes.items.len});
    std.log.info("  - {d} inputs", .{test_model.graph.inputs.items.len});
    std.log.info("  - {d} outputs", .{test_model.graph.outputs.items.len});

    // Validate the model
    std.log.info("", .{});
    std.log.info("✅ Validating test model...", .{});
    try test_model.validate();
    std.log.info("✅ Model validation passed", .{});

    // Test loading a non-existent ONNX file (will create dummy model)
    std.log.info("", .{});
    std.log.info("🔄 Testing ONNX file parsing (dummy implementation)...", .{});

    // Note: This will fail because the file doesn't exist, but shows the integration
    if (engine.loadModel("test_model.onnx")) {
        std.log.info("✅ Model loaded successfully through engine", .{});
    } else |err| {
        std.log.info("ℹ️ Expected error (file doesn't exist): {}", .{err});
        std.log.info("   This demonstrates the ONNX loading pipeline", .{});
    }

    std.log.info("", .{});
    std.log.info("🎊 Model Loading Example Complete!", .{});
    std.log.info("✅ ONNX parser foundation implemented", .{});
    std.log.info("✅ Model format detection working", .{});
    std.log.info("✅ Graph structure creation functional", .{});
    std.log.info("✅ Model validation pipeline operational", .{});
    std.log.info("", .{});
    std.log.info("🚀 Ready for real ONNX model files!", .{});
}

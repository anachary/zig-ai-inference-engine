const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🔍 Testing ONNX parser import...", .{});

    // Try to import the ONNX parser
    const onnx_parser = @import("zig-onnx-parser");
    
    std.log.info("✅ ONNX parser imported successfully!", .{});
    
    // Try to create a parser config
    const parser_config = onnx_parser.ParserConfig{
        .max_model_size_mb = 1024,
        .strict_validation = false,
        .skip_unknown_ops = true,
        .verbose_logging = false, // Reduce noise
    };

    std.log.info("✅ Parser config created!", .{});

    // Try to create a parser
    var parser = onnx_parser.Parser.initWithConfig(allocator, parser_config);
    
    std.log.info("✅ ONNX parser created successfully!", .{});
    
    // Try to parse a model
    const model_path = "models/model_fp16.onnx";
    std.log.info("🔍 Attempting to parse: {s}", .{model_path});
    
    const model = parser.parseFile(model_path) catch |err| {
        std.log.err("❌ Failed to parse ONNX model: {}", .{err});
        std.log.err("This explains why the CLI is falling back to basic analysis", .{});
        return;
    };
    
    std.log.info("✅ Model parsed successfully!", .{});
    
    // Get model information
    const metadata = model.getMetadata();
    const inputs = model.getInputs();
    const outputs = model.getOutputs();
    
    std.log.info("📊 Model Information:", .{});
    std.log.info("  - Name: {s}", .{metadata.name});
    std.log.info("  - Producer: {s} v{s}", .{metadata.producer_name, metadata.producer_version});
    std.log.info("  - IR Version: {}", .{metadata.ir_version});
    std.log.info("  - Inputs: {}", .{inputs.len});
    std.log.info("  - Outputs: {}", .{outputs.len});
    
    if (inputs.len > 0) {
        std.log.info("🔍 Input Details:", .{});
        for (inputs, 0..) |input, i| {
            std.log.info("  {}. {s}: shape={any}, type={s}", .{i+1, input.name, input.shape, @tagName(input.dtype)});
        }
    }
}

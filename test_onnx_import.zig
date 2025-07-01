const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("Testing ONNX parser import...", .{});

    // Try to import the ONNX parser
    const onnx_parser = @import("zig-onnx-parser");
    
    std.log.info("✅ ONNX parser imported successfully!", .{});
    
    // Try to create a parser
    const parser_config = onnx_parser.ParserConfig{
        .max_model_size_mb = 1024,
        .strict_validation = false,
        .skip_unknown_ops = true,
        .verbose_logging = true,
    };

    var parser = onnx_parser.Parser.initWithConfig(allocator, parser_config);
    
    std.log.info("✅ ONNX parser created successfully!", .{});
    
    // Try to parse a model
    const model_path = "models/model_fp16.onnx";
    std.log.info("🔍 Attempting to parse: {s}", .{model_path});
    
    const model = parser.parseFile(model_path) catch |err| {
        std.log.err("❌ Failed to parse ONNX model: {}", .{err});
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
}

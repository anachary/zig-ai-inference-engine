const std = @import("std");
const print = std.debug.print;

// Import ONNX parser
const onnx_parser = @import("projects/zig-onnx-parser/src/parser.zig");
const model_format = @import("projects/zig-onnx-parser/src/formats/model.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🧪 Testing Real ONNX File Loading\n");
    print("================================\n\n");

    // Test with your real ONNX model
    const model_path = "models/model_fp16.onnx";
    
    print("📂 Testing file: {s}\n", .{model_path});
    
    // Check if file exists
    const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
        print("❌ Failed to open file: {}\n", .{err});
        print("💡 Make sure the file exists at: {s}\n", .{model_path});
        return;
    };
    defer file.close();
    
    // Get file size
    const file_size = try file.getEndPos();
    print("📊 File size: {d:.2} MB\n", .{@as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0)});
    
    // Read file content
    const file_data = try allocator.alloc(u8, file_size);
    defer allocator.free(file_data);
    
    _ = try file.readAll(file_data);
    print("✅ File read successfully\n\n");
    
    // Test format detection
    print("🔍 Testing Format Detection\n");
    print("---------------------------\n");
    
    const detected_format = model_format.ModelFormat.detectFormat(model_path, file_data);
    print("Detected format: {s}\n", .{detected_format.toString()});
    
    if (detected_format != .onnx) {
        print("⚠️  Expected ONNX format, got {s}\n", .{detected_format.toString()});
    } else {
        print("✅ Format detection successful\n");
    }
    
    print("\n🔧 Testing ONNX Parser\n");
    print("----------------------\n");
    
    // Create ONNX parser with relaxed validation
    var parser = onnx_parser.ONNXParser.init(allocator, .{
        .strict_validation = false,  // Allow experimental models
        .skip_unknown_ops = true,    // Skip unsupported operators
        .max_file_size = 1024 * 1024 * 1024, // 1GB limit
    });
    defer parser.deinit();
    
    // Try to parse the ONNX model
    const parsed_model = parser.parseFile(model_path) catch |err| {
        print("❌ Failed to parse ONNX model: {}\n", .{err});
        print("\n💡 Common issues and solutions:\n");
        print("   1. Model uses unsupported ONNX operators\n");
        print("   2. Model has complex subgraphs or control flow\n");
        print("   3. Model uses newer ONNX features\n");
        print("   4. Protobuf parsing limitations\n");
        print("\n🔧 Possible fixes:\n");
        print("   - Use a simpler model for testing\n");
        print("   - Convert model to older ONNX version\n");
        print("   - Implement missing operators\n");
        return;
    };
    
    print("✅ ONNX model parsed successfully!\n\n");
    
    // Print model information
    print("📋 Model Information\n");
    print("-------------------\n");
    const metadata = parsed_model.getMetadata();
    print("Name: {s}\n", .{metadata.name});
    print("Version: {s}\n", .{metadata.version});
    print("Producer: {s} v{s}\n", .{metadata.producer_name, metadata.producer_version});
    print("IR Version: {}\n", .{metadata.ir_version});
    print("Opset Version: {}\n", .{metadata.opset_version});
    
    print("\n📊 Model Statistics\n");
    print("------------------\n");
    const stats = parsed_model.getStats();
    stats.print();
    
    print("\n🔗 Model Inputs\n");
    print("---------------\n");
    const inputs = parsed_model.getInputs();
    for (inputs, 0..) |input, i| {
        print("Input {}: {s}\n", .{i, input.name});
        print("  Shape: [");
        for (input.shape, 0..) |dim, j| {
            if (j > 0) print(", ");
            if (dim < 0) {
                print("?");
            } else {
                print("{}", .{dim});
            }
        }
        print("]\n");
        print("  Type: {s}\n", .{@tagName(input.dtype)});
    }
    
    print("\n🔗 Model Outputs\n");
    print("----------------\n");
    const outputs = parsed_model.getOutputs();
    for (outputs, 0..) |output, i| {
        print("Output {}: {s}\n", .{i, output.name});
        print("  Shape: [");
        for (output.shape, 0..) |dim, j| {
            if (j > 0) print(", ");
            if (dim < 0) {
                print("?");
            } else {
                print("{}", .{dim});
            }
        }
        print("]\n");
        print("  Type: {s}\n", .{@tagName(output.dtype)});
    }
    
    print("\n🎉 Test completed successfully!\n");
    print("The ONNX parser can handle your real model file.\n");
}

const std = @import("std");
const lib = @import("src/lib.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧪 Testing SqueezeNet ONNX Parser", .{});
    std.log.info("=================================", .{});

    // Initialize ONNX parser
    var onnx_parser = lib.formats.onnx.parser.ONNXParser.init(allocator);
    
    const model_path = "models/squeezenet.onnx";
    std.log.info("📁 Attempting to parse: {s}", .{model_path});

    // Try to parse the SqueezeNet model
    if (onnx_parser.parseFile(model_path)) |model| {
        defer model.deinit();
        
        std.log.info("✅ Successfully parsed SqueezeNet ONNX model!", .{});
        std.log.info("📊 Model details:", .{});
        std.log.info("  Name: {s}", .{model.metadata.name});
        std.log.info("  Version: {s}", .{model.metadata.version});
        std.log.info("  Format: {s}", .{@tagName(model.metadata.format)});
        
        if (model.graph) |graph| {
            std.log.info("  Graph nodes: {d}", .{graph.nodes.len});
            std.log.info("  Graph inputs: {d}", .{graph.inputs.len});
            std.log.info("  Graph outputs: {d}", .{graph.outputs.len});
            
            // Show first few nodes
            std.log.info("📋 First few nodes:", .{});
            const max_nodes = @min(5, graph.nodes.len);
            for (graph.nodes[0..max_nodes], 0..) |node, i| {
                std.log.info("  {d}. {s} ({s})", .{ i + 1, node.name, node.op_type });
            }
        }
        
    } else |err| {
        std.log.err("❌ Failed to parse SqueezeNet model: {}", .{err});
        std.log.info("💡 This might be expected if the ONNX parser is still in development", .{});
        
        // Try to get some basic file info
        const file = std.fs.cwd().openFile(model_path, .{}) catch |file_err| {
            std.log.err("❌ Cannot even open file: {}", .{file_err});
            return;
        };
        defer file.close();
        
        const file_size = try file.getEndPos();
        std.log.info("📏 File size: {d} bytes ({d:.2} MB)", .{ file_size, @as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0) });
        
        // Read first few bytes to check if it's a valid ONNX file
        var header: [16]u8 = undefined;
        _ = try file.readAll(&header);
        std.log.info("🔍 File header (hex): {x}", .{std.fmt.fmtSliceHexLower(&header)});
    }

    std.log.info("", .{});
    std.log.info("🎯 Next steps:", .{});
    std.log.info("1. If parsing succeeded: Integrate with CLI for real ONNX inference", .{});
    std.log.info("2. If parsing failed: Continue development of ONNX parser", .{});
    std.log.info("3. Test with other ONNX models in the models/ directory", .{});
}

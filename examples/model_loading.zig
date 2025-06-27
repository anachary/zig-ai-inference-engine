const std = @import("std");
const lib = @import("zig-ai-engine");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.log.info("Model Loading Example", .{});
    
    // Initialize the AI engine
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 1024,
        .num_threads = 4,
    });
    defer engine.deinit();
    
    // TODO: Load a real model when implemented
    // try engine.loadModel("path/to/model.onnx");
    
    std.log.info("Model loading example completed", .{});
}

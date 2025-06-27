const std = @import("std");
const lib = @import("zig-ai-engine");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.log.info("Custom Operator Example", .{});
    
    // Create tensors for testing
    const shape = [_]usize{ 2, 2 };
    var a = try lib.tensor.Tensor.init(allocator, &shape, .f32);
    defer a.deinit();
    
    var b = try lib.tensor.Tensor.init(allocator, &shape, .f32);
    defer b.deinit();
    
    // Fill with test data
    try a.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try a.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try a.set_f32(&[_]usize{ 1, 0 }, 3.0);
    try a.set_f32(&[_]usize{ 1, 1 }, 4.0);
    
    try b.set_f32(&[_]usize{ 0, 0 }, 0.5);
    try b.set_f32(&[_]usize{ 0, 1 }, 1.5);
    try b.set_f32(&[_]usize{ 1, 0 }, 2.5);
    try b.set_f32(&[_]usize{ 1, 1 }, 3.5);
    
    std.log.info("Tensor A: {}", .{a});
    std.log.info("Tensor B: {}", .{b});
    
    // TODO: Implement custom operators when the framework is ready
    
    std.log.info("Custom operator example completed", .{});
}

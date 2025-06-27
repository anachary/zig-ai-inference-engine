const std = @import("std");
const lib = @import("zig-ai-engine");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("Simple Inference Example - Phase 1 Complete!", .{});

    // Initialize the enhanced AI engine
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 512,
        .num_threads = 2,
        .enable_profiling = true,
        .tensor_pool_size = 20,
    });
    defer engine.deinit();

    std.log.info("Engine initialized with enhanced features", .{});

    // Demonstrate tensor pool usage
    const shape = [_]usize{ 2, 3 };
    var input1 = try engine.get_tensor(&shape, .f32);
    var input2 = try engine.get_tensor(&shape, .f32);
    var result = try engine.get_tensor(&shape, .f32);

    // Fill input tensors with sample data
    std.log.info("Filling input tensors...", .{});
    try input1.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try input1.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try input1.set_f32(&[_]usize{ 0, 2 }, 3.0);
    try input1.set_f32(&[_]usize{ 1, 0 }, 4.0);
    try input1.set_f32(&[_]usize{ 1, 1 }, 5.0);
    try input1.set_f32(&[_]usize{ 1, 2 }, 6.0);

    try input2.set_f32(&[_]usize{ 0, 0 }, 0.1);
    try input2.set_f32(&[_]usize{ 0, 1 }, 0.2);
    try input2.set_f32(&[_]usize{ 0, 2 }, 0.3);
    try input2.set_f32(&[_]usize{ 1, 0 }, 0.4);
    try input2.set_f32(&[_]usize{ 1, 1 }, 0.5);
    try input2.set_f32(&[_]usize{ 1, 2 }, 0.6);

    std.log.info("Input1: {}", .{input1});
    std.log.info("Input2: {}", .{input2});

    // Demonstrate operator execution
    std.log.info("Executing Add operation...", .{});
    const add_inputs = [_]lib.Tensor{ input1, input2 };
    var add_outputs = [_]lib.Tensor{result};
    try engine.execute_operator("Add", &add_inputs, &add_outputs);

    std.log.info("Addition result: {}", .{result});
    std.log.info("Result[0,0] = {d:.3}", .{try result.get_f32(&[_]usize{ 0, 0 })});
    std.log.info("Result[1,2] = {d:.3}", .{try result.get_f32(&[_]usize{ 1, 2 })});

    // Demonstrate ReLU activation
    std.log.info("Applying ReLU activation...", .{});
    try input1.set_f32(&[_]usize{ 0, 0 }, -2.0);
    try input1.set_f32(&[_]usize{ 0, 1 }, 3.0);
    try input1.set_f32(&[_]usize{ 0, 2 }, -1.0);

    const relu_inputs = [_]lib.Tensor{input1};
    var relu_outputs = [_]lib.Tensor{result};
    try engine.execute_operator("ReLU", &relu_inputs, &relu_outputs);

    std.log.info("ReLU result: {}", .{result});
    std.log.info("ReLU[-2.0] = {d:.3}", .{try result.get_f32(&[_]usize{ 0, 0 })});
    std.log.info("ReLU[3.0] = {d:.3}", .{try result.get_f32(&[_]usize{ 0, 1 })});
    std.log.info("ReLU[-1.0] = {d:.3}", .{try result.get_f32(&[_]usize{ 0, 2 })});

    // Demonstrate matrix multiplication
    std.log.info("Testing matrix multiplication...", .{});
    var mat_a = try engine.get_tensor(&[_]usize{ 2, 3 }, .f32);
    var mat_b = try engine.get_tensor(&[_]usize{ 3, 2 }, .f32);
    var mat_result = try engine.get_tensor(&[_]usize{ 2, 2 }, .f32);

    // Fill matrices: A = [[1,2,3], [4,5,6]], B = [[1,2], [3,4], [5,6]]
    try mat_a.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try mat_a.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try mat_a.set_f32(&[_]usize{ 0, 2 }, 3.0);
    try mat_a.set_f32(&[_]usize{ 1, 0 }, 4.0);
    try mat_a.set_f32(&[_]usize{ 1, 1 }, 5.0);
    try mat_a.set_f32(&[_]usize{ 1, 2 }, 6.0);

    try mat_b.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try mat_b.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try mat_b.set_f32(&[_]usize{ 1, 0 }, 3.0);
    try mat_b.set_f32(&[_]usize{ 1, 1 }, 4.0);
    try mat_b.set_f32(&[_]usize{ 2, 0 }, 5.0);
    try mat_b.set_f32(&[_]usize{ 2, 1 }, 6.0);

    const matmul_inputs = [_]lib.Tensor{ mat_a, mat_b };
    var matmul_outputs = [_]lib.Tensor{mat_result};
    try engine.execute_operator("MatMul", &matmul_inputs, &matmul_outputs);

    std.log.info("Matrix multiplication result:", .{});
    std.log.info("  [{d:.1}, {d:.1}]", .{ try mat_result.get_f32(&[_]usize{ 0, 0 }), try mat_result.get_f32(&[_]usize{ 0, 1 }) });
    std.log.info("  [{d:.1}, {d:.1}]", .{ try mat_result.get_f32(&[_]usize{ 1, 0 }), try mat_result.get_f32(&[_]usize{ 1, 1 }) });

    // Show engine statistics
    const stats = engine.get_stats();
    std.log.info("Engine Statistics:", .{});
    std.log.info("  Available operators: {d}", .{stats.operators.total_operators});
    std.log.info("  Tensors in pool: {d}", .{stats.tensor_pool.total_pooled});
    std.log.info("  Memory usage: {d} bytes", .{stats.memory.current_usage});
    std.log.info("  Peak memory: {d} bytes", .{stats.memory.peak_usage});

    // Cleanup tensors (immediate deallocation to prevent memory leaks)
    engine.cleanup_tensor(input1);
    engine.cleanup_tensor(input2);
    engine.cleanup_tensor(result);
    engine.cleanup_tensor(mat_a);
    engine.cleanup_tensor(mat_b);
    engine.cleanup_tensor(mat_result);

    std.log.info("Phase 1 inference example completed successfully!", .{});
    std.log.info("✅ Tensor system with shape utilities", .{});
    std.log.info("✅ SIMD-optimized operations", .{});
    std.log.info("✅ Memory management with pools", .{});
    std.log.info("✅ Basic operators (Add, ReLU, MatMul, etc.)", .{});
    std.log.info("✅ Operator registry and execution", .{});
}

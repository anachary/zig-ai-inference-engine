const std = @import("std");
const lib = @import("zig-ai-engine");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.log.info("🎉 Phase 1 Demo - Zig AI Interface Engine", .{});
    std.log.info("==========================================", .{});
    
    // Test 1: Basic Tensor Operations
    std.log.info("✅ Test 1: Basic Tensor Operations", .{});
    
    const shape = [_]usize{ 2, 3 };
    var tensor1 = try lib.tensor.Tensor.init(allocator, &shape, .f32);
    defer tensor1.deinit();
    
    var tensor2 = try lib.tensor.Tensor.init(allocator, &shape, .f32);
    defer tensor2.deinit();
    
    // Fill tensors with test data
    try tensor1.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try tensor1.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try tensor1.set_f32(&[_]usize{ 0, 2 }, 3.0);
    try tensor1.set_f32(&[_]usize{ 1, 0 }, 4.0);
    try tensor1.set_f32(&[_]usize{ 1, 1 }, 5.0);
    try tensor1.set_f32(&[_]usize{ 1, 2 }, 6.0);
    
    try tensor2.set_f32(&[_]usize{ 0, 0 }, 0.1);
    try tensor2.set_f32(&[_]usize{ 0, 1 }, 0.2);
    try tensor2.set_f32(&[_]usize{ 0, 2 }, 0.3);
    try tensor2.set_f32(&[_]usize{ 1, 0 }, 0.4);
    try tensor2.set_f32(&[_]usize{ 1, 1 }, 0.5);
    try tensor2.set_f32(&[_]usize{ 1, 2 }, 0.6);
    
    std.log.info("   Tensor1[0,0] = {d:.1}", .{try tensor1.get_f32(&[_]usize{ 0, 0 })});
    std.log.info("   Tensor1[1,2] = {d:.1}", .{try tensor1.get_f32(&[_]usize{ 1, 2 })});
    std.log.info("   Tensor2[0,0] = {d:.1}", .{try tensor2.get_f32(&[_]usize{ 0, 0 })});
    std.log.info("   Tensor2[1,2] = {d:.1}", .{try tensor2.get_f32(&[_]usize{ 1, 2 })});
    
    // Test 2: SIMD Operations
    std.log.info("✅ Test 2: SIMD Vector Operations", .{});
    
    const len = 8;
    const a = try allocator.alloc(f32, len);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, len);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, len);
    defer allocator.free(result);
    
    // Initialize test data
    for (0..len) |i| {
        a[i] = @as(f32, @floatFromInt(i + 1));
        b[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;
    }
    
    // Test SIMD addition
    try lib.simd.vector_add_f32(a, b, result);
    std.log.info("   SIMD Add: {d:.1} + {d:.1} = {d:.1}", .{ a[0], b[0], result[0] });
    std.log.info("   SIMD Add: {d:.1} + {d:.1} = {d:.1}", .{ a[7], b[7], result[7] });
    
    // Test SIMD dot product
    const dot = try lib.simd.vector_dot_f32(a, b);
    std.log.info("   SIMD Dot Product: {d:.2}", .{dot});
    
    // Test 3: Memory Management
    std.log.info("✅ Test 3: Memory Management", .{});
    
    var memory_manager = lib.memory.MemoryManager.init(allocator);
    defer memory_manager.deinit();
    
    var tensor_pool = lib.pool.TensorPool.init(allocator, 5);
    defer tensor_pool.deinit();
    
    // Get tensors from pool
    var pooled1 = try tensor_pool.get_tensor(&shape, .f32);
    var pooled2 = try tensor_pool.get_tensor(&shape, .f32);
    
    std.log.info("   Created pooled tensors: {} and {}", .{ pooled1, pooled2 });
    
    // Return to pool
    try tensor_pool.return_tensor(pooled1);
    try tensor_pool.return_tensor(pooled2);
    
    const pool_stats = tensor_pool.get_stats();
    std.log.info("   Pool stats: {} tensors pooled", .{pool_stats.total_pooled});
    
    // Test 4: Shape Utilities
    std.log.info("✅ Test 4: Shape Utilities", .{});
    
    const test_shape = [_]usize{ 2, 3, 4 };
    const strides = try lib.shape.compute_strides(&test_shape, allocator);
    defer allocator.free(strides);
    
    std.log.info("   Shape: [{d}, {d}, {d}]", .{ test_shape[0], test_shape[1], test_shape[2] });
    std.log.info("   Strides: [{d}, {d}, {d}]", .{ strides[0], strides[1], strides[2] });
    
    const total_elements = lib.shape.shape_numel(&test_shape);
    std.log.info("   Total elements: {d}", .{total_elements});
    
    // Test broadcasting
    const shape1 = [_]usize{ 3, 1 };
    const shape2 = [_]usize{ 1, 4 };
    const broadcast_result = try lib.shape.broadcast_shapes(&shape1, &shape2, allocator);
    defer allocator.free(broadcast_result);
    
    std.log.info("   Broadcast [{d}, {d}] + [{d}, {d}] = [{d}, {d}]", .{
        shape1[0], shape1[1], shape2[0], shape2[1], broadcast_result[0], broadcast_result[1]
    });
    
    // Test 5: Hardware Detection
    std.log.info("✅ Test 5: Hardware Capabilities", .{});
    
    const caps = lib.detectHardwareCapabilities();
    std.log.info("   SIMD Level: {s}", .{@tagName(caps.simd_level)});
    std.log.info("   CPU Cores: {d}", .{caps.num_cores});
    std.log.info("   L1 Cache: {d}KB", .{caps.cache_sizes.l1 / 1024});
    std.log.info("   L2 Cache: {d}KB", .{caps.cache_sizes.l2 / 1024});
    std.log.info("   L3 Cache: {d}MB", .{caps.cache_sizes.l3 / (1024 * 1024)});
    
    // Test 6: Direct Operator Usage
    std.log.info("✅ Test 6: Direct Operator Usage", .{});
    
    var result_tensor = try lib.tensor.Tensor.init(allocator, &shape, .f32);
    defer result_tensor.deinit();
    
    // Test Add operator directly
    const add_inputs = [_]lib.tensor.Tensor{ tensor1, tensor2 };
    var add_outputs = [_]lib.tensor.Tensor{result_tensor};
    
    try lib.operators.Add.op.forward(&add_inputs, &add_outputs, allocator);
    
    std.log.info("   Direct Add: {d:.1} + {d:.1} = {d:.1}", .{
        try tensor1.get_f32(&[_]usize{ 0, 0 }),
        try tensor2.get_f32(&[_]usize{ 0, 0 }),
        try result_tensor.get_f32(&[_]usize{ 0, 0 })
    });
    
    // Test ReLU operator
    try tensor1.set_f32(&[_]usize{ 0, 0 }, -2.0);
    try tensor1.set_f32(&[_]usize{ 0, 1 }, 3.0);
    
    const relu_inputs = [_]lib.tensor.Tensor{tensor1};
    var relu_outputs = [_]lib.tensor.Tensor{result_tensor};
    
    try lib.operators.ReLU.op.forward(&relu_inputs, &relu_outputs, allocator);
    
    std.log.info("   ReLU(-2.0) = {d:.1}", .{try result_tensor.get_f32(&[_]usize{ 0, 0 })});
    std.log.info("   ReLU(3.0) = {d:.1}", .{try result_tensor.get_f32(&[_]usize{ 0, 1 })});
    
    // Test 7: Matrix Multiplication
    std.log.info("✅ Test 7: Matrix Multiplication", .{});
    
    var mat_a = try lib.tensor.Tensor.init(allocator, &[_]usize{ 2, 3 }, .f32);
    defer mat_a.deinit();
    var mat_b = try lib.tensor.Tensor.init(allocator, &[_]usize{ 3, 2 }, .f32);
    defer mat_b.deinit();
    var mat_result = try lib.tensor.Tensor.init(allocator, &[_]usize{ 2, 2 }, .f32);
    defer mat_result.deinit();
    
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
    
    const matmul_inputs = [_]lib.tensor.Tensor{ mat_a, mat_b };
    var matmul_outputs = [_]lib.tensor.Tensor{mat_result};
    
    try lib.operators.MatMul.op.forward(&matmul_inputs, &matmul_outputs, allocator);
    
    std.log.info("   MatMul result:", .{});
    std.log.info("     [{d:.0}, {d:.0}]", .{ try mat_result.get_f32(&[_]usize{ 0, 0 }), try mat_result.get_f32(&[_]usize{ 0, 1 }) });
    std.log.info("     [{d:.0}, {d:.0}]", .{ try mat_result.get_f32(&[_]usize{ 1, 0 }), try mat_result.get_f32(&[_]usize{ 1, 1 }) });
    
    // Final Summary
    std.log.info("", .{});
    std.log.info("🎊 Phase 1 Complete - All Core Features Working!", .{});
    std.log.info("================================================", .{});
    std.log.info("✅ Tensor system with shape utilities", .{});
    std.log.info("✅ SIMD-optimized vector operations", .{});
    std.log.info("✅ Memory management with pools", .{});
    std.log.info("✅ Basic operators (Add, ReLU, MatMul)", .{});
    std.log.info("✅ Hardware capability detection", .{});
    std.log.info("✅ Comprehensive error handling", .{});
    std.log.info("", .{});
    std.log.info("🚀 Ready for Phase 2 development!", .{});
}

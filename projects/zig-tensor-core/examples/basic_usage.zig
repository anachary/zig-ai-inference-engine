const std = @import("std");
const tensor_core = @import("zig-tensor-core");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== Zig Tensor Core - Basic Usage Example ===\n");

    // Initialize tensor core with desktop configuration
    const config = tensor_core.Config.forDevice(.desktop, 4096);
    var core = try tensor_core.TensorCore.init(allocator, config);
    defer core.deinit();

    std.log.info("Tensor Core initialized with config:");
    std.log.info("  Max memory: {}MB", .{config.max_memory_mb});
    std.log.info("  Tensor pool size: {}", .{config.tensor_pool_size});
    std.log.info("  SIMD enabled: {}", .{config.enable_simd});

    // Create tensors using convenience functions
    std.log.info("\n--- Creating Tensors ---");
    
    var zeros_tensor = try tensor_core.ops.zeros(allocator, &[_]usize{ 2, 3 }, .f32);
    defer zeros_tensor.deinit();
    std.log.info("Created zeros tensor: shape [{}, {}]", .{ zeros_tensor.shape[0], zeros_tensor.shape[1] });

    var ones_tensor = try tensor_core.ops.ones(allocator, &[_]usize{ 2, 3 }, .f32);
    defer ones_tensor.deinit();
    std.log.info("Created ones tensor: shape [{}, {}]", .{ ones_tensor.shape[0], ones_tensor.shape[1] });

    // Create tensor from data
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var data_tensor = try tensor_core.ops.fromSlice(allocator, &data, &[_]usize{ 2, 3 });
    defer data_tensor.deinit();
    std.log.info("Created tensor from data: shape [{}, {}]", .{ data_tensor.shape[0], data_tensor.shape[1] });

    // Print tensor values
    std.log.info("\nData tensor values:");
    for (0..data_tensor.shape[0]) |i| {
        for (0..data_tensor.shape[1]) |j| {
            const value = try data_tensor.getF32(&[_]usize{ i, j });
            std.log.info("  [{}, {}] = {d:.1}", .{ i, j, value });
        }
    }

    // Perform math operations
    std.log.info("\n--- Math Operations ---");
    
    var add_result = try tensor_core.math.add(allocator, ones_tensor, data_tensor);
    defer add_result.deinit();
    std.log.info("Addition result (ones + data):");
    for (0..add_result.shape[0]) |i| {
        for (0..add_result.shape[1]) |j| {
            const value = try add_result.getF32(&[_]usize{ i, j });
            std.log.info("  [{}, {}] = {d:.1}", .{ i, j, value });
        }
    }

    var mul_result = try tensor_core.math.mul(allocator, data_tensor, data_tensor);
    defer mul_result.deinit();
    std.log.info("\nElement-wise multiplication (data * data):");
    for (0..mul_result.shape[0]) |i| {
        for (0..mul_result.shape[1]) |j| {
            const value = try mul_result.getF32(&[_]usize{ i, j });
            std.log.info("  [{}, {}] = {d:.1}", .{ i, j, value });
        }
    }

    // Scalar operations
    var scalar_result = try tensor_core.math.scalarMul(allocator, data_tensor, @as(f32, 2.0));
    defer scalar_result.deinit();
    std.log.info("\nScalar multiplication (data * 2.0):");
    for (0..scalar_result.shape[0]) |i| {
        for (0..scalar_result.shape[1]) |j| {
            const value = try scalar_result.getF32(&[_]usize{ i, j });
            std.log.info("  [{}, {}] = {d:.1}", .{ i, j, value });
        }
    }

    // Reduction operations
    var sum_result = try tensor_core.math.sum(allocator, data_tensor);
    defer sum_result.deinit();
    const sum_value = try sum_result.getF32(&[_]usize{});
    std.log.info("\nSum of all elements: {d:.1}", .{sum_value});

    var mean_result = try tensor_core.math.mean(allocator, data_tensor);
    defer mean_result.deinit();
    const mean_value = try mean_result.getF32(&[_]usize{});
    std.log.info("Mean of all elements: {d:.1}", .{mean_value});

    // Matrix operations
    std.log.info("\n--- Matrix Operations ---");
    
    // Create matrices for multiplication
    var matrix_a = try tensor_core.Tensor.init(allocator, &[_]usize{ 2, 3 }, .f32);
    defer matrix_a.deinit();
    var matrix_b = try tensor_core.Tensor.init(allocator, &[_]usize{ 3, 2 }, .f32);
    defer matrix_b.deinit();

    // Fill matrices with test data
    try matrix_a.fill(@as(f32, 1.0));
    try matrix_b.fill(@as(f32, 2.0));

    var matmul_result = try tensor_core.math.matmul(allocator, matrix_a, matrix_b);
    defer matmul_result.deinit();
    std.log.info("Matrix multiplication result (2x3 * 3x2 = 2x2):");
    for (0..matmul_result.shape[0]) |i| {
        for (0..matmul_result.shape[1]) |j| {
            const value = try matmul_result.getF32(&[_]usize{ i, j });
            std.log.info("  [{}, {}] = {d:.1}", .{ i, j, value });
        }
    }

    // Transpose operation
    var transpose_result = try tensor_core.math.transpose(allocator, data_tensor);
    defer transpose_result.deinit();
    std.log.info("\nTranspose result (2x3 -> 3x2):");
    for (0..transpose_result.shape[0]) |i| {
        for (0..transpose_result.shape[1]) |j| {
            const value = try transpose_result.getF32(&[_]usize{ i, j });
            std.log.info("  [{}, {}] = {d:.1}", .{ i, j, value });
        }
    }

    // Memory and performance statistics
    std.log.info("\n--- Performance Statistics ---");
    
    const pool_stats = core.getPoolStats();
    std.log.info("Tensor Pool Statistics:");
    std.log.info("  Number of pools: {}", .{pool_stats.num_pools});
    std.log.info("  Total tensors: {}", .{pool_stats.total_tensors});
    std.log.info("  Active tensors: {}", .{pool_stats.active_tensors});
    std.log.info("  Cache hit ratio: {d:.2}%", .{pool_stats.hitRatio() * 100});

    if (core.getMemoryStats()) |memory_stats| {
        std.log.info("Memory Statistics:");
        std.log.info("  Total allocated: {} bytes", .{memory_stats.total_allocated});
        std.log.info("  Current usage: {} bytes", .{memory_stats.current_usage});
        std.log.info("  Peak usage: {} bytes", .{memory_stats.peak_usage});
        std.log.info("  Active allocations: {}", .{memory_stats.active_allocations});
    }

    // SIMD capabilities
    std.log.info("\n--- SIMD Information ---");
    std.log.info("SIMD available: {}", .{tensor_core.simd.isAvailable()});
    if (tensor_core.simd.isAvailable()) {
        std.log.info("Vector width for f32: {}", .{tensor_core.simd.vectorWidth(f32)});
        std.log.info("Vector width for f64: {}", .{tensor_core.simd.vectorWidth(f64)});
    }

    // Demonstrate tensor pool usage
    std.log.info("\n--- Tensor Pool Demo ---");
    
    var pooled_tensor = try core.createTensor(&[_]usize{ 4, 4 }, .f32);
    try pooled_tensor.fill(@as(f32, 42.0));
    std.log.info("Created pooled tensor, filled with 42.0");
    
    // Return to pool
    try core.returnTensor(pooled_tensor);
    std.log.info("Returned tensor to pool");
    
    // Get another tensor (should reuse from pool)
    var reused_tensor = try core.createTensor(&[_]usize{ 4, 4 }, .f32);
    defer reused_tensor.deinit(); // Clean up manually since we're not returning to pool
    
    const first_value = try reused_tensor.getF32(&[_]usize{ 0, 0 });
    std.log.info("Reused tensor first value: {d:.1} (should be 0.0 due to zeroing)", .{first_value});

    std.log.info("\n=== Example completed successfully! ===");
}

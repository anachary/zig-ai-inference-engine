const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

// Import the inference engine and operators
const inference_engine = @import("zig-inference-engine");
const tensor_core = @import("zig-tensor-core");

test "test core operators compilation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test that we can access operator modules
    const arithmetic = inference_engine.operators.arithmetic;
    const matrix = inference_engine.operators.matrix;
    const activation = inference_engine.operators.activation;

    // Test that operator info can be retrieved
    const add_info = arithmetic.Add.getInfo();
    const matmul_info = matrix.MatMul.getInfo();
    const relu_info = activation.ReLU.getInfo();
    const softmax_info = activation.Softmax.getInfo();

    // Verify operator properties
    try testing.expectEqualStrings("Add", add_info.name);
    try testing.expectEqualStrings("MatMul", matmul_info.name);
    try testing.expectEqualStrings("ReLU", relu_info.name);
    try testing.expectEqualStrings("Softmax", softmax_info.name);

    // Verify input/output constraints
    try testing.expect(add_info.min_inputs == 2);
    try testing.expect(add_info.max_inputs == 2);
    try testing.expect(add_info.min_outputs == 1);
    try testing.expect(add_info.max_outputs == 1);

    try testing.expect(matmul_info.min_inputs == 2);
    try testing.expect(matmul_info.min_outputs == 1);

    try testing.expect(relu_info.min_inputs == 1);
    try testing.expect(relu_info.min_outputs == 1);

    std.log.info("✅ Core operators compilation and info retrieval successful!", .{});
    std.log.info("📊 Verified operators: Add, MatMul, ReLU, Softmax", .{});

    _ = allocator; // Suppress unused warning
}

test "test tensor creation and basic operations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create simple tensors for testing
    const shape = [_]usize{ 2, 3 };

    var tensor_a = try tensor_core.Tensor.init(allocator, &shape, .f32);
    defer tensor_a.deinit();

    var tensor_b = try tensor_core.Tensor.init(allocator, &shape, .f32);
    defer tensor_b.deinit();

    var tensor_c = try tensor_core.Tensor.init(allocator, &shape, .f32);
    defer tensor_c.deinit();

    // Fill tensors with test data
    try tensor_a.setF32(&[_]usize{ 0, 0 }, 1.0);
    try tensor_a.setF32(&[_]usize{ 0, 1 }, 2.0);
    try tensor_a.setF32(&[_]usize{ 0, 2 }, 3.0);
    try tensor_a.setF32(&[_]usize{ 1, 0 }, 4.0);
    try tensor_a.setF32(&[_]usize{ 1, 1 }, 5.0);
    try tensor_a.setF32(&[_]usize{ 1, 2 }, 6.0);

    try tensor_b.setF32(&[_]usize{ 0, 0 }, 1.0);
    try tensor_b.setF32(&[_]usize{ 0, 1 }, 1.0);
    try tensor_b.setF32(&[_]usize{ 0, 2 }, 1.0);
    try tensor_b.setF32(&[_]usize{ 1, 0 }, 1.0);
    try tensor_b.setF32(&[_]usize{ 1, 1 }, 1.0);
    try tensor_b.setF32(&[_]usize{ 1, 2 }, 1.0);

    // Verify tensor values
    const val_a = try tensor_a.getF32(&[_]usize{ 0, 0 });
    const val_b = try tensor_b.getF32(&[_]usize{ 0, 0 });

    try testing.expectEqual(@as(f32, 1.0), val_a);
    try testing.expectEqual(@as(f32, 1.0), val_b);

    std.log.info("✅ Tensor creation and basic operations working!", .{});
}

test "test inference engine initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test engine configuration
    const config = inference_engine.Config{
        .device_type = .auto,
        .num_threads = 2,
        .enable_gpu = false,
        .optimization_level = .balanced,
        .memory_limit_mb = 1024,
    };

    // Initialize engine
    var engine = try inference_engine.Engine.init(allocator, config);
    defer engine.deinit();

    // Verify engine stats
    const stats = engine.getStats();
    std.log.info("🚀 Engine Statistics:", .{});
    std.log.info("  - Total inferences: {d}", .{stats.total_inferences});
    std.log.info("  - Peak memory: {d} MB", .{stats.peak_memory_mb});
    std.log.info("  - Average latency: {d:.2} ms", .{stats.average_latency_ms});
    std.log.info("  - Model loaded: {}", .{stats.model_loaded});

    // Basic engine validation
    try testing.expect(stats.total_inferences == 0); // No inferences run yet

    std.log.info("✅ Inference engine initialization successful!", .{});
}

test "test operator validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test Add operator validation directly
    const arithmetic = inference_engine.operators.arithmetic;
    const add_info = arithmetic.Add.getInfo();

    // Test valid input shapes
    const input_shapes = [_][]const usize{
        &[_]usize{ 2, 3 },
        &[_]usize{ 2, 3 },
    };

    var attributes = std.StringHashMap([]const u8).init(allocator);
    defer attributes.deinit();

    const output_shapes = try add_info.validate_fn(&input_shapes, attributes, allocator);
    defer {
        for (output_shapes) |shape| {
            allocator.free(shape);
        }
        allocator.free(output_shapes);
    }

    try testing.expect(output_shapes.len == 1);
    try testing.expect(output_shapes[0].len == 2);
    try testing.expectEqual(@as(usize, 2), output_shapes[0][0]);
    try testing.expectEqual(@as(usize, 3), output_shapes[0][1]);

    std.log.info("✅ Operator validation working correctly!", .{});
}

const std = @import("std");

// Import all test modules
test {
    // Core module tests
    _ = @import("core/tensor.zig");
    _ = @import("core/shape.zig");
    _ = @import("core/simd.zig");

    // Memory module tests
    _ = @import("memory/manager.zig");
    _ = @import("memory/pool.zig");

    // Engine module tests
    _ = @import("engine/operators.zig");
    _ = @import("engine/registry.zig");

    // Scheduler module tests (when implemented)
    // _ = @import("scheduler/task_queue.zig");

    // Network module tests (when implemented)
    // _ = @import("network/server.zig");

    // Library tests
    _ = @import("lib.zig");
}

// Integration tests
test "basic tensor operations integration" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const tensor = @import("core/tensor.zig");

    // Create a simple 2x3 matrix
    const shape = [_]usize{ 2, 3 };
    var a = try tensor.Tensor.init(allocator, &shape, .f32);
    defer a.deinit();

    var b = try tensor.Tensor.init(allocator, &shape, .f32);
    defer b.deinit();

    // Fill tensors with test data
    try a.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try a.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try a.set_f32(&[_]usize{ 0, 2 }, 3.0);
    try a.set_f32(&[_]usize{ 1, 0 }, 4.0);
    try a.set_f32(&[_]usize{ 1, 1 }, 5.0);
    try a.set_f32(&[_]usize{ 1, 2 }, 6.0);

    try b.set_f32(&[_]usize{ 0, 0 }, 0.5);
    try b.set_f32(&[_]usize{ 0, 1 }, 1.5);
    try b.set_f32(&[_]usize{ 0, 2 }, 2.5);
    try b.set_f32(&[_]usize{ 1, 0 }, 3.5);
    try b.set_f32(&[_]usize{ 1, 1 }, 4.5);
    try b.set_f32(&[_]usize{ 1, 2 }, 5.5);

    // Verify values
    try testing.expect(try a.get_f32(&[_]usize{ 0, 0 }) == 1.0);
    try testing.expect(try a.get_f32(&[_]usize{ 1, 2 }) == 6.0);
    try testing.expect(try b.get_f32(&[_]usize{ 0, 0 }) == 0.5);
    try testing.expect(try b.get_f32(&[_]usize{ 1, 2 }) == 5.5);
}

test "tensor utility functions integration" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const tensor = @import("core/tensor.zig");

    // Test zeros tensor
    const shape = [_]usize{ 3, 3 };
    var zeros_tensor = try tensor.zeros(allocator, &shape, .f32);
    defer zeros_tensor.deinit();

    for (0..3) |i| {
        for (0..3) |j| {
            try testing.expect(try zeros_tensor.get_f32(&[_]usize{ i, j }) == 0.0);
        }
    }

    // Test ones tensor
    var ones_tensor = try tensor.ones(allocator, &shape, .f32);
    defer ones_tensor.deinit();

    for (0..3) |i| {
        for (0..3) |j| {
            try testing.expect(try ones_tensor.get_f32(&[_]usize{ i, j }) == 1.0);
        }
    }

    // Test arange
    var arange_tensor = try tensor.arange(allocator, 0.0, 5.0, 1.0, .f32);
    defer arange_tensor.deinit();

    try testing.expect(arange_tensor.numel() == 5);
    try testing.expect(try arange_tensor.get_f32(&[_]usize{0}) == 0.0);
    try testing.expect(try arange_tensor.get_f32(&[_]usize{4}) == 4.0);
}

test "tensor reshape integration" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const tensor = @import("core/tensor.zig");

    // Create a 2x3 tensor
    const original_shape = [_]usize{ 2, 3 };
    var original = try tensor.Tensor.init(allocator, &original_shape, .f32);
    defer original.deinit();

    // Fill with sequential values
    for (0..2) |i| {
        for (0..3) |j| {
            const value = @as(f32, @floatFromInt(i * 3 + j));
            try original.set_f32(&[_]usize{ i, j }, value);
        }
    }

    // Reshape to 3x2
    const new_shape = [_]usize{ 3, 2 };
    var reshaped = try original.reshape(allocator, &new_shape);
    defer reshaped.deinit();

    try testing.expect(reshaped.numel() == original.numel());
    try testing.expect(reshaped.shape.len == 2);
    try testing.expect(reshaped.shape[0] == 3);
    try testing.expect(reshaped.shape[1] == 2);
}

test "memory management stress test" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const tensor = @import("core/tensor.zig");

    // Create and destroy many tensors to test memory management
    for (0..100) |i| {
        const size = (i % 10) + 1;
        const shape = [_]usize{ size, size };

        var t = try tensor.Tensor.init(allocator, &shape, .f32);
        defer t.deinit();

        // Fill with some data
        for (0..size) |row| {
            for (0..size) |col| {
                try t.set_f32(&[_]usize{ row, col }, @as(f32, @floatFromInt(row + col)));
            }
        }

        // Verify some values
        try testing.expect(try t.get_f32(&[_]usize{ 0, 0 }) == 0.0);
        if (size > 1) {
            try testing.expect(try t.get_f32(&[_]usize{ 1, 1 }) == 2.0);
        }
    }
}

test "error handling" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const tensor = @import("core/tensor.zig");

    // Test invalid shape
    const invalid_shape = [_]usize{0};
    try testing.expectError(tensor.TensorError.InvalidShape, tensor.Tensor.init(allocator, &invalid_shape, .f32));

    // Test index out of bounds
    const shape = [_]usize{ 2, 2 };
    var t = try tensor.Tensor.init(allocator, &shape, .f32);
    defer t.deinit();

    try testing.expectError(tensor.TensorError.IndexOutOfBounds, t.get_f32(&[_]usize{ 2, 0 }));
    try testing.expectError(tensor.TensorError.IndexOutOfBounds, t.get_f32(&[_]usize{ 0, 2 }));
    try testing.expectError(tensor.TensorError.IndexOutOfBounds, t.set_f32(&[_]usize{ 2, 0 }, 1.0));

    // Test wrong number of indices
    try testing.expectError(tensor.TensorError.IndexOutOfBounds, t.get_f32(&[_]usize{0}));
    try testing.expectError(tensor.TensorError.IndexOutOfBounds, t.get_f32(&[_]usize{ 0, 0, 0 }));
}

test "enhanced inference engine integration" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const lib = @import("lib.zig");

    // Initialize enhanced engine
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 256,
        .num_threads = 2,
        .enable_profiling = true,
        .tensor_pool_size = 10,
    });
    defer engine.deinit();

    // Test tensor pool functionality
    const shape = [_]usize{ 2, 3 };
    var tensor1 = try engine.get_tensor(&shape, .f32);
    defer engine.cleanup_tensor(tensor1);
    var tensor2 = try engine.get_tensor(&shape, .f32);
    defer engine.cleanup_tensor(tensor2);

    // Fill tensors with test data
    try tensor1.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try tensor1.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try tensor1.set_f32(&[_]usize{ 0, 2 }, 3.0);
    try tensor1.set_f32(&[_]usize{ 1, 0 }, 4.0);
    try tensor1.set_f32(&[_]usize{ 1, 1 }, 5.0);
    try tensor1.set_f32(&[_]usize{ 1, 2 }, 6.0);

    try tensor2.set_f32(&[_]usize{ 0, 0 }, 0.5);
    try tensor2.set_f32(&[_]usize{ 0, 1 }, 1.5);
    try tensor2.set_f32(&[_]usize{ 0, 2 }, 2.5);
    try tensor2.set_f32(&[_]usize{ 1, 0 }, 3.5);
    try tensor2.set_f32(&[_]usize{ 1, 1 }, 4.5);
    try tensor2.set_f32(&[_]usize{ 1, 2 }, 5.5);

    var result = try engine.get_tensor(&shape, .f32);
    defer engine.cleanup_tensor(result);

    // Test operator execution through engine
    const inputs = [_]lib.Tensor{ tensor1, tensor2 };
    var outputs = [_]lib.Tensor{result};

    try engine.execute_operator("Add", &inputs, &outputs);

    // Verify results
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 0, 0 }), 1.5, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 0, 1 }), 3.5, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 1, 2 }), 11.5, 1e-6);

    // Test ReLU operator
    try tensor1.set_f32(&[_]usize{ 0, 0 }, -1.0);
    try tensor1.set_f32(&[_]usize{ 0, 1 }, 2.0);

    const relu_inputs = [_]lib.Tensor{tensor1};
    var relu_outputs = [_]lib.Tensor{result};

    try engine.execute_operator("ReLU", &relu_inputs, &relu_outputs);

    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 0, 0 }), 0.0, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 0, 1 }), 2.0, 1e-6);

    // Test engine statistics
    const stats = engine.get_stats();
    try testing.expect(stats.operators.total_operators >= 6); // Built-in operators

    // Test resource reset
    engine.reset_temp_resources();
}

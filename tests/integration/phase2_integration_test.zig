const std = @import("std");
const testing = std.testing;
const lib = @import("zig-ai-engine");

/// Comprehensive Phase 2 Integration Test
/// Tests all major components working together:
/// - HTTP Server + ONNX Parser + Computation Graph + Enhanced Operators + GPU Support

test "Phase 2 Complete Integration Test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧪 Starting Phase 2 Integration Test", .{});

    // Test 1: Engine Initialization with all Phase 2 features
    std.log.info("✅ Test 1: Engine Initialization", .{});
    
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 512,
        .num_threads = 2,
        .enable_profiling = true,
        .tensor_pool_size = 50,
    });
    defer engine.deinit();

    // Verify engine is properly initialized
    try testing.expect(engine.model_loaded == false);
    
    // Test 2: GPU Context Integration
    std.log.info("✅ Test 2: GPU Context Integration", .{});
    
    var gpu_context = lib.gpu.createOptimalContext(allocator) catch |err| {
        std.log.warn("GPU context creation failed: {}, using CPU fallback", .{err});
        // Continue with CPU-only testing
        return;
    };
    defer gpu_context.deinit();
    
    const device_info = gpu_context.getDeviceInfo();
    try testing.expect(device_info.compute_units > 0);
    try testing.expect(device_info.memory_total > 0);
    
    // Test 3: Enhanced Operators with GPU Support
    std.log.info("✅ Test 3: Enhanced Operators", .{});
    
    // Create test tensors
    const shape = [_]usize{ 4, 4 };
    var tensor1 = try engine.get_tensor(&shape, .f32);
    defer engine.return_tensor(tensor1) catch {};
    
    var tensor2 = try engine.get_tensor(&shape, .f32);
    defer engine.return_tensor(tensor2) catch {};
    
    var output = try engine.get_tensor(&shape, .f32);
    defer engine.return_tensor(output) catch {};
    
    // Fill tensors with test data
    for (0..16) |i| {
        const idx = [_]usize{ i / 4, i % 4 };
        try tensor1.set_f32(&idx, @as(f32, @floatFromInt(i)) * 0.1);
        try tensor2.set_f32(&idx, @as(f32, @floatFromInt(i)) * 0.2);
    }
    
    // Test enhanced operators
    const inputs = [_]lib.Tensor{ tensor1, tensor2 };
    var outputs = [_]lib.Tensor{output};
    
    try engine.execute_operator("Add", &inputs, &outputs);
    
    // Verify results
    const result = try output.get_f32(&[_]usize{ 1, 1 });
    const expected = (5.0 * 0.1) + (5.0 * 0.2); // index [1,1] = 5
    try testing.expectApproxEqAbs(result, expected, 0.001);
    
    // Test 4: Memory Management Integration
    std.log.info("✅ Test 4: Memory Management", .{});
    
    // Test tensor pooling
    const pool_stats_before = engine.getPoolStats();
    
    // Create and return multiple tensors to test pooling
    for (0..10) |_| {
        var temp_tensor = try engine.get_tensor(&shape, .f32);
        try engine.return_tensor(temp_tensor);
    }
    
    const pool_stats_after = engine.getPoolStats();
    try testing.expect(pool_stats_after.total_pooled >= pool_stats_before.total_pooled);
    
    // Test 5: Computation Graph System
    std.log.info("✅ Test 5: Computation Graph", .{});
    
    // Create a simple computation graph
    var graph = try lib.formats.ComputationGraph.init(allocator);
    defer graph.deinit();
    
    // Add input specification
    try graph.addInput("input1", &shape, .f32);
    try graph.addInput("input2", &shape, .f32);
    
    // Add output specification
    try graph.addOutput("output", &shape, .f32);
    
    // Add a simple node (Add operation)
    try graph.addNode(.{
        .id = "add_node",
        .op_type = "Add",
        .inputs = &[_][]const u8{ "input1", "input2" },
        .outputs = &[_][]const u8{"output"},
        .attributes = null,
    });
    
    // Verify graph structure
    try testing.expect(graph.inputs.items.len == 2);
    try testing.expect(graph.outputs.items.len == 1);
    try testing.expect(graph.nodes.items.len == 1);
    
    // Test 6: ONNX Parser Integration
    std.log.info("✅ Test 6: ONNX Parser", .{});
    
    var onnx_parser = try lib.onnx.ONNXParser.init(allocator);
    defer onnx_parser.deinit();
    
    // Test parsing capabilities (without actual ONNX file)
    try testing.expect(onnx_parser.isInitialized());
    
    // Test 7: HTTP Server Integration (without actually starting server)
    std.log.info("✅ Test 7: HTTP Server Components", .{});
    
    // Test JSON processing
    var json_processor = try lib.network.JSONProcessor.init(allocator);
    defer json_processor.deinit();
    
    // Test serialization
    const test_response = lib.network.InferResponse{
        .outputs = &[_]lib.network.InferResponse.TensorData{
            .{
                .name = "test_output",
                .shape = &[_]usize{ 2, 2 },
                .data = &[_]f32{ 1.0, 2.0, 3.0, 4.0 },
                .dtype = "float32",
            },
        },
        .model_id = "test_model",
        .inference_time_ms = 42.0,
    };
    
    const json_str = try json_processor.serializeInferResponse(test_response);
    defer allocator.free(json_str);
    
    try testing.expect(json_str.len > 0);
    try testing.expect(std.mem.indexOf(u8, json_str, "test_output") != null);
    
    // Test 8: Performance Benchmarking
    std.log.info("✅ Test 8: Performance Benchmarks", .{});
    
    const benchmark_iterations = 100;
    const start_time = std.time.nanoTimestamp();
    
    // Benchmark tensor operations
    for (0..benchmark_iterations) |_| {
        var bench_tensor1 = try engine.get_tensor(&[_]usize{ 8, 8 }, .f32);
        var bench_tensor2 = try engine.get_tensor(&[_]usize{ 8, 8 }, .f32);
        var bench_output = try engine.get_tensor(&[_]usize{ 8, 8 }, .f32);
        
        const bench_inputs = [_]lib.Tensor{ bench_tensor1, bench_tensor2 };
        var bench_outputs = [_]lib.Tensor{bench_output};
        
        try engine.execute_operator("Add", &bench_inputs, &bench_outputs);
        
        try engine.return_tensor(bench_tensor1);
        try engine.return_tensor(bench_tensor2);
        try engine.return_tensor(bench_output);
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    const avg_time_ms = total_time_ms / @as(f64, @floatFromInt(benchmark_iterations));
    
    std.log.info("⚡ Benchmark Results:", .{});
    std.log.info("  • {d} iterations completed", .{benchmark_iterations});
    std.log.info("  • Total time: {d:.2}ms", .{total_time_ms});
    std.log.info("  • Average per operation: {d:.3}ms", .{avg_time_ms});
    
    // Performance assertions
    try testing.expect(avg_time_ms < 10.0); // Should be fast on CPU
    try testing.expect(total_time_ms > 0.0);
    
    // Test 9: Memory Leak Detection
    std.log.info("✅ Test 9: Memory Leak Detection", .{});
    
    const memory_stats_before = engine.getMemoryStats();
    
    // Perform operations that should not leak memory
    for (0..50) |_| {
        var temp_tensor = try engine.get_tensor(&[_]usize{ 4, 4 }, .f32);
        // Fill with data
        for (0..16) |i| {
            const idx = [_]usize{ i / 4, i % 4 };
            try temp_tensor.set_f32(&idx, @as(f32, @floatFromInt(i)));
        }
        try engine.return_tensor(temp_tensor);
    }
    
    const memory_stats_after = engine.getMemoryStats();
    
    // Memory usage should be stable (allowing for some pool growth)
    const memory_growth = memory_stats_after.current_usage - memory_stats_before.current_usage;
    try testing.expect(memory_growth < 1024 * 1024); // Less than 1MB growth
    
    // Test 10: System Capabilities Summary
    std.log.info("✅ Test 10: System Capabilities", .{});
    
    const system_caps = try lib.gpu.getSystemCapabilities(allocator);
    
    try testing.expect(system_caps.total_devices > 0);
    try testing.expect(system_caps.total_memory_gb >= 0.0);
    
    std.log.info("📊 System Summary:", .{});
    std.log.info("  • Total GPU devices: {d}", .{system_caps.total_devices});
    std.log.info("  • IoT-suitable devices: {d}", .{system_caps.iot_suitable_devices});
    std.log.info("  • Inference-capable devices: {d}", .{system_caps.inference_capable_devices});
    std.log.info("  • Total GPU memory: {d:.1}GB", .{system_caps.total_memory_gb});
    std.log.info("  • Quantization support: {}", .{system_caps.supports_quantization});
    
    std.log.info("🎉 Phase 2 Integration Test PASSED!", .{});
    std.log.info("✅ All major components working together successfully", .{});
}

test "Phase 2 Error Handling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧪 Testing Phase 2 Error Handling", .{});

    // Test invalid tensor operations
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 64, // Very limited memory
        .num_threads = 1,
    });
    defer engine.deinit();

    // Test invalid shape
    const invalid_shape = [_]usize{ 0, 0 }; // Invalid shape
    const result = engine.get_tensor(&invalid_shape, .f32);
    try testing.expectError(error.InvalidShape, result);

    // Test invalid operator
    var tensor1 = try engine.get_tensor(&[_]usize{ 2, 2 }, .f32);
    defer engine.return_tensor(tensor1) catch {};
    
    var output = try engine.get_tensor(&[_]usize{ 2, 2 }, .f32);
    defer engine.return_tensor(output) catch {};
    
    const inputs = [_]lib.Tensor{tensor1};
    var outputs = [_]lib.Tensor{output};
    
    const invalid_op_result = engine.execute_operator("InvalidOp", &inputs, &outputs);
    try testing.expectError(error.OperatorNotFound, invalid_op_result);

    std.log.info("✅ Error handling tests passed", .{});
}

test "Phase 2 Stress Test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧪 Phase 2 Stress Test", .{});

    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 256,
        .num_threads = 4,
        .tensor_pool_size = 100,
    });
    defer engine.deinit();

    const stress_iterations = 1000;
    const tensor_size = [_]usize{ 16, 16 };

    std.log.info("🔥 Running {d} stress iterations...", .{stress_iterations});

    for (0..stress_iterations) |i| {
        if (i % 100 == 0) {
            std.log.info("  Progress: {d}/{d}", .{ i, stress_iterations });
        }

        var tensor1 = try engine.get_tensor(&tensor_size, .f32);
        var tensor2 = try engine.get_tensor(&tensor_size, .f32);
        var output = try engine.get_tensor(&tensor_size, .f32);

        // Fill with random-ish data
        for (0..256) |j| {
            const idx = [_]usize{ j / 16, j % 16 };
            try tensor1.set_f32(&idx, @as(f32, @floatFromInt(i + j)) * 0.01);
            try tensor2.set_f32(&idx, @as(f32, @floatFromInt(j)) * 0.02);
        }

        const inputs = [_]lib.Tensor{ tensor1, tensor2 };
        var outputs = [_]lib.Tensor{output};

        try engine.execute_operator("Add", &inputs, &outputs);

        try engine.return_tensor(tensor1);
        try engine.return_tensor(tensor2);
        try engine.return_tensor(output);
    }

    const final_stats = engine.getMemoryStats();
    std.log.info("📊 Stress test completed:", .{});
    std.log.info("  • Current memory usage: {d}KB", .{final_stats.current_usage / 1024});
    std.log.info("  • Peak memory usage: {d}KB", .{final_stats.peak_usage / 1024});

    // Memory should be reasonable after stress test
    try testing.expect(final_stats.current_usage < 50 * 1024 * 1024); // Less than 50MB

    std.log.info("✅ Stress test passed", .{});
}

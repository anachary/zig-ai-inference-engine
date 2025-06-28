const std = @import("std");
const lib = @import("zig-ai-engine");

/// Phase 2 Performance Benchmarks
/// Comprehensive benchmarking of all Phase 2 components for IoT and data security applications

const BenchmarkResult = struct {
    name: []const u8,
    iterations: u32,
    total_time_ms: f64,
    avg_time_ms: f64,
    ops_per_second: f64,
    memory_usage_kb: f64,
};

const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    engine: lib.Engine,
    gpu_context: ?lib.gpu.GPUContext,
    results: std.ArrayList(BenchmarkResult),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        var engine = try lib.Engine.init(allocator, .{
            .max_memory_mb = 1024,
            .num_threads = 4,
            .enable_profiling = true,
            .tensor_pool_size = 200,
        });

        var gpu_context = lib.gpu.createOptimalContext(allocator) catch |err| {
            std.log.warn("GPU context creation failed: {}, using CPU only", .{err});
            null;
        };

        return Self{
            .allocator = allocator,
            .engine = engine,
            .gpu_context = gpu_context,
            .results = std.ArrayList(BenchmarkResult).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.gpu_context) |*ctx| {
            ctx.deinit();
        }
        self.engine.deinit();
        self.results.deinit();
    }

    pub fn runAllBenchmarks(self: *Self) !void {
        std.log.info("🚀 Starting Phase 2 Performance Benchmarks", .{});
        std.log.info("===========================================", .{});

        // Core tensor operations
        try self.benchmarkTensorOperations();
        try self.benchmarkEnhancedOperators();
        try self.benchmarkMemoryManagement();
        
        // GPU operations (if available)
        if (self.gpu_context != null) {
            try self.benchmarkGPUOperations();
        }
        
        // System integration
        try self.benchmarkComputationGraph();
        try self.benchmarkJSONProcessing();
        try self.benchmarkConcurrentOperations();
        
        // IoT-specific benchmarks
        try self.benchmarkIoTScenarios();
        
        // Security-focused benchmarks
        try self.benchmarkSecurityScenarios();

        self.printSummary();
    }

    fn benchmarkTensorOperations(self: *Self) !void {
        std.log.info("📊 Benchmarking Core Tensor Operations", .{});

        const iterations = 10000;
        const shapes = [_][]const usize{
            &[_]usize{ 4, 4 },     // Small tensors (IoT)
            &[_]usize{ 16, 16 },   // Medium tensors
            &[_]usize{ 64, 64 },   // Large tensors
        };

        for (shapes, 0..) |shape, i| {
            const shape_name = switch (i) {
                0 => "small",
                1 => "medium", 
                2 => "large",
                else => "unknown",
            };

            // Benchmark tensor creation/destruction
            const start_time = std.time.nanoTimestamp();
            var memory_before = self.engine.getMemoryStats().current_usage;

            for (0..iterations) |_| {
                var tensor = try self.engine.get_tensor(shape, .f32);
                try self.engine.return_tensor(tensor);
            }

            const end_time = std.time.nanoTimestamp();
            var memory_after = self.engine.getMemoryStats().current_usage;

            const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
            const avg_time_ms = total_time_ms / @as(f64, @floatFromInt(iterations));
            const ops_per_second = 1000.0 / avg_time_ms;
            const memory_usage_kb = @as(f64, @floatFromInt(memory_after - memory_before)) / 1024.0;

            try self.results.append(BenchmarkResult{
                .name = try std.fmt.allocPrint(self.allocator, "tensor_ops_{s}", .{shape_name}),
                .iterations = iterations,
                .total_time_ms = total_time_ms,
                .avg_time_ms = avg_time_ms,
                .ops_per_second = ops_per_second,
                .memory_usage_kb = memory_usage_kb,
            });

            std.log.info("  • {s} tensors: {d:.3}ms avg, {d:.0} ops/sec", .{ shape_name, avg_time_ms, ops_per_second });
        }
    }

    fn benchmarkEnhancedOperators(self: *Self) !void {
        std.log.info("📊 Benchmarking Enhanced Operators", .{});

        const iterations = 1000;
        const shape = [_]usize{ 32, 32 };
        const operators = [_][]const u8{ "Add", "Mul", "ReLU", "MatMul" };

        for (operators) |op_name| {
            var tensor1 = try self.engine.get_tensor(&shape, .f32);
            defer self.engine.return_tensor(tensor1) catch {};
            
            var tensor2 = try self.engine.get_tensor(&shape, .f32);
            defer self.engine.return_tensor(tensor2) catch {};
            
            var output = try self.engine.get_tensor(&shape, .f32);
            defer self.engine.return_tensor(output) catch {};

            // Fill with test data
            for (0..shape[0]) |i| {
                for (0..shape[1]) |j| {
                    const idx = [_]usize{ i, j };
                    try tensor1.set_f32(&idx, @as(f32, @floatFromInt(i + j)) * 0.01);
                    try tensor2.set_f32(&idx, @as(f32, @floatFromInt(i * j)) * 0.01);
                }
            }

            const inputs = [_]lib.Tensor{ tensor1, tensor2 };
            var outputs = [_]lib.Tensor{output};

            const start_time = std.time.nanoTimestamp();

            for (0..iterations) |_| {
                self.engine.execute_operator(op_name, &inputs, &outputs) catch |err| {
                    if (err == error.OperatorNotFound) {
                        std.log.warn("Operator {s} not implemented, skipping", .{op_name});
                        break;
                    }
                    return err;
                };
            }

            const end_time = std.time.nanoTimestamp();
            const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
            const avg_time_ms = total_time_ms / @as(f64, @floatFromInt(iterations));
            const ops_per_second = 1000.0 / avg_time_ms;

            try self.results.append(BenchmarkResult{
                .name = try std.fmt.allocPrint(self.allocator, "operator_{s}", .{op_name}),
                .iterations = iterations,
                .total_time_ms = total_time_ms,
                .avg_time_ms = avg_time_ms,
                .ops_per_second = ops_per_second,
                .memory_usage_kb = 0.0,
            });

            std.log.info("  • {s}: {d:.3}ms avg, {d:.0} ops/sec", .{ op_name, avg_time_ms, ops_per_second });
        }
    }

    fn benchmarkMemoryManagement(self: *Self) !void {
        std.log.info("📊 Benchmarking Memory Management", .{});

        const iterations = 5000;
        const shape = [_]usize{ 16, 16 };

        // Benchmark memory pooling efficiency
        const start_time = std.time.nanoTimestamp();
        const memory_before = self.engine.getMemoryStats();

        for (0..iterations) |_| {
            var tensors: [10]lib.Tensor = undefined;
            
            // Allocate multiple tensors
            for (&tensors) |*tensor| {
                tensor.* = try self.engine.get_tensor(&shape, .f32);
            }
            
            // Return them all
            for (tensors) |tensor| {
                try self.engine.return_tensor(tensor);
            }
        }

        const end_time = std.time.nanoTimestamp();
        const memory_after = self.engine.getMemoryStats();

        const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        const avg_time_ms = total_time_ms / @as(f64, @floatFromInt(iterations));
        const ops_per_second = 1000.0 / avg_time_ms;
        const memory_growth_kb = @as(f64, @floatFromInt(memory_after.current_usage - memory_before.current_usage)) / 1024.0;

        try self.results.append(BenchmarkResult{
            .name = try self.allocator.dupe(u8, "memory_pooling"),
            .iterations = iterations,
            .total_time_ms = total_time_ms,
            .avg_time_ms = avg_time_ms,
            .ops_per_second = ops_per_second,
            .memory_usage_kb = memory_growth_kb,
        });

        std.log.info("  • Memory pooling: {d:.3}ms avg, {d:.1}KB growth", .{ avg_time_ms, memory_growth_kb });
    }

    fn benchmarkGPUOperations(self: *Self) !void {
        std.log.info("📊 Benchmarking GPU Operations", .{});

        if (self.gpu_context == null) return;

        const iterations = 100;
        const size = 1024;
        const data_size = size * @sizeOf(f32);

        var gpu_ctx = &self.gpu_context.?;
        const memory_type = gpu_ctx.getRecommendedMemoryType(true, false);

        const start_time = std.time.nanoTimestamp();

        for (0..iterations) |_| {
            var buffer1 = gpu_ctx.allocateBuffer(data_size, memory_type) catch continue;
            defer gpu_ctx.freeBuffer(buffer1) catch {};
            
            var buffer2 = gpu_ctx.allocateBuffer(data_size, memory_type) catch continue;
            defer gpu_ctx.freeBuffer(buffer2) catch {};
            
            var output_buffer = gpu_ctx.allocateBuffer(data_size, memory_type) catch continue;
            defer gpu_ctx.freeBuffer(output_buffer) catch {};

            var inputs = [_]lib.gpu.GPUBuffer{ buffer1, buffer2 };
            var outputs = [_]lib.gpu.GPUBuffer{output_buffer};

            gpu_ctx.executeOperator("Add", inputs[0..], outputs[0..]) catch continue;
        }

        const end_time = std.time.nanoTimestamp();
        const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        const avg_time_ms = total_time_ms / @as(f64, @floatFromInt(iterations));
        const ops_per_second = 1000.0 / avg_time_ms;

        try self.results.append(BenchmarkResult{
            .name = try self.allocator.dupe(u8, "gpu_operations"),
            .iterations = iterations,
            .total_time_ms = total_time_ms,
            .avg_time_ms = avg_time_ms,
            .ops_per_second = ops_per_second,
            .memory_usage_kb = 0.0,
        });

        std.log.info("  • GPU operations: {d:.3}ms avg, {d:.0} ops/sec", .{ avg_time_ms, ops_per_second });
    }

    fn benchmarkComputationGraph(self: *Self) !void {
        std.log.info("📊 Benchmarking Computation Graph", .{});

        const iterations = 100;
        const shape = [_]usize{ 8, 8 };

        const start_time = std.time.nanoTimestamp();

        for (0..iterations) |_| {
            var graph = try lib.formats.ComputationGraph.init(self.allocator);
            defer graph.deinit();

            try graph.addInput("input1", &shape, .f32);
            try graph.addInput("input2", &shape, .f32);
            try graph.addOutput("output", &shape, .f32);

            try graph.addNode(.{
                .id = "add_node",
                .op_type = "Add",
                .inputs = &[_][]const u8{ "input1", "input2" },
                .outputs = &[_][]const u8{"output"},
                .attributes = null,
            });
        }

        const end_time = std.time.nanoTimestamp();
        const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        const avg_time_ms = total_time_ms / @as(f64, @floatFromInt(iterations));
        const ops_per_second = 1000.0 / avg_time_ms;

        try self.results.append(BenchmarkResult{
            .name = try self.allocator.dupe(u8, "computation_graph"),
            .iterations = iterations,
            .total_time_ms = total_time_ms,
            .avg_time_ms = avg_time_ms,
            .ops_per_second = ops_per_second,
            .memory_usage_kb = 0.0,
        });

        std.log.info("  • Graph creation: {d:.3}ms avg, {d:.0} graphs/sec", .{ avg_time_ms, ops_per_second });
    }

    fn benchmarkJSONProcessing(self: *Self) !void {
        std.log.info("📊 Benchmarking JSON Processing", .{});

        const iterations = 1000;

        var json_processor = try lib.network.JSONProcessor.init(self.allocator);
        defer json_processor.deinit();

        const test_response = lib.network.InferResponse{
            .outputs = &[_]lib.network.InferResponse.TensorData{
                .{
                    .name = "benchmark_output",
                    .shape = &[_]usize{ 4, 4 },
                    .data = &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 },
                    .dtype = "float32",
                },
            },
            .model_id = "benchmark_model",
            .inference_time_ms = 123.45,
        };

        const start_time = std.time.nanoTimestamp();

        for (0..iterations) |_| {
            const json_str = try json_processor.serializeInferResponse(test_response);
            defer self.allocator.free(json_str);
        }

        const end_time = std.time.nanoTimestamp();
        const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        const avg_time_ms = total_time_ms / @as(f64, @floatFromInt(iterations));
        const ops_per_second = 1000.0 / avg_time_ms;

        try self.results.append(BenchmarkResult{
            .name = try self.allocator.dupe(u8, "json_processing"),
            .iterations = iterations,
            .total_time_ms = total_time_ms,
            .avg_time_ms = avg_time_ms,
            .ops_per_second = ops_per_second,
            .memory_usage_kb = 0.0,
        });

        std.log.info("  • JSON serialization: {d:.3}ms avg, {d:.0} ops/sec", .{ avg_time_ms, ops_per_second });
    }

    fn benchmarkConcurrentOperations(self: *Self) !void {
        std.log.info("📊 Benchmarking Concurrent Operations", .{});
        // TODO: Implement concurrent benchmarks when threading is added
        std.log.info("  • Concurrent benchmarks: TODO (requires threading support)", .{});
    }

    fn benchmarkIoTScenarios(self: *Self) !void {
        std.log.info("📊 Benchmarking IoT Scenarios", .{});

        // Simulate lightweight inference on small tensors (typical for IoT)
        const iterations = 2000;
        const iot_shape = [_]usize{ 8, 8 }; // Small tensors for IoT devices

        const start_time = std.time.nanoTimestamp();
        const memory_before = self.engine.getMemoryStats().current_usage;

        for (0..iterations) |_| {
            var input = try self.engine.get_tensor(&iot_shape, .f32);
            defer self.engine.return_tensor(input) catch {};
            
            var output = try self.engine.get_tensor(&iot_shape, .f32);
            defer self.engine.return_tensor(output) catch {};

            // Simulate simple inference pipeline
            const inputs = [_]lib.Tensor{input};
            var outputs = [_]lib.Tensor{output};
            
            try self.engine.execute_operator("ReLU", &inputs, &outputs);
        }

        const end_time = std.time.nanoTimestamp();
        const memory_after = self.engine.getMemoryStats().current_usage;

        const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        const avg_time_ms = total_time_ms / @as(f64, @floatFromInt(iterations));
        const ops_per_second = 1000.0 / avg_time_ms;
        const memory_usage_kb = @as(f64, @floatFromInt(memory_after - memory_before)) / 1024.0;

        try self.results.append(BenchmarkResult{
            .name = try self.allocator.dupe(u8, "iot_inference"),
            .iterations = iterations,
            .total_time_ms = total_time_ms,
            .avg_time_ms = avg_time_ms,
            .ops_per_second = ops_per_second,
            .memory_usage_kb = memory_usage_kb,
        });

        std.log.info("  • IoT inference: {d:.3}ms avg, {d:.0} inferences/sec", .{ avg_time_ms, ops_per_second });
    }

    fn benchmarkSecurityScenarios(self: *Self) !void {
        std.log.info("📊 Benchmarking Security Scenarios", .{});

        // Simulate secure inference with data isolation
        const iterations = 500;
        const secure_shape = [_]usize{ 16, 16 };

        const start_time = std.time.nanoTimestamp();

        for (0..iterations) |_| {
            // Create isolated tensors for secure processing
            var secure_input = try self.engine.get_tensor(&secure_shape, .f32);
            defer self.engine.return_tensor(secure_input) catch {};
            
            var secure_output = try self.engine.get_tensor(&secure_shape, .f32);
            defer self.engine.return_tensor(secure_output) catch {};

            // Fill with "sensitive" data
            for (0..secure_shape[0]) |i| {
                for (0..secure_shape[1]) |j| {
                    const idx = [_]usize{ i, j };
                    try secure_input.set_f32(&idx, @as(f32, @floatFromInt(i ^ j)) * 0.01); // XOR pattern
                }
            }

            const inputs = [_]lib.Tensor{secure_input};
            var outputs = [_]lib.Tensor{secure_output};
            
            try self.engine.execute_operator("ReLU", &inputs, &outputs);
            
            // Simulate secure cleanup (tensors are returned to pool)
        }

        const end_time = std.time.nanoTimestamp();
        const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        const avg_time_ms = total_time_ms / @as(f64, @floatFromInt(iterations));
        const ops_per_second = 1000.0 / avg_time_ms;

        try self.results.append(BenchmarkResult{
            .name = try self.allocator.dupe(u8, "security_inference"),
            .iterations = iterations,
            .total_time_ms = total_time_ms,
            .avg_time_ms = avg_time_ms,
            .ops_per_second = ops_per_second,
            .memory_usage_kb = 0.0,
        });

        std.log.info("  • Security inference: {d:.3}ms avg, {d:.0} secure ops/sec", .{ avg_time_ms, ops_per_second });
    }

    fn printSummary(self: *Self) void {
        std.log.info("\n🎯 Phase 2 Benchmark Summary", .{});
        std.log.info("============================", .{});

        var total_ops: f64 = 0;
        var fastest_op: ?BenchmarkResult = null;
        var slowest_op: ?BenchmarkResult = null;

        for (self.results.items) |result| {
            total_ops += result.ops_per_second;
            
            if (fastest_op == null or result.ops_per_second > fastest_op.?.ops_per_second) {
                fastest_op = result;
            }
            
            if (slowest_op == null or result.ops_per_second < slowest_op.?.ops_per_second) {
                slowest_op = result;
            }

            std.log.info("• {s}: {d:.3}ms avg ({d:.0} ops/sec)", .{
                result.name,
                result.avg_time_ms,
                result.ops_per_second,
            });
        }

        if (fastest_op) |fastest| {
            std.log.info("\n🏆 Fastest: {s} ({d:.0} ops/sec)", .{ fastest.name, fastest.ops_per_second });
        }
        
        if (slowest_op) |slowest| {
            std.log.info("🐌 Slowest: {s} ({d:.0} ops/sec)", .{ slowest.name, slowest.ops_per_second });
        }

        const avg_performance = total_ops / @as(f64, @floatFromInt(self.results.items.len));
        std.log.info("📊 Average performance: {d:.0} ops/sec", .{avg_performance});
        
        std.log.info("\n✅ Phase 2 benchmarks complete!", .{});
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var benchmark_suite = try BenchmarkSuite.init(allocator);
    defer benchmark_suite.deinit();

    try benchmark_suite.runAllBenchmarks();
}

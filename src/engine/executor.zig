const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");
const operators = @import("operators.zig");
const registry = @import("registry.zig");
const formats = @import("../formats/model.zig");
const graph = @import("graph.zig");

pub const ExecutorError = error{
    ModelNotLoaded,
    InvalidInput,
    ExecutionFailed,
    OutOfMemory,
};

pub const ExecutionStats = struct {
    total_time_ms: f64,
    node_times_ms: []f64,
    memory_peak_mb: f64,
    nodes_executed: u32,

    pub fn init(allocator: Allocator, num_nodes: usize) !ExecutionStats {
        return ExecutionStats{
            .total_time_ms = 0.0,
            .node_times_ms = try allocator.alloc(f64, num_nodes),
            .memory_peak_mb = 0.0,
            .nodes_executed = 0,
        };
    }

    pub fn deinit(self: *ExecutionStats, allocator: Allocator) void {
        allocator.free(self.node_times_ms);
    }

    pub fn print(self: *const ExecutionStats) void {
        std.log.info("=== Execution Statistics ===", .{});
        std.log.info("Total time: {d:.2} ms", .{self.total_time_ms});
        std.log.info("Nodes executed: {d}", .{self.nodes_executed});
        std.log.info("Peak memory: {d:.1} MB", .{self.memory_peak_mb});
        std.log.info("Average time per node: {d:.2} ms", .{if (self.nodes_executed > 0) self.total_time_ms / @as(f64, @floatFromInt(self.nodes_executed)) else 0.0});
    }
};

pub const ModelExecutor = struct {
    allocator: Allocator,
    model: ?*formats.Model,
    graph_executor: graph.GraphExecutor,
    optimizer: graph.GraphOptimizer,
    execution_stats: ?ExecutionStats,
    enable_profiling: bool,

    pub fn init(allocator: Allocator, operator_registry: *registry.OperatorRegistry, enable_profiling: bool) ModelExecutor {
        return ModelExecutor{
            .allocator = allocator,
            .model = null,
            .graph_executor = graph.GraphExecutor.init(allocator, operator_registry),
            .optimizer = graph.GraphOptimizer.init(allocator),
            .execution_stats = null,
            .enable_profiling = enable_profiling,
        };
    }

    pub fn deinit(self: *ModelExecutor) void {
        if (self.execution_stats) |*stats| {
            stats.deinit(self.allocator);
        }
        self.graph_executor.deinit();
    }

    pub fn loadModel(self: *ModelExecutor, model: *formats.Model) !void {
        self.model = model;

        // Optimize the model graph
        try self.optimizer.optimize(&model.graph);

        // Prepare the graph executor
        try self.graph_executor.prepare(&model.graph);

        // Initialize execution stats if profiling is enabled
        if (self.enable_profiling) {
            self.execution_stats = try ExecutionStats.init(self.allocator, model.graph.nodes.items.len);
        }

        std.log.info("Model loaded and prepared for execution", .{});
    }

    pub fn execute(self: *ModelExecutor, inputs: []tensor.Tensor) ![]tensor.Tensor {
        if (self.model == null) {
            return ExecutorError.ModelNotLoaded;
        }

        const model = self.model.?;

        // Validate inputs
        if (inputs.len != model.graph.inputs.items.len) {
            std.log.err("Expected {} inputs, got {}", .{ model.graph.inputs.items.len, inputs.len });
            return ExecutorError.InvalidInput;
        }

        for (model.graph.inputs.items, inputs) |input_spec, input_tensor| {
            if (!input_spec.isCompatible(&input_tensor)) {
                std.log.err("Input tensor incompatible with spec: {s}", .{input_spec.name});
                return ExecutorError.InvalidInput;
            }
        }

        // Execute the graph
        const start_time = std.time.nanoTimestamp();

        const outputs = self.graph_executor.execute(&model.graph, inputs) catch |err| {
            std.log.err("Graph execution failed: {}", .{err});
            return ExecutorError.ExecutionFailed;
        };

        const end_time = std.time.nanoTimestamp();

        // Update execution stats
        if (self.execution_stats) |*stats| {
            stats.total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
            stats.nodes_executed = @intCast(model.graph.nodes.items.len);
            // TODO: Track actual memory usage
            stats.memory_peak_mb = 100.0; // Placeholder
        }

        return outputs;
    }

    pub fn getExecutionStats(self: *const ModelExecutor) ?*const ExecutionStats {
        if (self.execution_stats) |*stats| {
            return stats;
        }
        return null;
    }

    pub fn benchmark(self: *ModelExecutor, inputs: []tensor.Tensor, num_iterations: u32) !BenchmarkResult {
        if (self.model == null) {
            return ExecutorError.ModelNotLoaded;
        }

        var times = std.ArrayList(f64).init(self.allocator);
        defer times.deinit();

        var total_time: f64 = 0.0;
        var min_time: f64 = std.math.floatMax(f64);
        var max_time: f64 = 0.0;

        std.log.info("Running benchmark with {} iterations...", .{num_iterations});

        var i: u32 = 0;
        while (i < num_iterations) : (i += 1) {
            const start_time = std.time.nanoTimestamp();

            const outputs = try self.execute(inputs);

            // Clean up outputs
            for (outputs) |output| {
                var mutable_output = output;
                mutable_output.deinit();
            }
            self.allocator.free(outputs);

            const end_time = std.time.nanoTimestamp();
            const iteration_time = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

            try times.append(iteration_time);
            total_time += iteration_time;
            min_time = @min(min_time, iteration_time);
            max_time = @max(max_time, iteration_time);

            if (i % 10 == 0) {
                std.log.info("Completed {} iterations", .{i + 1});
            }
        }

        // Calculate statistics
        const avg_time = total_time / @as(f64, @floatFromInt(num_iterations));

        // Calculate median
        std.sort.heap(f64, times.items, {}, comptime std.sort.asc(f64));
        const median_time = if (times.items.len % 2 == 0)
            (times.items[times.items.len / 2 - 1] + times.items[times.items.len / 2]) / 2.0
        else
            times.items[times.items.len / 2];

        // Calculate standard deviation
        var variance: f64 = 0.0;
        for (times.items) |time| {
            const diff = time - avg_time;
            variance += diff * diff;
        }
        variance /= @as(f64, @floatFromInt(num_iterations));
        const std_dev = @sqrt(variance);

        return BenchmarkResult{
            .num_iterations = num_iterations,
            .avg_time_ms = avg_time,
            .min_time_ms = min_time,
            .max_time_ms = max_time,
            .median_time_ms = median_time,
            .std_dev_ms = std_dev,
            .throughput_fps = 1000.0 / avg_time,
        };
    }
};

pub const BenchmarkResult = struct {
    num_iterations: u32,
    avg_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    median_time_ms: f64,
    std_dev_ms: f64,
    throughput_fps: f64,

    pub fn print(self: *const BenchmarkResult) void {
        std.log.info("=== Benchmark Results ===", .{});
        std.log.info("Iterations: {d}", .{self.num_iterations});
        std.log.info("Average time: {d:.2} ms", .{self.avg_time_ms});
        std.log.info("Median time: {d:.2} ms", .{self.median_time_ms});
        std.log.info("Min time: {d:.2} ms", .{self.min_time_ms});
        std.log.info("Max time: {d:.2} ms", .{self.max_time_ms});
        std.log.info("Std deviation: {d:.2} ms", .{self.std_dev_ms});
        std.log.info("Throughput: {d:.1} FPS", .{self.throughput_fps});
    }
};

pub const ParallelExecutor = struct {
    allocator: Allocator,
    num_threads: u32,
    thread_pool: ?std.Thread.Pool,

    pub fn init(allocator: Allocator, num_threads: u32) !ParallelExecutor {
        var thread_pool = std.Thread.Pool{};
        try thread_pool.init(.{ .allocator = allocator, .n_jobs = num_threads });

        return ParallelExecutor{
            .allocator = allocator,
            .num_threads = num_threads,
            .thread_pool = thread_pool,
        };
    }

    pub fn deinit(self: *ParallelExecutor) void {
        if (self.thread_pool) |*pool| {
            pool.deinit();
        }
    }

    pub fn executeBatch(self: *ParallelExecutor, executor: *ModelExecutor, batch_inputs: [][]tensor.Tensor) ![][]tensor.Tensor {
        // For now, execute sequentially
        // TODO: Implement parallel batch execution
        var batch_outputs = try self.allocator.alloc([]tensor.Tensor, batch_inputs.len);

        for (batch_inputs, 0..) |inputs, i| {
            batch_outputs[i] = try executor.execute(inputs);
        }

        return batch_outputs;
    }
};

// Utility functions for execution
pub fn createDummyInputs(allocator: Allocator, input_specs: []const formats.TensorSpec) ![]tensor.Tensor {
    var inputs = try allocator.alloc(tensor.Tensor, input_specs.len);

    for (input_specs, 0..) |spec, i| {
        // Convert i32 shape to usize shape
        var shape = try allocator.alloc(usize, spec.shape.len);
        defer allocator.free(shape);

        for (spec.shape, 0..) |dim, j| {
            shape[j] = if (dim == -1) 1 else @intCast(dim); // Use 1 for dynamic dimensions
        }

        inputs[i] = try tensor.Tensor.init(allocator, shape, spec.dtype);

        // Fill with dummy data
        const numel = inputs[i].numel();
        var k: usize = 0;
        while (k < numel) : (k += 1) {
            try inputs[i].set_f32_flat(k, @as(f32, @floatFromInt(k % 10)) * 0.1);
        }
    }

    return inputs;
}

pub fn cleanupTensors(allocator: Allocator, tensors: []tensor.Tensor) void {
    for (tensors) |t| {
        var mutable_tensor = t;
        mutable_tensor.deinit();
    }
    allocator.free(tensors);
}

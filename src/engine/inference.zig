const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");
const memory = @import("../memory/manager.zig");
const pool = @import("../memory/pool.zig");
const operators = @import("operators.zig");
const registry = @import("registry.zig");
const network = @import("../network/server.zig");
const formats = @import("../formats/model.zig");
const onnx = @import("../formats/onnx/parser.zig");
const executor = @import("executor.zig");
const optimizer = @import("optimizer.zig");

pub const InferenceError = error{
    ModelNotLoaded,
    InvalidInput,
    InferenceFailure,
    OutOfMemory,
};

pub const InferenceEngine = struct {
    allocator: Allocator,
    memory_manager: memory.MemoryManager,
    tensor_pool: pool.TensorPool,
    memory_tracker: pool.MemoryTracker,
    operator_registry: registry.OperatorRegistry,
    execution_context: registry.ExecutionContext,
    server: ?network.HTTPServer,
    model_loaded: bool,
    loaded_model: ?formats.Model,
    onnx_parser: onnx.ONNXParser,
    model_executor: executor.ModelExecutor,
    graph_optimizer: optimizer.GraphOptimizer,
    config: Config,

    const Self = @This();

    const Config = struct {
        max_memory_mb: u32,
        num_threads: ?u32,
        enable_profiling: bool = false,
        tensor_pool_size: usize = 100,
    };

    pub fn init(allocator: Allocator, config: struct {
        max_memory_mb: u32,
        num_threads: ?u32,
        enable_profiling: bool = false,
        tensor_pool_size: usize = 100,
    }) !Self {
        const engine_config = Config{
            .max_memory_mb = config.max_memory_mb,
            .num_threads = config.num_threads,
            .enable_profiling = config.enable_profiling,
            .tensor_pool_size = config.tensor_pool_size,
        };

        var self = Self{
            .allocator = allocator,
            .memory_manager = memory.MemoryManager.init(allocator),
            .tensor_pool = pool.TensorPool.init(allocator, config.tensor_pool_size),
            .memory_tracker = pool.MemoryTracker.init(allocator),
            .operator_registry = registry.OperatorRegistry.init(allocator),
            .execution_context = undefined, // Will be initialized below
            .server = null,
            .model_loaded = false,
            .loaded_model = null,
            .onnx_parser = onnx.ONNXParser.init(allocator),
            .model_executor = undefined, // Will be initialized below
            .graph_optimizer = optimizer.GraphOptimizer.init(allocator, .{}),
            .config = engine_config,
        };

        try self.operator_registry.register_builtin_operators();
        self.execution_context = registry.ExecutionContext.init(allocator);
        self.model_executor = executor.ModelExecutor.init(allocator, &self.operator_registry, engine_config.enable_profiling);

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.server) |*server| {
            server.deinit();
        }
        if (self.loaded_model) |*model| {
            model.deinit();
        }
        self.model_executor.deinit();
        self.execution_context.deinit();
        self.operator_registry.deinit();
        self.tensor_pool.deinit();
        self.memory_tracker.deinit();
        self.memory_manager.deinit();
    }

    pub fn loadModel(self: *Self, model_path: []const u8) !void {
        std.log.info("Loading model from: {s}", .{model_path});

        // Determine model format from file extension
        const format = formats.ModelFormat.fromPath(model_path);

        switch (format) {
            .onnx => {
                std.log.info("Detected ONNX model format", .{});
                self.loaded_model = try self.onnx_parser.parseFile(model_path);
                try self.loaded_model.?.validate();

                // Load model into executor
                try self.model_executor.loadModel(&self.loaded_model.?);

                self.model_loaded = true;
                std.log.info("ONNX model loaded and prepared for execution", .{});
            },
            else => {
                std.log.err("Unsupported model format: {}", .{format});
                return InferenceError.ModelNotLoaded;
            },
        }
    }

    pub fn infer(self: *Self, input: tensor.Tensor) !tensor.Tensor {
        if (!self.model_loaded) return InferenceError.ModelNotLoaded;

        const inputs = [_]tensor.Tensor{input};
        const outputs = try self.model_executor.execute(&inputs);

        if (outputs.len == 0) {
            return InferenceError.ModelNotLoaded;
        }

        // Return first output (caller owns the memory)
        return outputs[0];
    }

    pub fn inferBatch(self: *Self, inputs: []tensor.Tensor) ![]tensor.Tensor {
        if (!self.model_loaded) return InferenceError.ModelNotLoaded;

        return self.model_executor.execute(inputs);
    }

    pub fn startServer(self: *Self, port: u16) !void {
        self.server = try network.HTTPServer.init(self.allocator, port);
        self.server.?.setInferenceEngine(self);
        try self.server.?.start();
    }

    /// Execute a single operator
    pub fn execute_operator(
        self: *Self,
        op_name: []const u8,
        inputs: []const tensor.Tensor,
        outputs: []tensor.Tensor,
    ) !void {
        try self.execution_context.execute_operator(&self.operator_registry, op_name, inputs, outputs);
    }

    /// Get a tensor from the pool
    pub fn get_tensor(self: *Self, shape: []const usize, dtype: tensor.DataType) !tensor.Tensor {
        return self.tensor_pool.get_tensor(shape, dtype);
    }

    /// Return a tensor to the pool
    pub fn return_tensor(self: *Self, t: tensor.Tensor) !void {
        try self.tensor_pool.return_tensor(t);
    }

    /// Force cleanup of a tensor (immediate deallocation)
    pub fn cleanup_tensor(self: *Self, t: tensor.Tensor) void {
        self.tensor_pool.cleanup_tensor(t);
    }

    /// Get engine statistics
    pub fn get_stats(self: *const Self) EngineStats {
        const pool_stats = self.tensor_pool.get_stats();
        const memory_stats = self.memory_tracker.get_stats();
        const registry_stats = self.operator_registry.get_stats();

        return EngineStats{
            .model_loaded = self.model_loaded,
            .tensor_pool = pool_stats,
            .memory = memory_stats,
            .operators = registry_stats,
        };
    }

    /// Reset temporary resources
    pub fn reset_temp_resources(self: *Self) void {
        self.execution_context.reset_temp_tensors();
        self.memory_manager.reset_temporary();
    }
};

pub const EngineStats = struct {
    model_loaded: bool,
    tensor_pool: pool.PoolStats,
    memory: pool.MemoryStats,
    operators: registry.RegistryStats,
};

// TODO: Implement computation graph, model loading, etc.

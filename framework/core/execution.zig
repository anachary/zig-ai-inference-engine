const std = @import("std");
const Allocator = std.mem.Allocator;
const interfaces = @import("interfaces.zig");
const registry = @import("../operators/registry.zig");

const Tensor = interfaces.Tensor;
const Attributes = interfaces.Attributes;
const ExecutionContext = interfaces.ExecutionContext;
const FrameworkError = interfaces.FrameworkError;
const Profiler = interfaces.Profiler;
const MemoryManager = interfaces.MemoryManager;
const OperatorRegistry = registry.OperatorRegistry;

/// Execution engine for running computational graphs
pub const ExecutionEngine = struct {
    allocator: Allocator,
    operator_registry: *OperatorRegistry,
    memory_manager: MemoryManager,
    profiler: ?Profiler,
    context: ExecutionContext,
    optimization_enabled: bool,

    const Self = @This();

    /// Execution configuration
    pub const Config = struct {
        device: ExecutionContext.Device = .auto,
        optimization_level: ExecutionContext.OptimizationLevel = .basic,
        enable_profiling: bool = false,
        enable_memory_tracking: bool = true,
        max_memory_mb: ?usize = null,
        num_threads: ?u32 = null,
    };

    /// Computational graph node
    pub const GraphNode = struct {
        id: u32,
        operator_name: []const u8,
        operator_version: ?[]const u8,
        inputs: []const u32,  // Node IDs
        outputs: []const u32, // Node IDs
        attributes: Attributes,
    };

    /// Computational graph
    pub const Graph = struct {
        nodes: []const GraphNode,
        inputs: []const GraphInput,
        outputs: []const GraphOutput,
        values: std.HashMap(u32, Tensor),
        allocator: Allocator,

        pub const GraphInput = struct {
            id: u32,
            name: []const u8,
            shape: []const usize,
            dtype: Tensor.DataType,
        };

        pub const GraphOutput = struct {
            id: u32,
            name: []const u8,
        };

        pub fn init(allocator: Allocator) Graph {
            return Graph{
                .nodes = &.{},
                .inputs = &.{},
                .outputs = &.{},
                .values = std.HashMap(u32, Tensor).init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Graph) void {
            // Clean up tensors
            var value_iter = self.values.iterator();
            while (value_iter.next()) |entry| {
                entry.value_ptr.deinit();
            }
            self.values.deinit();
        }

        pub fn setInput(self: *Graph, input_id: u32, tensor: Tensor) !void {
            try self.values.put(input_id, tensor);
        }

        pub fn getOutput(self: *const Graph, output_id: u32) ?Tensor {
            return self.values.get(output_id);
        }
    };

    /// Execution plan for optimized execution
    pub const ExecutionPlan = struct {
        nodes: []const PlanNode,
        memory_plan: MemoryPlan,
        parallelization_plan: ParallelizationPlan,
        allocator: Allocator,

        pub const PlanNode = struct {
            node_id: u32,
            operator_name: []const u8,
            operator_version: ?[]const u8,
            input_tensors: []const u32,
            output_tensors: []const u32,
            attributes: Attributes,
            optimization_hints: ?registry.OperatorInterface.OptimizationHint,
            execution_order: u32,
        };

        pub const MemoryPlan = struct {
            total_memory_required: usize,
            tensor_lifetimes: std.HashMap(u32, Lifetime),
            memory_reuse_map: std.HashMap(u32, u32),

            pub const Lifetime = struct {
                start_node: u32,
                end_node: u32,
            };
        };

        pub const ParallelizationPlan = struct {
            parallel_groups: []const []const u32,
            dependencies: std.HashMap(u32, []const u32),
        };

        pub fn deinit(self: *ExecutionPlan) void {
            self.memory_plan.tensor_lifetimes.deinit();
            self.memory_plan.memory_reuse_map.deinit();
            self.parallelization_plan.dependencies.deinit();
            self.allocator.free(self.nodes);
        }
    };

    /// Initialize execution engine
    pub fn init(allocator: Allocator, operator_registry: *OperatorRegistry, config: Config) !Self {
        var context = ExecutionContext.init(allocator);
        context.device = config.device;
        context.optimization_level = config.optimization_level;
        context.profiling_enabled = config.enable_profiling;

        var profiler: ?Profiler = null;
        if (config.enable_profiling) {
            profiler = Profiler.init(allocator);
        }

        return Self{
            .allocator = allocator,
            .operator_registry = operator_registry,
            .memory_manager = MemoryManager.init(allocator),
            .profiler = profiler,
            .context = context,
            .optimization_enabled = config.optimization_level != .none,
        };
    }

    /// Deinitialize execution engine
    pub fn deinit(self: *Self) void {
        if (self.profiler) |*profiler| {
            profiler.deinit();
        }
    }

    /// Execute a computational graph
    pub fn executeGraph(self: *Self, graph: *Graph) !void {
        if (self.profiler) |*profiler| {
            try profiler.startEvent("graph_execution");
        }
        defer if (self.profiler) |*profiler| {
            profiler.endEvent(self.memory_manager.getCurrentUsage());
        };

        if (self.optimization_enabled) {
            const plan = try self.createExecutionPlan(graph);
            defer plan.deinit();
            try self.executeOptimizedPlan(graph, &plan);
        } else {
            try self.executeGraphDirect(graph);
        }
    }

    /// Execute graph directly without optimization
    fn executeGraphDirect(self: *Self, graph: *Graph) !void {
        // Simple topological execution
        for (graph.nodes) |node| {
            try self.executeNode(graph, &node);
        }
    }

    /// Execute a single node
    fn executeNode(self: *Self, graph: *Graph, node: *const GraphNode) !void {
        if (self.profiler) |*profiler| {
            const event_name = try std.fmt.allocPrint(self.allocator, "node_{s}", .{node.operator_name});
            defer self.allocator.free(event_name);
            try profiler.startEvent(event_name);
        }
        defer if (self.profiler) |*profiler| {
            profiler.endEvent(self.memory_manager.getCurrentUsage());
        };

        // Collect input tensors
        var input_tensors = try self.allocator.alloc(Tensor, node.inputs.len);
        defer self.allocator.free(input_tensors);

        for (node.inputs, 0..) |input_id, i| {
            input_tensors[i] = graph.values.get(input_id) orelse return FrameworkError.InvalidInput;
        }

        // Infer output shapes
        var input_shapes = try self.allocator.alloc([]const usize, input_tensors.len);
        defer self.allocator.free(input_shapes);

        for (input_tensors, 0..) |tensor, i| {
            input_shapes[i] = tensor.shape;
        }

        const output_shapes = try self.operator_registry.inferShapes(
            node.operator_name,
            input_shapes,
            &node.attributes,
            node.operator_version,
        );
        defer {
            for (output_shapes) |shape| {
                self.allocator.free(shape);
            }
            self.allocator.free(output_shapes);
        }

        // Create output tensors
        var output_tensors = try self.allocator.alloc(Tensor, node.outputs.len);
        defer self.allocator.free(output_tensors);

        for (node.outputs, 0..) |output_id, i| {
            if (i < output_shapes.len) {
                // Use inferred data type (for now, use same as first input)
                const dtype = if (input_tensors.len > 0) input_tensors[0].dtype else .f32;
                output_tensors[i] = try Tensor.init(self.allocator, output_shapes[i], dtype);
                try graph.values.put(output_id, output_tensors[i]);
            }
        }

        // Execute the operator
        try self.operator_registry.executeOperator(
            node.operator_name,
            input_tensors,
            output_tensors,
            &node.attributes,
            &self.context,
            node.operator_version,
        );
    }

    /// Create execution plan for optimized execution
    fn createExecutionPlan(self: *Self, graph: *const Graph) !ExecutionPlan {
        _ = self;
        _ = graph;
        
        // TODO: Implement execution plan creation
        // This would include:
        // 1. Topological sorting
        // 2. Memory optimization
        // 3. Operator fusion
        // 4. Parallelization analysis
        
        return ExecutionPlan{
            .nodes = &.{},
            .memory_plan = ExecutionPlan.MemoryPlan{
                .total_memory_required = 0,
                .tensor_lifetimes = std.HashMap(u32, ExecutionPlan.MemoryPlan.Lifetime).init(self.allocator),
                .memory_reuse_map = std.HashMap(u32, u32).init(self.allocator),
            },
            .parallelization_plan = ExecutionPlan.ParallelizationPlan{
                .parallel_groups = &.{},
                .dependencies = std.HashMap(u32, []const u32).init(self.allocator),
            },
            .allocator = self.allocator,
        };
    }

    /// Execute optimized execution plan
    fn executeOptimizedPlan(self: *Self, graph: *Graph, plan: *const ExecutionPlan) !void {
        _ = self;
        _ = graph;
        _ = plan;
        
        // TODO: Implement optimized execution
        // This would execute nodes according to the optimized plan
        std.log.info("Optimized execution not yet implemented, falling back to direct execution");
    }

    /// Get execution statistics
    pub fn getExecutionStats(self: *const Self) ExecutionStats {
        return ExecutionStats{
            .total_memory_used = self.memory_manager.getCurrentUsage(),
            .peak_memory_used = self.memory_manager.getPeakUsage(),
            .profiling_data = if (self.profiler) |profiler| profiler.getReport() else &.{},
        };
    }

    /// Execution statistics
    pub const ExecutionStats = struct {
        total_memory_used: usize,
        peak_memory_used: usize,
        profiling_data: []const Profiler.Event,
    };

    /// Validate graph before execution
    pub fn validateGraph(self: *Self, graph: *const Graph) !void {
        // Check that all operators are available
        for (graph.nodes) |node| {
            if (!self.operator_registry.hasOperator(node.operator_name, node.operator_version)) {
                std.log.err("Operator not found: {s}", .{node.operator_name});
                return FrameworkError.OperatorNotFound;
            }
        }

        // Check that all inputs are provided
        for (graph.inputs) |input| {
            if (!graph.values.contains(input.id)) {
                std.log.err("Graph input not provided: {s}", .{input.name});
                return FrameworkError.InvalidInput;
            }
        }

        // TODO: Add more validation:
        // - Check for cycles
        // - Validate tensor shapes and types
        // - Check operator constraints
    }

    /// Set device for execution
    pub fn setDevice(self: *Self, device: ExecutionContext.Device) void {
        self.context.device = device;
    }

    /// Enable/disable profiling
    pub fn setProfilingEnabled(self: *Self, enabled: bool) !void {
        if (enabled and self.profiler == null) {
            self.profiler = Profiler.init(self.allocator);
        } else if (!enabled and self.profiler != null) {
            self.profiler.?.deinit();
            self.profiler = null;
        }
        self.context.profiling_enabled = enabled;
    }
};

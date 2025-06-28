const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");
const operators = @import("operators.zig");
const registry = @import("registry.zig");
const formats = @import("../formats/model.zig");
const onnx_nodes = @import("../formats/onnx/nodes.zig");

pub const GraphError = error{
    CyclicGraph,
    UnconnectedNode,
    MissingInput,
    InvalidTopology,
    ExecutionFailed,
    OutOfMemory,
};

pub const ExecutionPlan = struct {
    execution_order: []usize, // Indices into nodes array
    memory_plan: MemoryPlan,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ExecutionPlan {
        return ExecutionPlan{
            .execution_order = &[_]usize{},
            .memory_plan = MemoryPlan.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ExecutionPlan) void {
        if (self.execution_order.len > 0) {
            self.allocator.free(self.execution_order);
        }
        self.memory_plan.deinit();
    }
};

pub const MemoryPlan = struct {
    tensor_lifetimes: std.StringHashMap(Lifetime),
    peak_memory_usage: usize,
    allocator: Allocator,

    const Lifetime = struct {
        first_use: usize, // Step index
        last_use: usize, // Step index
        size_bytes: usize,
    };

    pub fn init(allocator: Allocator) MemoryPlan {
        return MemoryPlan{
            .tensor_lifetimes = std.StringHashMap(Lifetime).init(allocator),
            .peak_memory_usage = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MemoryPlan) void {
        var iter = self.tensor_lifetimes.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.tensor_lifetimes.deinit();
    }

    pub fn addTensor(self: *MemoryPlan, name: []const u8, first_use: usize, last_use: usize, size_bytes: usize) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        try self.tensor_lifetimes.put(name_copy, Lifetime{
            .first_use = first_use,
            .last_use = last_use,
            .size_bytes = size_bytes,
        });
    }

    pub fn computePeakMemory(self: *MemoryPlan) usize {
        // Simple algorithm: sum all overlapping tensors
        // TODO: Implement more sophisticated memory planning
        var peak: usize = 0;
        var iter = self.tensor_lifetimes.valueIterator();
        while (iter.next()) |lifetime| {
            peak += lifetime.size_bytes;
        }
        self.peak_memory_usage = peak;
        return peak;
    }
};

pub const GraphExecutor = struct {
    allocator: Allocator,
    execution_plan: ?ExecutionPlan,
    tensor_registry: std.StringHashMap(tensor.Tensor),
    operator_registry: *registry.OperatorRegistry,
    onnx_executor: onnx_nodes.NodeExecutor,

    pub fn init(allocator: Allocator, operator_registry: *registry.OperatorRegistry) GraphExecutor {
        return GraphExecutor{
            .allocator = allocator,
            .execution_plan = null,
            .tensor_registry = std.StringHashMap(tensor.Tensor).init(allocator),
            .operator_registry = operator_registry,
            .onnx_executor = onnx_nodes.NodeExecutor.init(allocator),
        };
    }

    pub fn deinit(self: *GraphExecutor) void {
        if (self.execution_plan) |*plan| {
            plan.deinit();
        }

        var iter = self.tensor_registry.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var t = entry.value_ptr;
            t.deinit();
        }
        self.tensor_registry.deinit();
    }

    pub fn prepare(self: *GraphExecutor, graph: *const formats.ComputationGraph) !void {
        // Create execution plan
        self.execution_plan = try self.createExecutionPlan(graph);

        // Pre-allocate tensors based on memory plan
        try self.preallocateTensors();

        std.log.info("Graph executor prepared with {} steps", .{self.execution_plan.?.execution_order.len});
    }

    pub fn execute(self: *GraphExecutor, graph: *const formats.ComputationGraph, inputs: []tensor.Tensor) ![]tensor.Tensor {
        if (self.execution_plan == null) {
            return GraphError.InvalidTopology;
        }

        // Set input tensors
        try self.setInputs(graph, inputs);

        // Execute nodes in planned order
        const plan = self.execution_plan.?;
        for (plan.execution_order) |node_idx| {
            try self.executeNode(graph, node_idx);
        }

        // Collect output tensors
        return self.collectOutputs(graph);
    }

    fn createExecutionPlan(self: *GraphExecutor, graph: *const formats.ComputationGraph) !ExecutionPlan {
        var plan = ExecutionPlan.init(self.allocator);

        // Topological sort using Kahn's algorithm
        var execution_order = std.ArrayList(usize).init(self.allocator);
        defer execution_order.deinit();

        var in_degree = try self.allocator.alloc(usize, graph.nodes.items.len);
        defer self.allocator.free(in_degree);

        // Initialize in-degrees
        for (in_degree) |*degree| {
            degree.* = 0;
        }

        // Count incoming edges for each node
        for (graph.edges.items) |edge| {
            // Find target node index
            for (graph.nodes.items, 0..) |node, i| {
                if (std.mem.eql(u8, node.id, edge.to_node)) {
                    in_degree[i] += 1;
                    break;
                }
            }
        }

        // Find nodes with no incoming edges
        var queue = std.ArrayList(usize).init(self.allocator);
        defer queue.deinit();

        for (in_degree, 0..) |degree, i| {
            if (degree == 0) {
                try queue.append(i);
            }
        }

        // Process nodes
        while (queue.items.len > 0) {
            const node_idx = queue.orderedRemove(0);
            try execution_order.append(node_idx);

            // Reduce in-degree of dependent nodes
            const current_node = graph.nodes.items[node_idx];
            for (graph.edges.items) |edge| {
                if (std.mem.eql(u8, edge.from_node, current_node.id)) {
                    // Find target node
                    for (graph.nodes.items, 0..) |node, i| {
                        if (std.mem.eql(u8, node.id, edge.to_node)) {
                            in_degree[i] -= 1;
                            if (in_degree[i] == 0) {
                                try queue.append(i);
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Check for cycles
        if (execution_order.items.len != graph.nodes.items.len) {
            return GraphError.CyclicGraph;
        }

        plan.execution_order = try execution_order.toOwnedSlice();
        return plan;
    }

    fn preallocateTensors(self: *GraphExecutor) !void {
        // TODO: Implement tensor preallocation based on memory plan
        _ = self;
        std.log.debug("Preallocating tensors for execution", .{});
    }

    fn setInputs(self: *GraphExecutor, graph: *const formats.ComputationGraph, inputs: []tensor.Tensor) !void {
        if (inputs.len != graph.inputs.items.len) {
            return GraphError.MissingInput;
        }

        for (graph.inputs.items, inputs) |input_spec, input_tensor| {
            if (!input_spec.isCompatible(&input_tensor)) {
                return GraphError.MissingInput;
            }

            const name_copy = try self.allocator.dupe(u8, input_spec.name);
            try self.tensor_registry.put(name_copy, input_tensor);
        }
    }

    fn executeNode(self: *GraphExecutor, graph: *const formats.ComputationGraph, node_idx: usize) !void {
        const node = &graph.nodes.items[node_idx];

        std.log.debug("Executing node: {s} (type: {s})", .{ node.id, node.op_type });

        // Collect input tensors
        var input_tensors = std.ArrayList(tensor.Tensor).init(self.allocator);
        defer input_tensors.deinit();

        for (node.inputs) |input_name| {
            if (self.tensor_registry.get(input_name)) |input_tensor| {
                try input_tensors.append(input_tensor);
            } else {
                std.log.err("Missing input tensor: {s}", .{input_name});
                return GraphError.MissingInput;
            }
        }

        // Create output tensors (for now, use same shape as first input)
        var output_tensors = std.ArrayList(tensor.Tensor).init(self.allocator);
        defer output_tensors.deinit();

        for (node.outputs) |output_name| {
            // For now, create output tensor with same shape as first input
            if (input_tensors.items.len > 0) {
                const input_shape = input_tensors.items[0].shape;
                var output_tensor = try tensor.Tensor.init(self.allocator, input_shape, .f32);
                try output_tensors.append(output_tensor);

                // Register output tensor
                const name_copy = try self.allocator.dupe(u8, output_name);
                try self.tensor_registry.put(name_copy, output_tensor);
            }
        }

        // Execute the node
        if (self.operator_registry.get(node.op_type)) |op| {
            // Use built-in operator
            try op.forward(input_tensors.items, output_tensors.items, self.allocator);
        } else {
            // Try ONNX node executor
            try self.onnx_executor.execute(node, input_tensors.items, output_tensors.items);
        }
    }

    fn collectOutputs(self: *GraphExecutor, graph: *const formats.ComputationGraph) ![]tensor.Tensor {
        var outputs = try self.allocator.alloc(tensor.Tensor, graph.outputs.items.len);

        for (graph.outputs.items, 0..) |output_spec, i| {
            if (self.tensor_registry.get(output_spec.name)) |output_tensor| {
                outputs[i] = output_tensor;
            } else {
                // Create dummy output tensor
                const shape = [_]usize{1};
                outputs[i] = try tensor.Tensor.init(self.allocator, &shape, .f32);
            }
        }

        return outputs;
    }
};

pub const GraphOptimizer = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) GraphOptimizer {
        return GraphOptimizer{
            .allocator = allocator,
        };
    }

    pub fn optimize(self: *GraphOptimizer, graph: *formats.ComputationGraph) !void {
        try self.eliminateDeadCode(graph);
        try self.fuseOperators(graph);
        try self.optimizeMemoryLayout(graph);
    }

    fn eliminateDeadCode(self: *GraphOptimizer, graph: *formats.ComputationGraph) !void {
        // Mark all nodes that contribute to outputs
        var reachable = std.ArrayList(bool).init(self.allocator);
        defer reachable.deinit();

        try reachable.resize(graph.nodes.items.len);
        for (reachable.items) |*item| {
            item.* = false;
        }

        // TODO: Implement backward traversal from outputs
        // For now, mark all nodes as reachable
        for (reachable.items) |*item| {
            item.* = true;
        }

        std.log.info("Dead code elimination: {} nodes remain reachable", .{graph.nodes.items.len});
    }

    fn fuseOperators(self: *GraphOptimizer, graph: *formats.ComputationGraph) !void {
        _ = self;
        _ = graph;

        // TODO: Implement operator fusion patterns
        // Common patterns:
        // - Conv + BatchNorm + ReLU
        // - MatMul + Add (bias)
        // - Add + ReLU

        std.log.info("Operator fusion: checking for fusion opportunities", .{});
    }

    fn optimizeMemoryLayout(self: *GraphOptimizer, graph: *formats.ComputationGraph) !void {
        _ = self;
        _ = graph;

        // TODO: Implement memory layout optimization
        // - Reorder operations to minimize memory usage
        // - Insert memory copy operations where beneficial

        std.log.info("Memory layout optimization: analyzing tensor layouts", .{});
    }
};

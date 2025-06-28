const std = @import("std");
const Allocator = std.mem.Allocator;
const model = @import("../model.zig");
const tensor = @import("../../core/tensor.zig");
const parser = @import("parser.zig");

pub const GraphError = error{
    CyclicGraph,
    UnconnectedNode,
    MissingInput,
    InvalidTopology,
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
        self.allocator.free(self.execution_order);
        self.memory_plan.deinit();
    }
};

pub const MemoryPlan = struct {
    tensor_lifetimes: std.StringHashMap(Lifetime),
    peak_memory_usage: usize,
    allocator: Allocator,
    
    const Lifetime = struct {
        first_use: usize, // Step index
        last_use: usize,  // Step index
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

pub const GraphOptimizer = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) GraphOptimizer {
        return GraphOptimizer{
            .allocator = allocator,
        };
    }
    
    pub fn optimize(self: *GraphOptimizer, graph: *model.ComputationGraph) !void {
        try self.eliminateDeadCode(graph);
        try self.fuseOperators(graph);
        try self.optimizeMemoryLayout(graph);
    }
    
    fn eliminateDeadCode(self: *GraphOptimizer, graph: *model.ComputationGraph) !void {
        _ = self;
        
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
    
    fn fuseOperators(self: *GraphOptimizer, graph: *model.ComputationGraph) !void {
        _ = self;
        _ = graph;
        
        // TODO: Implement operator fusion patterns
        // Common patterns:
        // - Conv + BatchNorm + ReLU
        // - MatMul + Add (bias)
        // - Add + ReLU
        
        std.log.info("Operator fusion: checking for fusion opportunities", .{});
    }
    
    fn optimizeMemoryLayout(self: *GraphOptimizer, graph: *model.ComputationGraph) !void {
        _ = self;
        _ = graph;
        
        // TODO: Implement memory layout optimization
        // - Reorder operations to minimize memory usage
        // - Insert memory copy operations where beneficial
        
        std.log.info("Memory layout optimization: analyzing tensor layouts", .{});
    }
};

pub const GraphExecutor = struct {
    allocator: Allocator,
    execution_plan: ?ExecutionPlan,
    tensor_registry: std.StringHashMap(tensor.Tensor),
    
    pub fn init(allocator: Allocator) GraphExecutor {
        return GraphExecutor{
            .allocator = allocator,
            .execution_plan = null,
            .tensor_registry = std.StringHashMap(tensor.Tensor).init(allocator),
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
    
    pub fn prepare(self: *GraphExecutor, graph: *const model.ComputationGraph) !void {
        // Create execution plan
        self.execution_plan = try self.createExecutionPlan(graph);
        
        // Pre-allocate tensors based on memory plan
        try self.preallocateTensors();
        
        std.log.info("Graph executor prepared with {} steps", .{self.execution_plan.?.execution_order.len});
    }
    
    pub fn execute(self: *GraphExecutor, graph: *const model.ComputationGraph, inputs: []tensor.Tensor) ![]tensor.Tensor {
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
    
    fn createExecutionPlan(self: *GraphExecutor, graph: *const model.ComputationGraph) !ExecutionPlan {
        var plan = ExecutionPlan.init(self.allocator);
        
        // Topological sort
        var execution_order = std.ArrayList(usize).init(self.allocator);
        defer execution_order.deinit();
        
        // Simple topological sort (Kahn's algorithm)
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
    
    fn setInputs(self: *GraphExecutor, graph: *const model.ComputationGraph, inputs: []tensor.Tensor) !void {
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
    
    fn executeNode(self: *GraphExecutor, graph: *const model.ComputationGraph, node_idx: usize) !void {
        _ = self;
        _ = graph;
        _ = node_idx;
        
        // TODO: Implement actual node execution
        // This would dispatch to the appropriate operator implementation
        std.log.debug("Executing node {d}", .{node_idx});
    }
    
    fn collectOutputs(self: *GraphExecutor, graph: *const model.ComputationGraph) ![]tensor.Tensor {
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

// Utility functions for graph analysis
pub fn analyzeGraph(allocator: Allocator, graph: *const model.ComputationGraph) !GraphAnalysis {
    var analysis = GraphAnalysis.init(allocator);
    
    // Count operator types
    for (graph.nodes.items) |node| {
        const count = analysis.op_counts.get(node.op_type) orelse 0;
        try analysis.op_counts.put(try allocator.dupe(u8, node.op_type), count + 1);
    }
    
    // Estimate memory usage
    analysis.estimated_memory_mb = estimateMemoryUsage(graph);
    
    // Count parameters
    analysis.parameter_count = countParameters(graph);
    
    return analysis;
}

pub const GraphAnalysis = struct {
    op_counts: std.StringHashMap(u32),
    estimated_memory_mb: f64,
    parameter_count: u64,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) GraphAnalysis {
        return GraphAnalysis{
            .op_counts = std.StringHashMap(u32).init(allocator),
            .estimated_memory_mb = 0.0,
            .parameter_count = 0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *GraphAnalysis) void {
        var iter = self.op_counts.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.op_counts.deinit();
    }
    
    pub fn print(self: *const GraphAnalysis) void {
        std.log.info("=== Graph Analysis ===", .{});
        std.log.info("Estimated memory: {d:.1} MB", .{self.estimated_memory_mb});
        std.log.info("Parameter count: {d}", .{self.parameter_count});
        std.log.info("Operator counts:", .{});
        
        var iter = self.op_counts.iterator();
        while (iter.next()) |entry| {
            std.log.info("  {s}: {d}", .{ entry.key_ptr.*, entry.value_ptr.* });
        }
    }
};

fn estimateMemoryUsage(graph: *const model.ComputationGraph) f64 {
    _ = graph;
    // TODO: Implement memory estimation based on tensor shapes
    return 100.0; // Placeholder
}

fn countParameters(graph: *const model.ComputationGraph) u64 {
    _ = graph;
    // TODO: Count parameters in weight tensors
    return 1000000; // Placeholder
}

const std = @import("std");

/// Computation graph node
pub const GraphNode = struct {
    id: u32,
    op_type: OpType,
    inputs: []u32,
    outputs: []u32,
    
    pub const OpType = enum {
        input,
        output,
        matmul,
        add,
        relu,
        softmax,
        embedding,
        layer_norm,
        attention,
    };
};

/// Computation graph executor (placeholder)
pub const GraphExecutor = struct {
    nodes: []GraphNode,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) GraphExecutor {
        return GraphExecutor{
            .nodes = &[_]GraphNode{},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *GraphExecutor) void {
        self.allocator.free(self.nodes);
    }
};

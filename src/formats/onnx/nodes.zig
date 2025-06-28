const std = @import("std");
const Allocator = std.mem.Allocator;
const model = @import("../model.zig");
const tensor = @import("../../core/tensor.zig");

pub const NodeError = error{
    UnsupportedOperation,
    InvalidAttribute,
    ShapeMismatch,
    MissingInput,
    OutOfMemory,
};

// ONNX Node type definitions and their implementations
pub const NodeType = enum {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Gemm,

    // Activation functions
    Relu,
    Sigmoid,
    Tanh,
    Softmax,

    // Convolution and pooling
    Conv,
    MaxPool,
    AveragePool,

    // Normalization
    BatchNormalization,
    LayerNormalization,

    // Shape operations
    Reshape,
    Transpose,
    Squeeze,
    Unsqueeze,

    // Utility operations
    Concat,
    Split,
    Constant,
    Identity,

    pub fn fromString(op_type: []const u8) ?NodeType {
        const type_map = std.ComptimeStringMap(NodeType, .{
            .{ "Add", .Add },
            .{ "Sub", .Sub },
            .{ "Mul", .Mul },
            .{ "Div", .Div },
            .{ "MatMul", .MatMul },
            .{ "Gemm", .Gemm },
            .{ "Relu", .Relu },
            .{ "Sigmoid", .Sigmoid },
            .{ "Tanh", .Tanh },
            .{ "Softmax", .Softmax },
            .{ "Conv", .Conv },
            .{ "MaxPool", .MaxPool },
            .{ "AveragePool", .AveragePool },
            .{ "BatchNormalization", .BatchNormalization },
            .{ "LayerNormalization", .LayerNormalization },
            .{ "Reshape", .Reshape },
            .{ "Transpose", .Transpose },
            .{ "Squeeze", .Squeeze },
            .{ "Unsqueeze", .Unsqueeze },
            .{ "Concat", .Concat },
            .{ "Split", .Split },
            .{ "Constant", .Constant },
            .{ "Identity", .Identity },
        });

        return type_map.get(op_type);
    }

    pub fn toString(self: NodeType) []const u8 {
        return switch (self) {
            .Add => "Add",
            .Sub => "Sub",
            .Mul => "Mul",
            .Div => "Div",
            .MatMul => "MatMul",
            .Gemm => "Gemm",
            .Relu => "Relu",
            .Sigmoid => "Sigmoid",
            .Tanh => "Tanh",
            .Softmax => "Softmax",
            .Conv => "Conv",
            .MaxPool => "MaxPool",
            .AveragePool => "AveragePool",
            .BatchNormalization => "BatchNormalization",
            .LayerNormalization => "LayerNormalization",
            .Reshape => "Reshape",
            .Transpose => "Transpose",
            .Squeeze => "Squeeze",
            .Unsqueeze => "Unsqueeze",
            .Concat => "Concat",
            .Split => "Split",
            .Constant => "Constant",
            .Identity => "Identity",
        };
    }
};

// Node execution interface
pub const NodeExecutor = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) NodeExecutor {
        return NodeExecutor{
            .allocator = allocator,
        };
    }

    pub fn execute(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        const node_type = NodeType.fromString(node.op_type) orelse return NodeError.UnsupportedOperation;

        switch (node_type) {
            .Add => try self.executeAdd(node, inputs, outputs),
            .Sub => try self.executeSub(node, inputs, outputs),
            .Mul => try self.executeMul(node, inputs, outputs),
            .Div => try self.executeDiv(node, inputs, outputs),
            .MatMul => try self.executeMatMul(node, inputs, outputs),
            .Relu => try self.executeRelu(node, inputs, outputs),
            .Sigmoid => try self.executeSigmoid(node, inputs, outputs),
            .Tanh => try self.executeTanh(node, inputs, outputs),
            .Softmax => try self.executeSoftmax(node, inputs, outputs),
            .Reshape => try self.executeReshape(node, inputs, outputs),
            .Transpose => try self.executeTranspose(node, inputs, outputs),
            .Identity => try self.executeIdentity(node, inputs, outputs),
            .Constant => try self.executeConstant(node, inputs, outputs),
            else => {
                std.log.warn("Node type {} not yet implemented", .{node_type});
                return NodeError.UnsupportedOperation;
            },
        }
    }

    // Arithmetic operations
    fn executeAdd(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 2 or outputs.len != 1) return NodeError.MissingInput;

        const a = &inputs[0];
        const b = &inputs[1];
        var result = &outputs[0];

        // Simple element-wise addition (assuming same shape)
        if (!std.mem.eql(usize, a.shape, b.shape)) return NodeError.ShapeMismatch;

        const numel = a.numel();
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const val_a = try a.get_f32_flat(i);
            const val_b = try b.get_f32_flat(i);
            try result.set_f32_flat(i, val_a + val_b);
        }
    }

    fn executeSub(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 2 or outputs.len != 1) return NodeError.MissingInput;

        const a = &inputs[0];
        const b = &inputs[1];
        var result = &outputs[0];

        if (!std.mem.eql(usize, a.shape, b.shape)) return NodeError.ShapeMismatch;

        const numel = a.numel();
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const val_a = try a.get_f32_flat(i);
            const val_b = try b.get_f32_flat(i);
            try result.set_f32_flat(i, val_a - val_b);
        }
    }

    fn executeMul(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 2 or outputs.len != 1) return NodeError.MissingInput;

        const a = &inputs[0];
        const b = &inputs[1];
        var result = &outputs[0];

        if (!std.mem.eql(usize, a.shape, b.shape)) return NodeError.ShapeMismatch;

        const numel = a.numel();
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const val_a = try a.get_f32_flat(i);
            const val_b = try b.get_f32_flat(i);
            try result.set_f32_flat(i, val_a * val_b);
        }
    }

    fn executeDiv(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 2 or outputs.len != 1) return NodeError.MissingInput;

        const a = &inputs[0];
        const b = &inputs[1];
        var result = &outputs[0];

        if (!std.mem.eql(usize, a.shape, b.shape)) return NodeError.ShapeMismatch;

        const numel = a.numel();
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const val_a = try a.get_f32_flat(i);
            const val_b = try b.get_f32_flat(i);
            if (val_b == 0.0) return NodeError.InvalidAttribute;
            try result.set_f32_flat(i, val_a / val_b);
        }
    }

    fn executeMatMul(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 2 or outputs.len != 1) return NodeError.MissingInput;

        const a = &inputs[0];
        const b = &inputs[1];
        var result = &outputs[0];

        // Simple 2D matrix multiplication
        if (a.shape.len != 2 or b.shape.len != 2) return NodeError.ShapeMismatch;
        if (a.shape[1] != b.shape[0]) return NodeError.ShapeMismatch;

        const m = a.shape[0];
        const n = b.shape[1];
        const k = a.shape[1];

        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var sum: f32 = 0.0;
                var l: usize = 0;
                while (l < k) : (l += 1) {
                    const a_val = try a.get_f32(&[_]usize{ i, l });
                    const b_val = try b.get_f32(&[_]usize{ l, j });
                    sum += a_val * b_val;
                }
                try result.set_f32(&[_]usize{ i, j }, sum);
            }
        }
    }

    // Activation functions
    fn executeRelu(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 1 or outputs.len != 1) return NodeError.MissingInput;

        const input = &inputs[0];
        var result = &outputs[0];

        const numel = input.numel();
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const val = try input.get_f32_flat(i);
            try result.set_f32_flat(i, @max(0.0, val));
        }
    }

    fn executeSigmoid(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 1 or outputs.len != 1) return NodeError.MissingInput;

        const input = &inputs[0];
        var result = &outputs[0];

        const numel = input.numel();
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const val = try input.get_f32_flat(i);
            const sigmoid_val = 1.0 / (1.0 + @exp(-val));
            try result.set_f32_flat(i, sigmoid_val);
        }
    }

    fn executeTanh(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 1 or outputs.len != 1) return NodeError.MissingInput;

        const input = &inputs[0];
        var result = &outputs[0];

        const numel = input.numel();
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const val = try input.get_f32_flat(i);
            try result.set_f32_flat(i, std.math.tanh(val));
        }
    }

    fn executeSoftmax(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 1 or outputs.len != 1) return NodeError.MissingInput;

        const input = &inputs[0];
        var result = &outputs[0];

        // Simple softmax implementation (assumes 1D or applies to last dimension)
        const numel = input.numel();

        // Find max for numerical stability
        var max_val: f32 = -std.math.inf(f32);
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const val = try input.get_f32_flat(i);
            max_val = @max(max_val, val);
        }

        // Compute exp and sum
        var sum: f32 = 0.0;
        i = 0;
        while (i < numel) : (i += 1) {
            const val = try input.get_f32_flat(i);
            const exp_val = @exp(val - max_val);
            try result.set_f32_flat(i, exp_val);
            sum += exp_val;
        }

        // Normalize
        i = 0;
        while (i < numel) : (i += 1) {
            const exp_val = try result.get_f32_flat(i);
            try result.set_f32_flat(i, exp_val / sum);
        }
    }

    // Shape operations
    fn executeReshape(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;
        _ = inputs;
        _ = outputs;

        // TODO: Implement reshape operation
        return NodeError.UnsupportedOperation;
    }

    fn executeTranspose(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;
        _ = inputs;
        _ = outputs;

        // TODO: Implement transpose operation
        return NodeError.UnsupportedOperation;
    }

    // Utility operations
    fn executeIdentity(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = node;

        if (inputs.len != 1 or outputs.len != 1) return NodeError.MissingInput;

        const input = &inputs[0];
        var result = &outputs[0];

        // Copy input to output
        const numel = input.numel();
        var i: usize = 0;
        while (i < numel) : (i += 1) {
            const val = try input.get_f32_flat(i);
            try result.set_f32_flat(i, val);
        }
    }

    fn executeConstant(self: *NodeExecutor, node: *const model.GraphNode, inputs: []tensor.Tensor, outputs: []tensor.Tensor) !void {
        _ = self;
        _ = inputs;

        if (outputs.len != 1) return NodeError.MissingInput;

        // Get constant value from node attributes
        if (node.attributes.get("value")) |attr_value| {
            switch (attr_value) {
                .tensor => |constant_tensor| {
                    var result = &outputs[0];
                    const numel = constant_tensor.numel();
                    var i: usize = 0;
                    while (i < numel) : (i += 1) {
                        const val = try constant_tensor.get_f32_flat(i);
                        try result.set_f32_flat(i, val);
                    }
                },
                .float => |float_val| {
                    var result = &outputs[0];
                    const numel = result.numel();
                    var i: usize = 0;
                    while (i < numel) : (i += 1) {
                        try result.set_f32_flat(i, @floatCast(float_val));
                    }
                },
                else => return NodeError.InvalidAttribute,
            }
        } else {
            return NodeError.InvalidAttribute;
        }
    }
};

const std = @import("std");
const mod = @import("mod.zig");
const activations = @import("../math/activations.zig");

const Operator = mod.Operator;
const Tensor = mod.Tensor;
const OpContext = mod.OpContext;

/// Activation operator types
pub const ActivationType = enum {
    relu,
    gelu,
    silu,
    sigmoid,
    tanh,
    softmax,
    leaky_relu,
    elu,
};

/// Generic activation operator
pub const ActivationOp = struct {
    activation_type: ActivationType,
    alpha: f32, // For leaky_relu and elu
    
    pub fn init(activation_type: ActivationType, alpha: f32) ActivationOp {
        return ActivationOp{
            .activation_type = activation_type,
            .alpha = alpha,
        };
    }
    
    pub fn deinit(self: *ActivationOp) void {
        _ = self;
        // No cleanup needed
    }
    
    pub fn forward(
        self: *ActivationOp,
        inputs: []const Tensor,
        outputs: []Tensor,
        context: *OpContext,
    ) !void {
        _ = context;
        std.debug.assert(inputs.len == 1);
        std.debug.assert(outputs.len == 1);
        
        const input = inputs[0];
        var output = &outputs[0];
        
        // Check that input and output have the same shape
        std.debug.assert(input.shape.len == output.shape.len);
        for (input.shape, output.shape) |in_dim, out_dim| {
            std.debug.assert(in_dim == out_dim);
        }
        
        switch (self.activation_type) {
            .relu => activations.relu(input.data, output.data),
            .gelu => activations.gelu(input.data, output.data),
            .silu => activations.silu(input.data, output.data),
            .sigmoid => activations.sigmoid(input.data, output.data),
            .tanh => activations.tanh_activation(input.data, output.data),
            .softmax => {
                if (input.shape.len == 1) {
                    // Single vector softmax
                    activations.softmax(input.data, output.data);
                } else {
                    // Apply softmax to each row (last dimension)
                    try self.applySoftmaxBatched(input, output);
                }
            },
            .leaky_relu => activations.leakyRelu(input.data, output.data, self.alpha),
            .elu => activations.elu(input.data, output.data, self.alpha),
        }
    }
    
    fn applySoftmaxBatched(self: *ActivationOp, input: Tensor, output: *Tensor) !void {
        _ = self;
        
        if (input.shape.len == 2) {
            const batch_size = input.shape[0];
            const feature_size = input.shape[1];
            
            for (0..batch_size) |b| {
                // Extract row data
                var input_row = std.ArrayList(f32).init(std.heap.page_allocator);
                defer input_row.deinit();
                var output_row = std.ArrayList(f32).init(std.heap.page_allocator);
                defer output_row.deinit();
                
                try input_row.resize(feature_size);
                try output_row.resize(feature_size);
                
                for (0..feature_size) |f| {
                    input_row.items[f] = input.get(&[_]usize{ b, f });
                }
                
                // Apply softmax to this row
                activations.softmax(input_row.items, output_row.items);
                
                // Copy back to output tensor
                for (0..feature_size) |f| {
                    output.set(&[_]usize{ b, f }, output_row.items[f]);
                }
            }
        } else {
            // For higher dimensions, apply softmax along the last dimension
            return error.UnsupportedSoftmaxShape;
        }
    }
    
    pub fn getOutputShape(self: *ActivationOp, input_shapes: []const []const usize) []const usize {
        _ = self;
        std.debug.assert(input_shapes.len == 1);
        // Activation functions preserve input shape
        return input_shapes[0];
    }
};

// VTable implementations
fn activationDeinit(impl: *anyopaque, allocator: std.mem.Allocator) void {
    _ = allocator;
    const activation_op: *ActivationOp = @ptrCast(@alignCast(impl));
    activation_op.deinit();
}

fn activationForward(
    impl: *anyopaque,
    inputs: []const Tensor,
    outputs: []Tensor,
    context: *OpContext,
) anyerror!void {
    const activation_op: *ActivationOp = @ptrCast(@alignCast(impl));
    return activation_op.forward(inputs, outputs, context);
}

fn activationGetOutputShape(impl: *anyopaque, input_shapes: []const []const usize) []const usize {
    const activation_op: *ActivationOp = @ptrCast(@alignCast(impl));
    return activation_op.getOutputShape(input_shapes);
}

/// Create ReLU operator
pub fn createReLUOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    _ = params;
    
    var activation_op = try allocator.create(ActivationOp);
    activation_op.* = ActivationOp.init(.relu, 0.0);
    
    const vtable = &Operator.VTable{
        .deinit = activationDeinit,
        .forward = activationForward,
        .getOutputShape = activationGetOutputShape,
    };
    
    return Operator.init("ReLU", .activation, vtable, activation_op);
}

/// Create GELU operator
pub fn createGELUOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    _ = params;
    
    var activation_op = try allocator.create(ActivationOp);
    activation_op.* = ActivationOp.init(.gelu, 0.0);
    
    const vtable = &Operator.VTable{
        .deinit = activationDeinit,
        .forward = activationForward,
        .getOutputShape = activationGetOutputShape,
    };
    
    return Operator.init("GELU", .activation, vtable, activation_op);
}

/// Create Softmax operator
pub fn createSoftmaxOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    _ = params;
    
    var activation_op = try allocator.create(ActivationOp);
    activation_op.* = ActivationOp.init(.softmax, 0.0);
    
    const vtable = &Operator.VTable{
        .deinit = activationDeinit,
        .forward = activationForward,
        .getOutputShape = activationGetOutputShape,
    };
    
    return Operator.init("Softmax", .activation, vtable, activation_op);
}

/// Create Leaky ReLU operator
pub fn createLeakyReLUOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    const alpha = @as(f32, @floatCast(params.object.get("alpha").?.float));
    
    var activation_op = try allocator.create(ActivationOp);
    activation_op.* = ActivationOp.init(.leaky_relu, alpha);
    
    const vtable = &Operator.VTable{
        .deinit = activationDeinit,
        .forward = activationForward,
        .getOutputShape = activationGetOutputShape,
    };
    
    return Operator.init("LeakyReLU", .activation, vtable, activation_op);
}

/// Create SiLU operator
pub fn createSiLUOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    _ = params;
    
    var activation_op = try allocator.create(ActivationOp);
    activation_op.* = ActivationOp.init(.silu, 0.0);
    
    const vtable = &Operator.VTable{
        .deinit = activationDeinit,
        .forward = activationForward,
        .getOutputShape = activationGetOutputShape,
    };
    
    return Operator.init("SiLU", .activation, vtable, activation_op);
}

test "relu operator" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var relu_op = ActivationOp.init(.relu, 0.0);
    defer relu_op.deinit();
    
    // Create input tensor with negative and positive values
    const shape = [_]usize{4};
    var input = try Tensor.init(allocator, &shape);
    defer input.deinit();
    var output = try Tensor.init(allocator, &shape);
    defer output.deinit();
    
    input.set(&[_]usize{0}, -2.0);
    input.set(&[_]usize{1}, -1.0);
    input.set(&[_]usize{2}, 1.0);
    input.set(&[_]usize{3}, 2.0);
    
    var context = OpContext.init(allocator, .cpu);
    try relu_op.forward(&[_]Tensor{input}, &[_]Tensor{output}, &context);
    
    try testing.expect(output.get(&[_]usize{0}) == 0.0);
    try testing.expect(output.get(&[_]usize{1}) == 0.0);
    try testing.expect(output.get(&[_]usize{2}) == 1.0);
    try testing.expect(output.get(&[_]usize{3}) == 2.0);
}

test "softmax operator" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var softmax_op = ActivationOp.init(.softmax, 0.0);
    defer softmax_op.deinit();
    
    // Create input tensor
    const shape = [_]usize{3};
    var input = try Tensor.init(allocator, &shape);
    defer input.deinit();
    var output = try Tensor.init(allocator, &shape);
    defer output.deinit();
    
    input.set(&[_]usize{0}, 1.0);
    input.set(&[_]usize{1}, 2.0);
    input.set(&[_]usize{2}, 3.0);
    
    var context = OpContext.init(allocator, .cpu);
    try softmax_op.forward(&[_]Tensor{input}, &[_]Tensor{output}, &context);
    
    // Check that probabilities sum to 1
    var sum: f32 = 0.0;
    for (0..3) |i| {
        sum += output.get(&[_]usize{i});
    }
    try testing.expectApproxEqRel(sum, 1.0, 1e-6);
    
    // Check that output is in ascending order (since input is ascending)
    try testing.expect(output.get(&[_]usize{0}) < output.get(&[_]usize{1}));
    try testing.expect(output.get(&[_]usize{1}) < output.get(&[_]usize{2}));
}

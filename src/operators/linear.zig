const std = @import("std");
const mod = @import("mod.zig");
const math = @import("../math/mod.zig");

const Operator = mod.Operator;
const Tensor = mod.Tensor;
const OpContext = mod.OpContext;

/// Linear (fully connected) operator
pub const LinearOp = struct {
    input_size: usize,
    output_size: usize,
    use_bias: bool,

    // Weights stored as tensors
    weight: Tensor,
    bias: ?Tensor,

    pub fn init(
        allocator: std.mem.Allocator,
        input_size: usize,
        output_size: usize,
        use_bias: bool,
    ) !LinearOp {
        const weight_shape = [_]usize{ input_size, output_size };
        var weight = try Tensor.init(allocator, &weight_shape);

        // Xavier initialization
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = rng.random();
        const xavier_std = @sqrt(2.0 / @as(f32, @floatFromInt(input_size + output_size)));

        for (weight.data) |*val| {
            val.* = random.floatNorm(f32) * xavier_std;
        }

        var bias: ?Tensor = null;
        if (use_bias) {
            const bias_shape = [_]usize{output_size};
            bias = try Tensor.init(allocator, &bias_shape);
            @memset(bias.?.data, 0.0);
        }

        return LinearOp{
            .input_size = input_size,
            .output_size = output_size,
            .use_bias = use_bias,
            .weight = weight,
            .bias = bias,
        };
    }

    pub fn deinit(self: *LinearOp) void {
        self.weight.deinit();
        if (self.bias) |*bias| {
            bias.deinit();
        }
    }

    pub fn forward(
        self: *LinearOp,
        inputs: []const Tensor,
        outputs: []Tensor,
        context: *OpContext,
    ) !void {
        _ = context;
        std.debug.assert(inputs.len == 1);
        std.debug.assert(outputs.len == 1);

        const input = inputs[0];
        var output = &outputs[0];

        // Input should be [batch_size, input_size] or [input_size]
        const input_features = if (input.shape.len == 2) input.shape[1] else input.shape[0];

        std.debug.assert(input_features == self.input_size);
        std.debug.assert(output.shape.len == 2 or output.shape.len == 1);

        if (input.shape.len == 2) {
            // Batch processing: output = input @ weight + bias
            try self.forwardBatch(input, output);
        } else {
            // Single vector: output = input @ weight + bias
            try self.forwardVector(input, output);
        }
    }

    fn forwardBatch(self: *LinearOp, input: Tensor, output: *Tensor) !void {
        const batch_size = input.shape[0];

        // Matrix multiplication: output = input @ weight
        for (0..batch_size) |b| {
            for (0..self.output_size) |o| {
                var sum: f32 = 0.0;
                for (0..self.input_size) |i| {
                    const input_val = input.get(&[_]usize{ b, i });
                    const weight_val = self.weight.get(&[_]usize{ i, o });
                    sum += input_val * weight_val;
                }

                // Add bias if present
                if (self.bias) |bias| {
                    sum += bias.get(&[_]usize{o});
                }

                output.set(&[_]usize{ b, o }, sum);
            }
        }
    }

    fn forwardVector(self: *LinearOp, input: Tensor, output: *Tensor) !void {
        // Vector-matrix multiplication: output = input @ weight
        for (0..self.output_size) |o| {
            var sum: f32 = 0.0;
            for (0..self.input_size) |i| {
                const input_val = input.get(&[_]usize{i});
                const weight_val = self.weight.get(&[_]usize{ i, o });
                sum += input_val * weight_val;
            }

            // Add bias if present
            if (self.bias) |bias| {
                sum += bias.get(&[_]usize{o});
            }

            output.set(&[_]usize{o}, sum);
        }
    }

    pub fn getOutputShape(self: *LinearOp, input_shapes: []const []const usize) []const usize {
        std.debug.assert(input_shapes.len == 1);
        const input_shape = input_shapes[0];

        if (input_shape.len == 2) {
            // Batch processing: [batch_size, input_size] -> [batch_size, output_size]
            return &[_]usize{ input_shape[0], self.output_size };
        } else {
            // Single vector: [input_size] -> [output_size]
            return &[_]usize{self.output_size};
        }
    }

    pub fn loadWeights(self: *LinearOp, weight_data: []const f32, bias_data: ?[]const f32) !void {
        if (weight_data.len != self.weight.data.len) {
            return error.WeightSizeMismatch;
        }

        @memcpy(self.weight.data, weight_data);

        if (bias_data) |bias| {
            if (self.bias == null) {
                return error.BiasNotExpected;
            }
            if (bias.len != self.bias.?.data.len) {
                return error.BiasSizeMismatch;
            }
            @memcpy(self.bias.?.data, bias);
        }
    }
};

// VTable implementations
fn linearDeinit(impl: *anyopaque, allocator: std.mem.Allocator) void {
    _ = allocator;
    const linear_op: *LinearOp = @ptrCast(@alignCast(impl));
    linear_op.deinit();
}

fn linearForward(
    impl: *anyopaque,
    inputs: []const Tensor,
    outputs: []Tensor,
    context: *OpContext,
) anyerror!void {
    const linear_op: *LinearOp = @ptrCast(@alignCast(impl));
    return linear_op.forward(inputs, outputs, context);
}

fn linearGetOutputShape(impl: *anyopaque, input_shapes: []const []const usize) []const usize {
    const linear_op: *LinearOp = @ptrCast(@alignCast(impl));
    return linear_op.getOutputShape(input_shapes);
}

/// Create linear operator from JSON parameters
pub fn createLinearOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    const input_size = @as(usize, @intCast(params.object.get("input_size").?.integer));
    const output_size = @as(usize, @intCast(params.object.get("output_size").?.integer));
    const use_bias = params.object.get("use_bias").?.bool;

    var linear_op = try allocator.create(LinearOp);
    linear_op.* = try LinearOp.init(allocator, input_size, output_size, use_bias);

    const vtable = &Operator.VTable{
        .deinit = linearDeinit,
        .forward = linearForward,
        .getOutputShape = linearGetOutputShape,
    };

    return Operator.init("Linear", .linear, vtable, linear_op);
}

test "linear operator creation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var linear_op = try LinearOp.init(allocator, 3, 2, true);
    defer linear_op.deinit();

    try testing.expect(linear_op.input_size == 3);
    try testing.expect(linear_op.output_size == 2);
    try testing.expect(linear_op.use_bias == true);
}

test "linear operator forward" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var linear_op = try LinearOp.init(allocator, 2, 3, false);
    defer linear_op.deinit();

    // Set known weights
    linear_op.weight.set(&[_]usize{ 0, 0 }, 1.0);
    linear_op.weight.set(&[_]usize{ 0, 1 }, 2.0);
    linear_op.weight.set(&[_]usize{ 0, 2 }, 3.0);
    linear_op.weight.set(&[_]usize{ 1, 0 }, 4.0);
    linear_op.weight.set(&[_]usize{ 1, 1 }, 5.0);
    linear_op.weight.set(&[_]usize{ 1, 2 }, 6.0);

    // Create input tensor
    const input_shape = [_]usize{2};
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    input.set(&[_]usize{0}, 1.0);
    input.set(&[_]usize{1}, 2.0);

    // Create output tensor
    const output_shape = [_]usize{3};
    var output = try Tensor.init(allocator, &output_shape);
    defer output.deinit();

    // Forward pass
    var context = OpContext.init(allocator, .cpu);
    try linear_op.forward(&[_]Tensor{input}, &[_]Tensor{output}, &context);

    // Check results: [1, 2] @ [[1, 2, 3], [4, 5, 6]] = [9, 12, 15]
    try testing.expectApproxEqRel(output.get(&[_]usize{0}), 9.0, 1e-6);
    try testing.expectApproxEqRel(output.get(&[_]usize{1}), 12.0, 1e-6);
    try testing.expectApproxEqRel(output.get(&[_]usize{2}), 15.0, 1e-6);
}

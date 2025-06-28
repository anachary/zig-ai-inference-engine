const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../../core/tensor.zig");
const simd = @import("../../core/simd.zig");
const operators = @import("../operators.zig");

/// Sigmoid activation operator
pub const Sigmoid = struct {
    pub const op = operators.Operator{
        .name = "Sigmoid",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0];
        var output = outputs[0];

        if (input.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, input.shape, output.shape)) {
            return operators.OperatorError.ShapeMismatch;
        }

        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)))[0..input.numel()];
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)))[0..output.numel()];

        for (input_data, output_data) |x, *y| {
            y.* = 1.0 / (1.0 + @exp(-x));
        }
    }
};

/// Tanh activation operator
pub const Tanh = struct {
    pub const op = operators.Operator{
        .name = "Tanh",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0];
        var output = outputs[0];

        if (input.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, input.shape, output.shape)) {
            return operators.OperatorError.ShapeMismatch;
        }

        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)))[0..input.numel()];
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)))[0..output.numel()];

        for (input_data, output_data) |x, *y| {
            y.* = std.math.tanh(x);
        }
    }
};

/// GELU activation operator (Gaussian Error Linear Unit)
pub const GELU = struct {
    pub const op = operators.Operator{
        .name = "GELU",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0];
        var output = outputs[0];

        if (input.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, input.shape, output.shape)) {
            return operators.OperatorError.ShapeMismatch;
        }

        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)))[0..input.numel()];
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)))[0..output.numel()];

        for (input_data, output_data) |x, *y| {
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            const sqrt_2_over_pi = 0.7978845608028654;
            const coeff = 0.044715;
            const inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            y.* = 0.5 * x * (1.0 + std.math.tanh(inner));
        }
    }
};

/// Swish/SiLU activation operator
pub const Swish = struct {
    pub const op = operators.Operator{
        .name = "Swish",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0];
        var output = outputs[0];

        if (input.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, input.shape, output.shape)) {
            return operators.OperatorError.ShapeMismatch;
        }

        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)))[0..input.numel()];
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)))[0..output.numel()];

        for (input_data, output_data) |x, *y| {
            // Swish(x) = x * sigmoid(x)
            const sigmoid_x = 1.0 / (1.0 + @exp(-x));
            y.* = x * sigmoid_x;
        }
    }
};

/// LeakyReLU activation operator
pub const LeakyReLU = struct {
    pub const op = operators.Operator{
        .name = "LeakyReLU",
        .forward_fn = forward,
    };

    pub const Params = struct {
        negative_slope: f32 = 0.01,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0];
        var output = outputs[0];

        if (input.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, input.shape, output.shape)) {
            return operators.OperatorError.ShapeMismatch;
        }

        const params = Params{};
        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)))[0..input.numel()];
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)))[0..output.numel()];

        for (input_data, output_data) |x, *y| {
            y.* = if (x > 0) x else params.negative_slope * x;
        }
    }
};

/// ELU activation operator (Exponential Linear Unit)
pub const ELU = struct {
    pub const op = operators.Operator{
        .name = "ELU",
        .forward_fn = forward,
    };

    pub const Params = struct {
        alpha: f32 = 1.0,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0];
        var output = outputs[0];

        if (input.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, input.shape, output.shape)) {
            return operators.OperatorError.ShapeMismatch;
        }

        const params = Params{};
        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)))[0..input.numel()];
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)))[0..output.numel()];

        for (input_data, output_data) |x, *y| {
            y.* = if (x > 0) x else params.alpha * (@exp(x) - 1.0);
        }
    }
};

// Test functions
test "sigmoid activation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const shape = [_]usize{4};
    var input = try tensor.Tensor.init(allocator, &shape, .f32);
    defer input.deinit();

    var output = try tensor.Tensor.init(allocator, &shape, .f32);
    defer output.deinit();

    // Test values: [-2, -1, 0, 1]
    try input.set_f32(&[_]usize{0}, -2.0);
    try input.set_f32(&[_]usize{1}, -1.0);
    try input.set_f32(&[_]usize{2}, 0.0);
    try input.set_f32(&[_]usize{3}, 1.0);

    const inputs = [_]tensor.Tensor{input};
    var outputs = [_]tensor.Tensor{output};

    try Sigmoid.op.forward(&inputs, &outputs, allocator);

    // Check sigmoid(0) = 0.5
    const result = try output.get_f32(&[_]usize{2});
    try testing.expect(std.math.fabs(result - 0.5) < 1e-6);
}

test "gelu activation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const shape = [_]usize{3};
    var input = try tensor.Tensor.init(allocator, &shape, .f32);
    defer input.deinit();

    var output = try tensor.Tensor.init(allocator, &shape, .f32);
    defer output.deinit();

    // Test values: [-1, 0, 1]
    try input.set_f32(&[_]usize{0}, -1.0);
    try input.set_f32(&[_]usize{1}, 0.0);
    try input.set_f32(&[_]usize{2}, 1.0);

    const inputs = [_]tensor.Tensor{input};
    var outputs = [_]tensor.Tensor{output};

    try GELU.op.forward(&inputs, &outputs, allocator);

    // Check GELU(0) ≈ 0
    const result = try output.get_f32(&[_]usize{1});
    try testing.expect(std.math.fabs(result) < 1e-6);
}

const std = @import("std");
const Allocator = std.mem.Allocator;
pub const tensor = @import("../core/tensor.zig");
const simd = @import("../core/simd.zig");

pub const OperatorError = error{
    InvalidInput,
    ShapeMismatch,
    UnsupportedDataType,
    OutOfMemory,
    InvalidOperation,
};

/// Base operator interface
pub const Operator = struct {
    name: []const u8,
    forward_fn: *const fn (inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) OperatorError!void,

    const Self = @This();

    pub fn forward(self: *const Self, inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) OperatorError!void {
        return self.forward_fn(inputs, outputs, allocator);
    }
};

/// Element-wise addition operator
pub const Add = struct {
    pub const op = Operator{
        .name = "Add",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) OperatorError!void {
        _ = allocator;

        if (inputs.len != 2 or outputs.len != 1) {
            return OperatorError.InvalidInput;
        }

        const a = inputs[0];
        const b = inputs[1];
        var result = outputs[0];

        if (a.dtype != b.dtype or a.dtype != result.dtype) {
            return OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, a.shape, b.shape) or !std.mem.eql(usize, a.shape, result.shape)) {
            return OperatorError.ShapeMismatch;
        }

        switch (a.dtype) {
            .f32 => {
                const a_data = @as([*]const f32, @ptrCast(@alignCast(a.data.ptr)))[0..a.numel()];
                const b_data = @as([*]const f32, @ptrCast(@alignCast(b.data.ptr)))[0..b.numel()];
                const result_data = @as([*]f32, @ptrCast(@alignCast(result.data.ptr)))[0..result.numel()];

                simd.vector_add_f32(a_data, b_data, result_data) catch {
                    return OperatorError.InvalidOperation;
                };
            },
            else => return OperatorError.UnsupportedDataType,
        }
    }
};

/// Element-wise subtraction operator
pub const Sub = struct {
    pub const op = Operator{
        .name = "Sub",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) OperatorError!void {
        _ = allocator;

        if (inputs.len != 2 or outputs.len != 1) {
            return OperatorError.InvalidInput;
        }

        const a = inputs[0];
        const b = inputs[1];
        var result = outputs[0];

        if (a.dtype != b.dtype or a.dtype != result.dtype) {
            return OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, a.shape, b.shape) or !std.mem.eql(usize, a.shape, result.shape)) {
            return OperatorError.ShapeMismatch;
        }

        switch (a.dtype) {
            .f32 => {
                const a_data = @as([*]const f32, @ptrCast(@alignCast(a.data.ptr)))[0..a.numel()];
                const b_data = @as([*]const f32, @ptrCast(@alignCast(b.data.ptr)))[0..b.numel()];
                const result_data = @as([*]f32, @ptrCast(@alignCast(result.data.ptr)))[0..result.numel()];

                simd.vector_sub_f32(a_data, b_data, result_data) catch {
                    return OperatorError.InvalidOperation;
                };
            },
            else => return OperatorError.UnsupportedDataType,
        }
    }
};

/// Element-wise multiplication operator
pub const Mul = struct {
    pub const op = Operator{
        .name = "Mul",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) OperatorError!void {
        _ = allocator;

        if (inputs.len != 2 or outputs.len != 1) {
            return OperatorError.InvalidInput;
        }

        const a = inputs[0];
        const b = inputs[1];
        var result = outputs[0];

        if (a.dtype != b.dtype or a.dtype != result.dtype) {
            return OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, a.shape, b.shape) or !std.mem.eql(usize, a.shape, result.shape)) {
            return OperatorError.ShapeMismatch;
        }

        switch (a.dtype) {
            .f32 => {
                const a_data = @as([*]const f32, @ptrCast(@alignCast(a.data.ptr)))[0..a.numel()];
                const b_data = @as([*]const f32, @ptrCast(@alignCast(b.data.ptr)))[0..b.numel()];
                const result_data = @as([*]f32, @ptrCast(@alignCast(result.data.ptr)))[0..result.numel()];

                simd.vector_mul_f32(a_data, b_data, result_data) catch {
                    return OperatorError.InvalidOperation;
                };
            },
            else => return OperatorError.UnsupportedDataType,
        }
    }
};

/// ReLU activation operator
pub const ReLU = struct {
    pub const op = Operator{
        .name = "ReLU",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return OperatorError.InvalidInput;
        }

        const input = inputs[0];
        var result = outputs[0];

        if (input.dtype != result.dtype) {
            return OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, input.shape, result.shape)) {
            return OperatorError.ShapeMismatch;
        }

        switch (input.dtype) {
            .f32 => {
                const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)))[0..input.numel()];
                const result_data = @as([*]f32, @ptrCast(@alignCast(result.data.ptr)))[0..result.numel()];

                simd.vector_relu_f32(input_data, result_data) catch {
                    return OperatorError.InvalidOperation;
                };
            },
            else => return OperatorError.UnsupportedDataType,
        }
    }
};

/// Matrix multiplication operator
pub const MatMul = struct {
    pub const op = Operator{
        .name = "MatMul",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) OperatorError!void {
        _ = allocator;

        if (inputs.len != 2 or outputs.len != 1) {
            return OperatorError.InvalidInput;
        }

        const a = inputs[0];
        const b = inputs[1];
        var result = outputs[0];

        if (a.dtype != b.dtype or a.dtype != result.dtype) {
            return OperatorError.UnsupportedDataType;
        }

        // For now, only support 2D matrix multiplication
        if (a.shape.len != 2 or b.shape.len != 2 or result.shape.len != 2) {
            return OperatorError.ShapeMismatch;
        }

        const m = a.shape[0];
        const k = a.shape[1];
        const n = b.shape[1];

        if (b.shape[0] != k or result.shape[0] != m or result.shape[1] != n) {
            return OperatorError.ShapeMismatch;
        }

        switch (a.dtype) {
            .f32 => {
                const a_data = @as([*]const f32, @ptrCast(@alignCast(a.data.ptr)));
                const b_data = @as([*]const f32, @ptrCast(@alignCast(b.data.ptr)));
                const result_data = @as([*]f32, @ptrCast(@alignCast(result.data.ptr)));

                // Simple matrix multiplication (can be optimized with BLAS later)
                for (0..m) |i| {
                    for (0..n) |j| {
                        var sum: f32 = 0.0;
                        for (0..k) |l| {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                        result_data[i * n + j] = sum;
                    }
                }
            },
            else => return OperatorError.UnsupportedDataType,
        }
    }
};

/// Softmax activation operator
pub const Softmax = struct {
    pub const op = Operator{
        .name = "Softmax",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return OperatorError.InvalidInput;
        }

        const input = inputs[0];
        var result = outputs[0];

        if (input.dtype != result.dtype) {
            return OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, input.shape, result.shape)) {
            return OperatorError.ShapeMismatch;
        }

        switch (input.dtype) {
            .f32 => {
                const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)))[0..input.numel()];
                const result_data = @as([*]f32, @ptrCast(@alignCast(result.data.ptr)))[0..result.numel()];

                // Find max for numerical stability
                var max_val: f32 = input_data[0];
                for (input_data[1..]) |val| {
                    max_val = @max(max_val, val);
                }

                // Compute exp(x - max) and sum
                var sum: f32 = 0.0;
                for (input_data, result_data) |in_val, *out_val| {
                    const exp_val = @exp(in_val - max_val);
                    out_val.* = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for (result_data) |*val| {
                    val.* /= sum;
                }
            },
            else => return OperatorError.UnsupportedDataType,
        }
    }
};

test "basic operators" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test Add operator
    const shape = [_]usize{ 2, 2 };
    var a = try tensor.Tensor.init(allocator, &shape, .f32);
    defer a.deinit();
    var b = try tensor.Tensor.init(allocator, &shape, .f32);
    defer b.deinit();
    var result = try tensor.Tensor.init(allocator, &shape, .f32);
    defer result.deinit();

    // Fill test data
    try a.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try a.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try a.set_f32(&[_]usize{ 1, 0 }, 3.0);
    try a.set_f32(&[_]usize{ 1, 1 }, 4.0);

    try b.set_f32(&[_]usize{ 0, 0 }, 0.5);
    try b.set_f32(&[_]usize{ 0, 1 }, 1.5);
    try b.set_f32(&[_]usize{ 1, 0 }, 2.5);
    try b.set_f32(&[_]usize{ 1, 1 }, 3.5);

    // Test addition
    const inputs = [_]tensor.Tensor{ a, b };
    var outputs = [_]tensor.Tensor{result};

    try Add.op.forward(&inputs, &outputs, allocator);

    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 0, 0 }), 1.5, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 0, 1 }), 3.5, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 1, 0 }), 5.5, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 1, 1 }), 7.5, 1e-6);
}

test "matrix multiplication" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test 2x3 * 3x2 = 2x2
    var a = try tensor.Tensor.init(allocator, &[_]usize{ 2, 3 }, .f32);
    defer a.deinit();
    var b = try tensor.Tensor.init(allocator, &[_]usize{ 3, 2 }, .f32);
    defer b.deinit();
    var result = try tensor.Tensor.init(allocator, &[_]usize{ 2, 2 }, .f32);
    defer result.deinit();

    // Fill A: [[1, 2, 3], [4, 5, 6]]
    try a.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try a.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try a.set_f32(&[_]usize{ 0, 2 }, 3.0);
    try a.set_f32(&[_]usize{ 1, 0 }, 4.0);
    try a.set_f32(&[_]usize{ 1, 1 }, 5.0);
    try a.set_f32(&[_]usize{ 1, 2 }, 6.0);

    // Fill B: [[1, 2], [3, 4], [5, 6]]
    try b.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try b.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try b.set_f32(&[_]usize{ 1, 0 }, 3.0);
    try b.set_f32(&[_]usize{ 1, 1 }, 4.0);
    try b.set_f32(&[_]usize{ 2, 0 }, 5.0);
    try b.set_f32(&[_]usize{ 2, 1 }, 6.0);

    const inputs = [_]tensor.Tensor{ a, b };
    var outputs = [_]tensor.Tensor{result};

    try MatMul.op.forward(&inputs, &outputs, allocator);

    // Expected result: [[22, 28], [49, 64]]
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 0, 0 }), 22.0, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 0, 1 }), 28.0, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 1, 0 }), 49.0, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 1, 1 }), 64.0, 1e-6);
}

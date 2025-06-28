const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../../core/tensor.zig");
const operators = @import("../operators.zig");

pub const PoolError = error{
    InvalidInput,
    ShapeMismatch,
    UnsupportedDataType,
    InvalidKernel,
    InvalidStride,
    OutOfMemory,
};

/// Max Pooling 2D operator
pub const MaxPool2D = struct {
    pub const op = operators.Operator{
        .name = "MaxPool2D",
        .forward_fn = forward,
    };

    pub const Params = struct {
        kernel_size: [2]usize = .{ 2, 2 },
        stride: [2]usize = .{ 2, 2 },
        padding: [4]usize = .{ 0, 0, 0, 0 }, // top, bottom, left, right
        ceil_mode: bool = false,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0]; // [N, C, H, W]
        var output = outputs[0]; // [N, C, Out_H, Out_W]

        if (input.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (input.shape.len != 4 or output.shape.len != 4) {
            return operators.OperatorError.ShapeMismatch;
        }

        const batch_size = input.shape[0];
        const channels = input.shape[1];
        const in_height = input.shape[2];
        const in_width = input.shape[3];

        const out_height = output.shape[2];
        const out_width = output.shape[3];

        // Default parameters
        const params = Params{};

        try maxpool2d_forward(
            input,
            output,
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            params,
        );
    }

    fn maxpool2d_forward(
        input: tensor.Tensor,
        output: tensor.Tensor,
        batch_size: usize,
        channels: usize,
        in_height: usize,
        in_width: usize,
        out_height: usize,
        out_width: usize,
        params: Params,
    ) !void {
        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)));
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)));

        for (0..batch_size) |n| {
            for (0..channels) |c| {
                for (0..out_height) |oh| {
                    for (0..out_width) |ow| {
                        var max_val: f32 = -std.math.inf(f32);

                        const start_h = oh * params.stride[0];
                        const start_w = ow * params.stride[1];
                        const end_h = @min(start_h + params.kernel_size[0], in_height);
                        const end_w = @min(start_w + params.kernel_size[1], in_width);

                        for (start_h..end_h) |ih| {
                            for (start_w..end_w) |iw| {
                                const input_idx = n * (channels * in_height * in_width) +
                                    c * (in_height * in_width) +
                                    ih * in_width + iw;

                                max_val = @max(max_val, input_data[input_idx]);
                            }
                        }

                        const output_idx = n * (channels * out_height * out_width) +
                            c * (out_height * out_width) +
                            oh * out_width + ow;

                        output_data[output_idx] = max_val;
                    }
                }
            }
        }
    }
};

/// Average Pooling 2D operator
pub const AvgPool2D = struct {
    pub const op = operators.Operator{
        .name = "AvgPool2D",
        .forward_fn = forward,
    };

    pub const Params = struct {
        kernel_size: [2]usize = .{ 2, 2 },
        stride: [2]usize = .{ 2, 2 },
        padding: [4]usize = .{ 0, 0, 0, 0 },
        ceil_mode: bool = false,
        count_include_pad: bool = true,
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

        if (input.shape.len != 4 or output.shape.len != 4) {
            return operators.OperatorError.ShapeMismatch;
        }

        const batch_size = input.shape[0];
        const channels = input.shape[1];
        const in_height = input.shape[2];
        const in_width = input.shape[3];

        const out_height = output.shape[2];
        const out_width = output.shape[3];

        const params = Params{};

        try avgpool2d_forward(
            input,
            output,
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            params,
        );
    }

    fn avgpool2d_forward(
        input: tensor.Tensor,
        output: tensor.Tensor,
        batch_size: usize,
        channels: usize,
        in_height: usize,
        in_width: usize,
        out_height: usize,
        out_width: usize,
        params: Params,
    ) !void {
        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)));
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)));

        for (0..batch_size) |n| {
            for (0..channels) |c| {
                for (0..out_height) |oh| {
                    for (0..out_width) |ow| {
                        var sum: f32 = 0.0;
                        var count: usize = 0;

                        const start_h = oh * params.stride[0];
                        const start_w = ow * params.stride[1];
                        const end_h = @min(start_h + params.kernel_size[0], in_height);
                        const end_w = @min(start_w + params.kernel_size[1], in_width);

                        for (start_h..end_h) |ih| {
                            for (start_w..end_w) |iw| {
                                const input_idx = n * (channels * in_height * in_width) +
                                    c * (in_height * in_width) +
                                    ih * in_width + iw;

                                sum += input_data[input_idx];
                                count += 1;
                            }
                        }

                        const output_idx = n * (channels * out_height * out_width) +
                            c * (out_height * out_width) +
                            oh * out_width + ow;

                        output_data[output_idx] = if (count > 0) sum / @as(f32, @floatFromInt(count)) else 0.0;
                    }
                }
            }
        }
    }
};

/// Global Average Pooling operator
pub const GlobalAvgPool2D = struct {
    pub const op = operators.Operator{
        .name = "GlobalAvgPool2D",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0]; // [N, C, H, W]
        var output = outputs[0]; // [N, C, 1, 1]

        if (input.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (input.shape.len != 4 or output.shape.len != 4) {
            return operators.OperatorError.ShapeMismatch;
        }

        const batch_size = input.shape[0];
        const channels = input.shape[1];
        const height = input.shape[2];
        const width = input.shape[3];

        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)));
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)));

        for (0..batch_size) |n| {
            for (0..channels) |c| {
                var sum: f32 = 0.0;

                for (0..height) |h| {
                    for (0..width) |w| {
                        const input_idx = n * (channels * height * width) +
                            c * (height * width) +
                            h * width + w;

                        sum += input_data[input_idx];
                    }
                }

                const output_idx = n * channels + c;
                const total_elements = height * width;
                output_data[output_idx] = sum / @as(f32, @floatFromInt(total_elements));
            }
        }
    }
};

/// Adaptive Average Pooling operator
pub const AdaptiveAvgPool2D = struct {
    pub const op = operators.Operator{
        .name = "AdaptiveAvgPool2D",
        .forward_fn = forward,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;
        _ = inputs;
        _ = outputs;

        // TODO: Implement adaptive average pooling
        // This automatically determines the kernel size and stride to achieve the target output size
        return operators.OperatorError.InvalidOperation;
    }
};

// Utility functions
pub fn calculatePoolOutputSize(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) usize {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

// Test functions
test "maxpool2d basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create test tensors
    // Input: [1, 1, 4, 4] - single batch, single channel, 4x4 image
    const input_shape = [_]usize{ 1, 1, 4, 4 };
    var input = try tensor.Tensor.init(allocator, &input_shape, .f32);
    defer input.deinit();

    // Output: [1, 1, 2, 2] - single batch, single channel, 2x2 output (2x2 pooling)
    const output_shape = [_]usize{ 1, 1, 2, 2 };
    var output = try tensor.Tensor.init(allocator, &output_shape, .f32);
    defer output.deinit();

    // Fill input with test data
    for (0..4) |i| {
        for (0..4) |j| {
            try input.set_f32(&[_]usize{ 0, 0, i, j }, @as(f32, @floatFromInt(i * 4 + j + 1)));
        }
    }

    const inputs = [_]tensor.Tensor{input};
    var outputs = [_]tensor.Tensor{output};

    try MaxPool2D.op.forward(&inputs, &outputs, allocator);

    // Check output values
    // Expected: max of each 2x2 region
    const result_00 = try output.get_f32(&[_]usize{ 0, 0, 0, 0 }); // max(1,2,5,6) = 6
    const result_01 = try output.get_f32(&[_]usize{ 0, 0, 0, 1 }); // max(3,4,7,8) = 8
    const result_10 = try output.get_f32(&[_]usize{ 0, 0, 1, 0 }); // max(9,10,13,14) = 14
    const result_11 = try output.get_f32(&[_]usize{ 0, 0, 1, 1 }); // max(11,12,15,16) = 16

    try testing.expect(result_00 == 6.0);
    try testing.expect(result_01 == 8.0);
    try testing.expect(result_10 == 14.0);
    try testing.expect(result_11 == 16.0);
}

test "avgpool2d basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create test tensors
    const input_shape = [_]usize{ 1, 1, 2, 2 };
    var input = try tensor.Tensor.init(allocator, &input_shape, .f32);
    defer input.deinit();

    const output_shape = [_]usize{ 1, 1, 1, 1 };
    var output = try tensor.Tensor.init(allocator, &output_shape, .f32);
    defer output.deinit();

    // Fill input: [1, 2; 3, 4]
    try input.set_f32(&[_]usize{ 0, 0, 0, 0 }, 1.0);
    try input.set_f32(&[_]usize{ 0, 0, 0, 1 }, 2.0);
    try input.set_f32(&[_]usize{ 0, 0, 1, 0 }, 3.0);
    try input.set_f32(&[_]usize{ 0, 0, 1, 1 }, 4.0);

    const inputs = [_]tensor.Tensor{input};
    var outputs = [_]tensor.Tensor{output};

    try AvgPool2D.op.forward(&inputs, &outputs, allocator);

    // Check output value: average of (1,2,3,4) = 2.5
    const result = try output.get_f32(&[_]usize{ 0, 0, 0, 0 });
    try testing.expect(result == 2.5);
}

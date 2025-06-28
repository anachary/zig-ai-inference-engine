const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../../core/tensor.zig");
const operators = @import("../operators.zig");

pub const NormError = error{
    InvalidInput,
    ShapeMismatch,
    UnsupportedDataType,
    InvalidEpsilon,
    OutOfMemory,
};

/// Batch Normalization operator
pub const BatchNorm2D = struct {
    pub const op = operators.Operator{
        .name = "BatchNorm2D",
        .forward_fn = forward,
    };

    pub const Params = struct {
        eps: f32 = 1e-5,
        momentum: f32 = 0.1,
        training: bool = false,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        // inputs: [input, weight(gamma), bias(beta), running_mean, running_var]
        // outputs: [output]
        if (inputs.len < 3 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0]; // [N, C, H, W]
        const gamma = inputs[1]; // [C]
        const beta = inputs[2]; // [C]
        var output = outputs[0]; // [N, C, H, W]

        if (input.dtype != .f32 or gamma.dtype != .f32 or beta.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (input.shape.len != 4 or gamma.shape.len != 1 or beta.shape.len != 1 or output.shape.len != 4) {
            return operators.OperatorError.ShapeMismatch;
        }

        const batch_size = input.shape[0];
        const channels = input.shape[1];
        const height = input.shape[2];
        const width = input.shape[3];

        if (gamma.shape[0] != channels or beta.shape[0] != channels) {
            return operators.OperatorError.ShapeMismatch;
        }

        const params = Params{};

        try batchnorm2d_forward(
            input,
            gamma,
            beta,
            output,
            batch_size,
            channels,
            height,
            width,
            params,
        );
    }

    fn batchnorm2d_forward(
        input: tensor.Tensor,
        gamma: tensor.Tensor,
        beta: tensor.Tensor,
        output: tensor.Tensor,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        params: Params,
    ) !void {
        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)));
        const gamma_data = @as([*]const f32, @ptrCast(@alignCast(gamma.data.ptr)));
        const beta_data = @as([*]const f32, @ptrCast(@alignCast(beta.data.ptr)));
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)));

        const spatial_size = height * width;
        const total_spatial = batch_size * spatial_size;

        // Compute mean and variance for each channel
        for (0..channels) |c| {
            // Compute mean
            var sum: f32 = 0.0;
            for (0..batch_size) |n| {
                for (0..spatial_size) |s| {
                    const idx = n * (channels * spatial_size) + c * spatial_size + s;
                    sum += input_data[idx];
                }
            }
            const mean = sum / @as(f32, @floatFromInt(total_spatial));

            // Compute variance
            var var_sum: f32 = 0.0;
            for (0..batch_size) |n| {
                for (0..spatial_size) |s| {
                    const idx = n * (channels * spatial_size) + c * spatial_size + s;
                    const diff = input_data[idx] - mean;
                    var_sum += diff * diff;
                }
            }
            const variance = var_sum / @as(f32, @floatFromInt(total_spatial));

            // Normalize and scale
            const inv_std = 1.0 / @sqrt(variance + params.eps);
            for (0..batch_size) |n| {
                for (0..spatial_size) |s| {
                    const idx = n * (channels * spatial_size) + c * spatial_size + s;
                    const normalized = (input_data[idx] - mean) * inv_std;
                    output_data[idx] = gamma_data[c] * normalized + beta_data[c];
                }
            }
        }
    }
};

/// Layer Normalization operator
pub const LayerNorm = struct {
    pub const op = operators.Operator{
        .name = "LayerNorm",
        .forward_fn = forward,
    };

    pub const Params = struct {
        eps: f32 = 1e-5,
        normalized_shape: []const usize,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        // inputs: [input, weight(gamma), bias(beta)]
        // outputs: [output]
        if (inputs.len != 3 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0];
        const gamma = inputs[1];
        const beta = inputs[2];
        var output = outputs[0];

        if (input.dtype != .f32 or gamma.dtype != .f32 or beta.dtype != .f32 or output.dtype != .f32) {
            return operators.OperatorError.UnsupportedDataType;
        }

        if (!std.mem.eql(usize, input.shape, output.shape)) {
            return operators.OperatorError.ShapeMismatch;
        }

        const params = Params{ .normalized_shape = input.shape };

        try layernorm_forward(input, gamma, beta, output, params);
    }

    fn layernorm_forward(
        input: tensor.Tensor,
        gamma: tensor.Tensor,
        beta: tensor.Tensor,
        output: tensor.Tensor,
        params: Params,
    ) !void {
        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)));
        const gamma_data = @as([*]const f32, @ptrCast(@alignCast(gamma.data.ptr)));
        const beta_data = @as([*]const f32, @ptrCast(@alignCast(beta.data.ptr)));
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)));

        const total_elements = input.numel();
        const normalized_size = gamma.numel();

        // For simplicity, assume we're normalizing over the last dimension
        const batch_size = total_elements / normalized_size;

        for (0..batch_size) |batch_idx| {
            const offset = batch_idx * normalized_size;

            // Compute mean
            var sum: f32 = 0.0;
            for (0..normalized_size) |i| {
                sum += input_data[offset + i];
            }
            const mean = sum / @as(f32, @floatFromInt(normalized_size));

            // Compute variance
            var var_sum: f32 = 0.0;
            for (0..normalized_size) |i| {
                const diff = input_data[offset + i] - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f32, @floatFromInt(normalized_size));

            // Normalize and scale
            const inv_std = 1.0 / @sqrt(variance + params.eps);
            for (0..normalized_size) |i| {
                const normalized = (input_data[offset + i] - mean) * inv_std;
                output_data[offset + i] = gamma_data[i] * normalized + beta_data[i];
            }
        }
    }
};

/// Group Normalization operator
pub const GroupNorm = struct {
    pub const op = operators.Operator{
        .name = "GroupNorm",
        .forward_fn = forward,
    };

    pub const Params = struct {
        num_groups: usize,
        eps: f32 = 1e-5,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;
        _ = inputs;
        _ = outputs;

        // TODO: Implement group normalization
        // Groups channels and normalizes within each group
        return operators.OperatorError.InvalidOperation;
    }
};

/// Instance Normalization operator
pub const InstanceNorm2D = struct {
    pub const op = operators.Operator{
        .name = "InstanceNorm2D",
        .forward_fn = forward,
    };

    pub const Params = struct {
        eps: f32 = 1e-5,
        affine: bool = true,
    };

    fn forward(inputs: []const tensor.Tensor, outputs: []tensor.Tensor, allocator: Allocator) operators.OperatorError!void {
        _ = allocator;

        if (inputs.len < 1 or outputs.len != 1) {
            return operators.OperatorError.InvalidInput;
        }

        const input = inputs[0]; // [N, C, H, W]
        var output = outputs[0]; // [N, C, H, W]

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

        const params = Params{};

        try instancenorm2d_forward(
            input,
            output,
            batch_size,
            channels,
            height,
            width,
            params,
        );
    }

    fn instancenorm2d_forward(
        input: tensor.Tensor,
        output: tensor.Tensor,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        params: Params,
    ) !void {
        const input_data = @as([*]const f32, @ptrCast(@alignCast(input.data.ptr)));
        const output_data = @as([*]f32, @ptrCast(@alignCast(output.data.ptr)));

        const spatial_size = height * width;

        // Normalize each instance (N, C) separately
        for (0..batch_size) |n| {
            for (0..channels) |c| {
                // Compute mean for this instance
                var sum: f32 = 0.0;
                for (0..spatial_size) |s| {
                    const idx = n * (channels * spatial_size) + c * spatial_size + s;
                    sum += input_data[idx];
                }
                const mean = sum / @as(f32, @floatFromInt(spatial_size));

                // Compute variance for this instance
                var var_sum: f32 = 0.0;
                for (0..spatial_size) |s| {
                    const idx = n * (channels * spatial_size) + c * spatial_size + s;
                    const diff = input_data[idx] - mean;
                    var_sum += diff * diff;
                }
                const variance = var_sum / @as(f32, @floatFromInt(spatial_size));

                // Normalize
                const inv_std = 1.0 / @sqrt(variance + params.eps);
                for (0..spatial_size) |s| {
                    const idx = n * (channels * spatial_size) + c * spatial_size + s;
                    output_data[idx] = (input_data[idx] - mean) * inv_std;
                }
            }
        }
    }
};

// Test functions
test "batchnorm2d basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create test tensors
    const input_shape = [_]usize{ 1, 2, 2, 2 }; // [N=1, C=2, H=2, W=2]
    var input = try tensor.Tensor.init(allocator, &input_shape, .f32);
    defer input.deinit();

    const param_shape = [_]usize{2}; // [C=2]
    var gamma = try tensor.Tensor.init(allocator, &param_shape, .f32);
    defer gamma.deinit();

    var beta = try tensor.Tensor.init(allocator, &param_shape, .f32);
    defer beta.deinit();

    var output = try tensor.Tensor.init(allocator, &input_shape, .f32);
    defer output.deinit();

    // Fill input with test data
    try input.set_f32(&[_]usize{ 0, 0, 0, 0 }, 1.0);
    try input.set_f32(&[_]usize{ 0, 0, 0, 1 }, 2.0);
    try input.set_f32(&[_]usize{ 0, 0, 1, 0 }, 3.0);
    try input.set_f32(&[_]usize{ 0, 0, 1, 1 }, 4.0);

    try input.set_f32(&[_]usize{ 0, 1, 0, 0 }, 5.0);
    try input.set_f32(&[_]usize{ 0, 1, 0, 1 }, 6.0);
    try input.set_f32(&[_]usize{ 0, 1, 1, 0 }, 7.0);
    try input.set_f32(&[_]usize{ 0, 1, 1, 1 }, 8.0);

    // Set gamma and beta
    try gamma.set_f32(&[_]usize{0}, 1.0);
    try gamma.set_f32(&[_]usize{1}, 1.0);
    try beta.set_f32(&[_]usize{0}, 0.0);
    try beta.set_f32(&[_]usize{1}, 0.0);

    const inputs = [_]tensor.Tensor{ input, gamma, beta };
    var outputs = [_]tensor.Tensor{output};

    try BatchNorm2D.op.forward(&inputs, &outputs, allocator);

    // The output should be normalized (mean=0, std=1 for each channel)
    // Just check that the operation completed without error
    const result = try output.get_f32(&[_]usize{ 0, 0, 0, 0 });
    try testing.expect(result != 0.0); // Should be some normalized value
}

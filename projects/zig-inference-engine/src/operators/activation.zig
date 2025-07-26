const std = @import("std");
const Allocator = std.mem.Allocator;

// Import common interfaces and registry
const common_interfaces = @import("common-interfaces");
const TensorInterface = common_interfaces.TensorInterface;
const OperatorInfo = @import("registry.zig").OperatorInfo;
const OperatorFn = @import("registry.zig").OperatorFn;
const ValidatorFn = @import("registry.zig").ValidatorFn;

/// ReLU activation function: f(x) = max(0, x)
pub const ReLU = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "ReLU",
            .description = "Rectified Linear Unit activation function",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
            .compute_fn = compute,
            .validate_fn = validate,
        };
    }

    fn compute(
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror!void {
        _ = attributes;
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        const numel = input.numel();
        const input_data = input.data();
        const output_data = output.data();

        switch (input.dtype()) {
            .f32 => {
                const in_f32 = std.mem.bytesAsSlice(f32, input_data);
                const out_f32 = std.mem.bytesAsSlice(f32, output_data);

                for (0..numel) |i| {
                    out_f32[i] = @max(0.0, in_f32[i]);
                }
            },
            .f16 => {
                const in_f16 = std.mem.bytesAsSlice(f16, input_data);
                const out_f16 = std.mem.bytesAsSlice(f16, output_data);

                for (0..numel) |i| {
                    out_f16[i] = @max(0.0, in_f16[i]);
                }
            },
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        if (input_shapes.len != 1) {
            return error.InvalidInputCount;
        }

        var result = try allocator.alloc([]usize, 1);
        result[0] = try allocator.dupe(usize, input_shapes[0]);
        return result;
    }
};

/// Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
pub const Sigmoid = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Sigmoid",
            .description = "Sigmoid activation function",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
            .compute_fn = compute,
            .validate_fn = validate,
        };
    }

    fn compute(
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror!void {
        _ = attributes;
        _ = allocator;

        const input = &inputs[0];
        const output = &outputs[0];

        const numel = input.numel();
        const input_data = input.data();
        const output_data = output.data();

        switch (input.dtype()) {
            .f32 => {
                const in_f32 = std.mem.bytesAsSlice(f32, input_data);
                const out_f32 = std.mem.bytesAsSlice(f32, output_data);

                for (0..numel) |i| {
                    out_f32[i] = 1.0 / (1.0 + std.math.exp(-in_f32[i]));
                }
            },
            .f16 => {
                const in_f16 = std.mem.bytesAsSlice(f16, input_data);
                const out_f16 = std.mem.bytesAsSlice(f16, output_data);

                for (0..numel) |i| {
                    out_f16[i] = 1.0 / (1.0 + std.math.exp(-in_f16[i]));
                }
            },
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        var result = try allocator.alloc([]usize, 1);
        result[0] = try allocator.dupe(usize, input_shapes[0]);
        return result;
    }
};

/// Tanh activation function: f(x) = tanh(x)
pub const Tanh = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Tanh",
            .description = "Hyperbolic tangent activation function",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
            .compute_fn = compute,
            .validate_fn = validate,
        };
    }

    fn compute(
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror!void {
        _ = attributes;
        _ = allocator;

        const input = &inputs[0];
        const output = &outputs[0];

        const numel = input.numel();
        const input_data = input.data();
        const output_data = output.data();

        switch (input.dtype()) {
            .f32 => {
                const in_f32 = std.mem.bytesAsSlice(f32, input_data);
                const out_f32 = std.mem.bytesAsSlice(f32, output_data);

                for (0..numel) |i| {
                    out_f32[i] = std.math.tanh(in_f32[i]);
                }
            },
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        var result = try allocator.alloc([]usize, 1);
        result[0] = try allocator.dupe(usize, input_shapes[0]);
        return result;
    }
};

/// Softmax activation function: f(x_i) = exp(x_i) / sum(exp(x_j))
pub const Softmax = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Softmax",
            .description = "Softmax activation function",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
            .compute_fn = compute,
            .validate_fn = validate,
        };
    }

    fn compute(
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror!void {
        _ = allocator;

        const input = &inputs[0];
        const output = &outputs[0];

        // Parse axis attribute (default to last dimension)
        const axis = parseAxisAttribute(attributes, "axis", input.shape().len) orelse (input.shape().len - 1);

        switch (input.dtype()) {
            .f32 => {
                try softmaxF32(input, output, axis);
            },
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        var result = try allocator.alloc([]usize, 1);
        result[0] = try allocator.dupe(usize, input_shapes[0]);
        return result;
    }
};

/// GELU activation function: f(x) = x * Φ(x) where Φ is the CDF of standard normal
pub const GELU = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "GELU",
            .description = "Gaussian Error Linear Unit activation function",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
            .compute_fn = compute,
            .validate_fn = validate,
        };
    }

    fn compute(
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror!void {
        _ = attributes;
        _ = allocator;

        const input = &inputs[0];
        const output = &outputs[0];

        const numel = input.numel();
        const input_data = input.data();
        const output_data = output.data();

        switch (input.dtype()) {
            .f32 => {
                const in_f32 = std.mem.bytesAsSlice(f32, input_data);
                const out_f32 = std.mem.bytesAsSlice(f32, output_data);

                for (0..numel) |i| {
                    const x = in_f32[i];
                    // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
                    const x_cubed = x * x * x;
                    const inner = std.math.sqrt(2.0 / std.math.pi) * (x + 0.044715 * x_cubed);
                    out_f32[i] = 0.5 * x * (1.0 + std.math.tanh(inner));
                }
            },
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        var result = try allocator.alloc([]usize, 1);
        result[0] = try allocator.dupe(usize, input_shapes[0]);
        return result;
    }
};

/// Swish activation function: f(x) = x * sigmoid(x)
pub const Swish = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Swish",
            .description = "Swish activation function (x * sigmoid(x))",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
            .compute_fn = compute,
            .validate_fn = validate,
        };
    }

    fn compute(
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror!void {
        _ = attributes;
        _ = allocator;

        const input = &inputs[0];
        const output = &outputs[0];

        const numel = input.numel();
        const input_data = input.data();
        const output_data = output.data();

        switch (input.dtype()) {
            .f32 => {
                const in_f32 = std.mem.bytesAsSlice(f32, input_data);
                const out_f32 = std.mem.bytesAsSlice(f32, output_data);

                for (0..numel) |i| {
                    const x = in_f32[i];
                    const sigmoid_x = 1.0 / (1.0 + std.math.exp(-x));
                    out_f32[i] = x * sigmoid_x;
                }
            },
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        var result = try allocator.alloc([]usize, 1);
        result[0] = try allocator.dupe(usize, input_shapes[0]);
        return result;
    }
};

// Helper functions

fn softmaxF32(input: *const TensorInterface, output: *const TensorInterface, axis: usize) !void {
    const input_data = std.mem.bytesAsSlice(f32, input.data());
    const output_data = std.mem.bytesAsSlice(f32, output.data());
    const shape = input.shape();

    if (axis >= shape.len) {
        return error.InvalidAxis;
    }

    const axis_size = shape[axis];
    const outer_size = calculateOuterSize(shape, axis);
    const inner_size = calculateInnerSize(shape, axis);

    for (0..outer_size) |outer| {
        for (0..inner_size) |inner| {
            // Find maximum for numerical stability
            var max_val: f32 = -std.math.inf(f32);
            for (0..axis_size) |i| {
                const idx = outer * axis_size * inner_size + i * inner_size + inner;
                max_val = @max(max_val, input_data[idx]);
            }

            // Compute exponentials and sum
            var sum: f32 = 0.0;
            for (0..axis_size) |i| {
                const idx = outer * axis_size * inner_size + i * inner_size + inner;
                const exp_val = std.math.exp(input_data[idx] - max_val);
                output_data[idx] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for (0..axis_size) |i| {
                const idx = outer * axis_size * inner_size + i * inner_size + inner;
                output_data[idx] /= sum;
            }
        }
    }
}

fn parseAxisAttribute(attributes: std.StringHashMap([]const u8), key: []const u8, ndim: usize) ?usize {
    if (attributes.get(key)) |value| {
        if (std.fmt.parseInt(i32, value, 10)) |axis| {
            // Handle negative axis
            const normalized_axis = if (axis < 0)
                @as(usize, @intCast(@as(i32, @intCast(ndim)) + axis))
            else
                @as(usize, @intCast(axis));

            if (normalized_axis < ndim) {
                return normalized_axis;
            }
        } else |_| {}
    }
    return null;
}

fn calculateOuterSize(shape: []const usize, axis: usize) usize {
    var size: usize = 1;
    for (0..axis) |i| {
        size *= shape[i];
    }
    return size;
}

fn calculateInnerSize(shape: []const usize, axis: usize) usize {
    var size: usize = 1;
    for (axis + 1..shape.len) |i| {
        size *= shape[i];
    }
    return size;
}

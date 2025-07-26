const std = @import("std");
const Allocator = std.mem.Allocator;

// Import common interfaces and registry
const common_interfaces = @import("common-interfaces");
const TensorInterface = common_interfaces.TensorInterface;
const OperatorInfo = @import("registry.zig").OperatorInfo;
const OperatorFn = @import("registry.zig").OperatorFn;
const ValidatorFn = @import("registry.zig").ValidatorFn;

/// Concatenation operator
pub const Concat = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Concat",
            .description = "Concatenate tensors along specified axis",
            .min_inputs = 2,
            .max_inputs = 100,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
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
        _ = inputs;
        _ = outputs;
        _ = attributes;
        _ = allocator;
        // TODO: Implement Concat
        return error.NotImplemented;
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = input_shapes;
        _ = attributes;
        _ = allocator;
        // TODO: Implement validation
        return error.NotImplemented;
    }
};

/// Split operator
pub const Split = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Split",
            .description = "Split tensor along specified axis",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 2,
            .max_outputs = 100,
            .supports_inplace = false,
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
        _ = inputs;
        _ = outputs;
        _ = attributes;
        _ = allocator;
        // TODO: Implement Split
        return error.NotImplemented;
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = input_shapes;
        _ = attributes;
        _ = allocator;
        // TODO: Implement validation
        return error.NotImplemented;
    }
};

/// Slice operator
pub const Slice = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Slice",
            .description = "Extract slice from tensor",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
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

        if (inputs.len != 1 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        // Parse slice parameters from attributes
        const starts = parseIntArrayAttribute(attributes, "starts") orelse return error.MissingAttribute;
        const ends = parseIntArrayAttribute(attributes, "ends") orelse return error.MissingAttribute;
        const axes = parseIntArrayAttribute(attributes, "axes");
        const steps = parseIntArrayAttribute(attributes, "steps");

        switch (input.dtype()) {
            .f32 => try sliceF32(input, output, starts, ends, axes, steps),
            .i32 => try sliceI32(input, output, starts, ends, axes, steps),
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = input_shapes;
        _ = attributes;
        _ = allocator;
        // TODO: Implement validation
        return error.NotImplemented;
    }
};

/// Squeeze operator
pub const Squeeze = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Squeeze",
            .description = "Remove dimensions of size 1",
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

        if (inputs.len != 1 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        // Parse axes to squeeze from attributes
        const axes = parseIntArrayAttribute(attributes, "axes");

        // Copy data (squeeze is just a shape change)
        const input_data = input.data();
        const output_data = output.data();
        @memcpy(output_data, input_data);

        // Squeeze operation removes dimensions of size 1
        // The actual shape change is handled by the tensor interface
        _ = axes; // TODO: Use axes to determine which dimensions to squeeze
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = input_shapes;
        _ = attributes;
        _ = allocator;
        // TODO: Implement validation
        return error.NotImplemented;
    }
};

/// Unsqueeze operator
pub const Unsqueeze = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Unsqueeze",
            .description = "Add dimensions of size 1",
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

        if (inputs.len != 1 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        // Parse axes to unsqueeze from attributes
        const axes = parseIntArrayAttribute(attributes, "axes");

        // Copy data (unsqueeze is just a shape change)
        const input_data = input.data();
        const output_data = output.data();
        @memcpy(output_data, input_data);

        // Unsqueeze operation adds dimensions of size 1
        // The actual shape change is handled by the tensor interface
        _ = axes; // TODO: Use axes to determine where to add dimensions
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = input_shapes;
        _ = attributes;
        _ = allocator;
        // TODO: Implement validation
        return error.NotImplemented;
    }
};

/// Gather operator
pub const Gather = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Gather",
            .description = "Gather elements from tensor using indices",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
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
        _ = inputs;
        _ = outputs;
        _ = attributes;
        _ = allocator;
        // TODO: Implement Gather
        return error.NotImplemented;
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = input_shapes;
        _ = attributes;
        _ = allocator;
        // TODO: Implement validation
        return error.NotImplemented;
    }
};

/// Scatter operator
pub const Scatter = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Scatter",
            .description = "Scatter elements into tensor using indices",
            .min_inputs = 3,
            .max_inputs = 3,
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
        _ = inputs;
        _ = outputs;
        _ = attributes;
        _ = allocator;
        // TODO: Implement Scatter
        return error.NotImplemented;
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = input_shapes;
        _ = attributes;
        _ = allocator;
        // TODO: Implement validation
        return error.NotImplemented;
    }
};

/// Reshape operator
pub const Reshape = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Reshape",
            .description = "Reshape tensor to new shape",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
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

        if (inputs.len != 2 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const input = &inputs[0];
        const shape_tensor = &inputs[1];
        const output = &outputs[0];

        // Extract new shape from shape tensor
        const shape_data = std.mem.bytesAsSlice(i64, shape_tensor.data());

        // Validate that total elements match
        var new_total: usize = 1;
        for (shape_data) |dim| {
            if (dim <= 0 and dim != -1) return error.InvalidShape;
            if (dim > 0) new_total *= @intCast(dim);
        }

        // Handle -1 dimension (infer size)
        var inferred_dim: ?usize = null;
        for (shape_data, 0..) |dim, i| {
            if (dim == -1) {
                if (inferred_dim != null) return error.MultipleInferredDims;
                inferred_dim = i;
            }
        }

        if (inferred_dim) |idx| {
            _ = idx;
            const remaining = input.numel() / new_total;
            new_total *= remaining;
        }

        if (new_total != input.numel()) {
            return error.ShapeMismatch;
        }

        // Copy data (reshape is just a view change)
        const input_data = input.data();
        const output_data = output.data();
        @memcpy(output_data, input_data);
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        if (input_shapes.len != 2) return error.InvalidInputCount;

        // First input is data tensor, second is shape tensor
        const data_shape = input_shapes[0];
        const shape_shape = input_shapes[1];

        // Shape tensor should be 1D
        if (shape_shape.len != 1) return error.InvalidShapeInput;

        // Output shape will be determined at runtime
        // For now, return a placeholder shape
        var output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, data_shape);

        return output_shapes;
    }
};

/// Constant operator
pub const Constant = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Constant",
            .description = "Output a constant tensor",
            .min_inputs = 0,
            .max_inputs = 0,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
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

        if (inputs.len != 0 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const output = &outputs[0];

        // Get constant value from attributes
        if (attributes.get("value")) |value_str| {
            // Parse the constant value and fill the output tensor
            // This is a simplified implementation
            const output_data = output.data();

            switch (output.dtype()) {
                .f32 => {
                    const value = std.fmt.parseFloat(f32, value_str) catch 0.0;
                    const data = std.mem.bytesAsSlice(f32, output_data);
                    for (data) |*elem| {
                        elem.* = value;
                    }
                },
                .i32 => {
                    const value = std.fmt.parseInt(i32, value_str, 10) catch 0;
                    const data = std.mem.bytesAsSlice(i32, output_data);
                    for (data) |*elem| {
                        elem.* = value;
                    }
                },
                else => return error.UnsupportedDataType,
            }
        } else {
            // Fill with zeros if no value specified
            @memset(output.data(), 0);
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = input_shapes;
        _ = attributes;

        // Output shape is determined by the 'shape' attribute
        var output_shapes = try allocator.alloc([]usize, 1);

        // Default to scalar if no shape specified
        output_shapes[0] = try allocator.alloc(usize, 1);
        output_shapes[0][0] = 1;

        return output_shapes;
    }
};

// Helper functions for attribute parsing
fn parseIntArrayAttribute(attributes: std.StringHashMap([]const u8), name: []const u8) ?[]i32 {
    _ = attributes;
    _ = name;
    // TODO: Implement proper attribute parsing
    // For now, return null to indicate missing attributes
    return null;
}

fn sliceF32(input: *const TensorInterface, output: *const TensorInterface, starts: []i32, ends: []i32, axes: ?[]i32, steps: ?[]i32) !void {
    _ = input;
    _ = output;
    _ = starts;
    _ = ends;
    _ = axes;
    _ = steps;
    // TODO: Implement F32 slicing
    return error.NotImplemented;
}

fn sliceI32(input: *const TensorInterface, output: *const TensorInterface, starts: []i32, ends: []i32, axes: ?[]i32, steps: ?[]i32) !void {
    _ = input;
    _ = output;
    _ = starts;
    _ = ends;
    _ = axes;
    _ = steps;
    // TODO: Implement I32 slicing
    return error.NotImplemented;
}

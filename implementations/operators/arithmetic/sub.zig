const std = @import("std");
const framework = @import("../../../framework/lib.zig");

const Tensor = framework.Tensor;
const Attributes = framework.Attributes;
const ExecutionContext = framework.ExecutionContext;
const FrameworkError = framework.FrameworkError;
const OperatorInterface = framework.OperatorInterface;
const BaseOperator = framework.BaseOperator;

/// Subtraction operator implementation
pub const Sub = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Sub",
            .version = "1.0.0",
            .description = "Element-wise subtraction of two tensors with broadcasting support",
            .domain = "ai.onnx",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = true,
            .type_constraints = &[_]OperatorInterface.TypeConstraint{
                OperatorInterface.TypeConstraint{
                    .name = "T",
                    .allowed_types = &[_]Tensor.DataType{ .f32, .f16, .i32, .i16, .i8 },
                    .description = "Constrain input and output types to numeric tensors",
                },
            },
        };
    }

    pub fn validate(
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void {
        _ = attributes;

        if (input_shapes.len != 2) {
            return FrameworkError.InvalidInput;
        }

        if (input_types[0] != input_types[1]) {
            return FrameworkError.DataTypeMismatch;
        }

        if (!Self.checkBroadcastCompatibility(input_shapes[0], input_shapes[1])) {
            return FrameworkError.ShapeMismatch;
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len != 2) {
            return FrameworkError.InvalidInput;
        }

        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try Self.calculateBroadcastShape(input_shapes[0], input_shapes[1], allocator);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = attributes;

        if (inputs.len != 2 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const a = &inputs[0];
        const b = &inputs[1];
        const output = &outputs[0];

        if (a.dtype != b.dtype or a.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        switch (a.dtype) {
            .f32 => try computeF32(a, b, output, context),
            .i32 => try computeI32(a, b, output, context),
            .i16 => try computeI16(a, b, output, context),
            .i8 => try computeI8(a, b, output, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn computeF32(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        const a_data = a.getData(f32);
        const b_data = b.getData(f32);
        const output_data = output.getMutableData(f32);

        if (Self.shapesEqual(a.shape, b.shape) and Self.shapesEqual(a.shape, output.shape)) {
            try subElementwise(f32, a_data, b_data, output_data, context);
        } else {
            try subWithBroadcasting(f32, a, b, output, context);
        }
    }

    fn computeI32(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        const a_data = a.getData(i32);
        const b_data = b.getData(i32);
        const output_data = output.getMutableData(i32);

        if (Self.shapesEqual(a.shape, b.shape) and Self.shapesEqual(a.shape, output.shape)) {
            try subElementwise(i32, a_data, b_data, output_data, context);
        } else {
            try subWithBroadcasting(i32, a, b, output, context);
        }
    }

    fn computeI16(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        const a_data = a.getData(i16);
        const b_data = b.getData(i16);
        const output_data = output.getMutableData(i16);

        if (Self.shapesEqual(a.shape, b.shape) and Self.shapesEqual(a.shape, output.shape)) {
            try subElementwise(i16, a_data, b_data, output_data, context);
        } else {
            try subWithBroadcasting(i16, a, b, output, context);
        }
    }

    fn computeI8(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        const a_data = a.getData(i8);
        const b_data = b.getData(i8);
        const output_data = output.getMutableData(i8);

        if (Self.shapesEqual(a.shape, b.shape) and Self.shapesEqual(a.shape, output.shape)) {
            try subElementwise(i8, a_data, b_data, output_data, context);
        } else {
            try subWithBroadcasting(i8, a, b, output, context);
        }
    }

    fn subElementwise(comptime T: type, a_data: []const T, b_data: []const T, output_data: []T, context: *ExecutionContext) !void {
        _ = context;
        
        if (a_data.len != b_data.len or a_data.len != output_data.len) {
            return FrameworkError.ShapeMismatch;
        }

        for (0..output_data.len) |i| {
            output_data[i] = a_data[i] - b_data[i];
        }
    }

    fn subWithBroadcasting(comptime T: type, a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const a_data = a.getData(T);
        const b_data = b.getData(T);
        const output_data = output.getMutableData(T);

        const output_elements = framework.utils.calculateTotalElements(output.shape);
        
        for (0..output_elements) |i| {
            const a_idx = calculateBroadcastIndex(i, output.shape, a.shape, a.strides);
            const b_idx = calculateBroadcastIndex(i, output.shape, b.shape, b.strides);
            
            output_data[i] = a_data[a_idx] - b_data[b_idx];
        }
    }

    fn calculateBroadcastIndex(linear_idx: usize, output_shape: []const usize, input_shape: []const usize, input_strides: []const usize) usize {
        var idx: usize = 0;
        var remaining = linear_idx;
        
        const ndim = output_shape.len;
        var i = ndim;
        while (i > 0) {
            i -= 1;
            const coord = remaining % output_shape[i];
            remaining /= output_shape[i];
            
            const input_coord = if (i < input_shape.len and input_shape[i] > 1) coord % input_shape[i] else 0;
            if (i < input_strides.len) {
                idx += input_coord * input_strides[i];
            }
        }
        
        return idx;
    }
});

// Similar implementations for Mul and Div
pub const Mul = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Mul",
            .version = "1.0.0",
            .description = "Element-wise multiplication of two tensors with broadcasting support",
            .domain = "ai.onnx",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = true,
            .type_constraints = &[_]OperatorInterface.TypeConstraint{
                OperatorInterface.TypeConstraint{
                    .name = "T",
                    .allowed_types = &[_]Tensor.DataType{ .f32, .f16, .i32, .i16, .i8 },
                    .description = "Constrain input and output types to numeric tensors",
                },
            },
        };
    }

    pub fn validate(
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void {
        _ = attributes;

        if (input_shapes.len != 2) {
            return FrameworkError.InvalidInput;
        }

        if (input_types[0] != input_types[1]) {
            return FrameworkError.DataTypeMismatch;
        }

        if (!Self.checkBroadcastCompatibility(input_shapes[0], input_shapes[1])) {
            return FrameworkError.ShapeMismatch;
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len != 2) {
            return FrameworkError.InvalidInput;
        }

        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try Self.calculateBroadcastShape(input_shapes[0], input_shapes[1], allocator);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = attributes;

        if (inputs.len != 2 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const a = &inputs[0];
        const b = &inputs[1];
        const output = &outputs[0];

        if (a.dtype != b.dtype or a.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        switch (a.dtype) {
            .f32 => try mulF32(a, b, output, context),
            .i32 => try mulI32(a, b, output, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn mulF32(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        const a_data = a.getData(f32);
        const b_data = b.getData(f32);
        const output_data = output.getMutableData(f32);

        if (framework.utils.shapesEqual(a.shape, b.shape) and framework.utils.shapesEqual(a.shape, output.shape)) {
            for (0..output_data.len) |i| {
                output_data[i] = a_data[i] * b_data[i];
            }
        } else {
            // Broadcasting multiplication
            const output_elements = framework.utils.calculateTotalElements(output.shape);
            for (0..output_elements) |i| {
                const a_idx = calculateBroadcastIndex(i, output.shape, a.shape, a.strides);
                const b_idx = calculateBroadcastIndex(i, output.shape, b.shape, b.strides);
                output_data[i] = a_data[a_idx] * b_data[b_idx];
            }
        }
    }

    fn mulI32(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        const a_data = a.getData(i32);
        const b_data = b.getData(i32);
        const output_data = output.getMutableData(i32);

        if (framework.utils.shapesEqual(a.shape, b.shape) and framework.utils.shapesEqual(a.shape, output.shape)) {
            for (0..output_data.len) |i| {
                output_data[i] = a_data[i] * b_data[i];
            }
        } else {
            const output_elements = framework.utils.calculateTotalElements(output.shape);
            for (0..output_elements) |i| {
                const a_idx = calculateBroadcastIndex(i, output.shape, a.shape, a.strides);
                const b_idx = calculateBroadcastIndex(i, output.shape, b.shape, b.strides);
                output_data[i] = a_data[a_idx] * b_data[b_idx];
            }
        }
    }

    fn calculateBroadcastIndex(linear_idx: usize, output_shape: []const usize, input_shape: []const usize, input_strides: []const usize) usize {
        var idx: usize = 0;
        var remaining = linear_idx;
        
        const ndim = output_shape.len;
        var i = ndim;
        while (i > 0) {
            i -= 1;
            const coord = remaining % output_shape[i];
            remaining /= output_shape[i];
            
            const input_coord = if (i < input_shape.len and input_shape[i] > 1) coord % input_shape[i] else 0;
            if (i < input_strides.len) {
                idx += input_coord * input_strides[i];
            }
        }
        
        return idx;
    }
});

pub const Div = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Div",
            .version = "1.0.0",
            .description = "Element-wise division of two tensors with broadcasting support",
            .domain = "ai.onnx",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = true,
            .type_constraints = &[_]OperatorInterface.TypeConstraint{
                OperatorInterface.TypeConstraint{
                    .name = "T",
                    .allowed_types = &[_]Tensor.DataType{ .f32, .f16 },
                    .description = "Constrain input and output types to floating point tensors",
                },
            },
        };
    }

    pub fn validate(
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void {
        _ = attributes;

        if (input_shapes.len != 2) {
            return FrameworkError.InvalidInput;
        }

        if (input_types[0] != input_types[1]) {
            return FrameworkError.DataTypeMismatch;
        }

        if (!Self.checkBroadcastCompatibility(input_shapes[0], input_shapes[1])) {
            return FrameworkError.ShapeMismatch;
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len != 2) {
            return FrameworkError.InvalidInput;
        }

        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try Self.calculateBroadcastShape(input_shapes[0], input_shapes[1], allocator);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = attributes;

        if (inputs.len != 2 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const a = &inputs[0];
        const b = &inputs[1];
        const output = &outputs[0];

        if (a.dtype != b.dtype or a.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        switch (a.dtype) {
            .f32 => try divF32(a, b, output, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn divF32(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        const a_data = a.getData(f32);
        const b_data = b.getData(f32);
        const output_data = output.getMutableData(f32);

        if (framework.utils.shapesEqual(a.shape, b.shape) and framework.utils.shapesEqual(a.shape, output.shape)) {
            for (0..output_data.len) |i| {
                if (b_data[i] == 0.0) {
                    return FrameworkError.ExecutionFailed; // Division by zero
                }
                output_data[i] = a_data[i] / b_data[i];
            }
        } else {
            const output_elements = framework.utils.calculateTotalElements(output.shape);
            for (0..output_elements) |i| {
                const a_idx = calculateBroadcastIndex(i, output.shape, a.shape, a.strides);
                const b_idx = calculateBroadcastIndex(i, output.shape, b.shape, b.strides);
                if (b_data[b_idx] == 0.0) {
                    return FrameworkError.ExecutionFailed;
                }
                output_data[i] = a_data[a_idx] / b_data[b_idx];
            }
        }
    }

    fn calculateBroadcastIndex(linear_idx: usize, output_shape: []const usize, input_shape: []const usize, input_strides: []const usize) usize {
        var idx: usize = 0;
        var remaining = linear_idx;
        
        const ndim = output_shape.len;
        var i = ndim;
        while (i > 0) {
            i -= 1;
            const coord = remaining % output_shape[i];
            remaining /= output_shape[i];
            
            const input_coord = if (i < input_shape.len and input_shape[i] > 1) coord % input_shape[i] else 0;
            if (i < input_strides.len) {
                idx += input_coord * input_strides[i];
            }
        }
        
        return idx;
    }
});

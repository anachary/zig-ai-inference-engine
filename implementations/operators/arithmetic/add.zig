const std = @import("std");
const framework = @import("../../../framework/lib.zig");

// Import existing implementation to preserve all hard work
const existing_arithmetic = @import("../../../projects/zig-inference-engine/src/operators/arithmetic.zig");

const Tensor = framework.Tensor;
const Attributes = framework.Attributes;
const ExecutionContext = framework.ExecutionContext;
const FrameworkError = framework.FrameworkError;
const OperatorInterface = framework.OperatorInterface;
const BaseOperator = framework.BaseOperator;

/// Add operator implementation using the new framework
pub const Add = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Add",
            .version = "1.0.0",
            .description = "Element-wise addition of two tensors with broadcasting support",
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

        // Check input count
        if (input_shapes.len != 2) {
            return FrameworkError.InvalidInput;
        }

        // Check data types match
        if (input_types[0] != input_types[1]) {
            return FrameworkError.DataTypeMismatch;
        }

        // Check broadcasting compatibility
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

        // Validate tensor compatibility
        if (a.dtype != b.dtype or a.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        // Dispatch to type-specific implementation
        switch (a.dtype) {
            .f32 => try computeF32(a, b, output, context),
            .f16 => try computeF16(a, b, output, context),
            .i32 => try computeI32(a, b, output, context),
            .i16 => try computeI16(a, b, output, context),
            .i8 => try computeI8(a, b, output, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    pub fn optimize(
        inputs: []const Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!OperatorInterface.OptimizationHint {
        _ = attributes;
        _ = context;

        const hint = OperatorInterface.OptimizationHint{
            .can_fuse = true,
            .preferred_memory_layout = .row_major,
            .vectorization_factor = getVectorizationFactor(inputs[0].dtype),
            .parallelization_strategy = .data_parallel,
            .memory_access_pattern = .sequential,
        };

        return hint;
    }

    // Type-specific compute implementations
    fn computeF32(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        const a_data = a.getData(f32);
        const b_data = b.getData(f32);
        const output_data = output.getMutableData(f32);

        if (Self.shapesEqual(a.shape, b.shape) and Self.shapesEqual(a.shape, output.shape)) {
            // Simple element-wise addition
            try addElementwise(f32, a_data, b_data, output_data, context);
        } else {
            // Broadcasting addition
            try addWithBroadcasting(f32, a, b, output, context);
        }
    }

    fn computeF16(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        // Similar implementation for f16
        _ = a;
        _ = b;
        _ = output;
        _ = context;
        // TODO: Implement f16 support
        return FrameworkError.UnsupportedOperation;
    }

    fn computeI32(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        const a_data = a.getData(i32);
        const b_data = b.getData(i32);
        const output_data = output.getMutableData(i32);

        if (Self.shapesEqual(a.shape, b.shape) and Self.shapesEqual(a.shape, output.shape)) {
            try addElementwise(i32, a_data, b_data, output_data, context);
        } else {
            try addWithBroadcasting(i32, a, b, output, context);
        }
    }

    fn computeI16(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        const a_data = a.getData(i16);
        const b_data = b.getData(i16);
        const output_data = output.getMutableData(i16);

        if (Self.shapesEqual(a.shape, b.shape) and Self.shapesEqual(a.shape, output.shape)) {
            try addElementwise(i16, a_data, b_data, output_data, context);
        } else {
            try addWithBroadcasting(i16, a, b, output, context);
        }
    }

    fn computeI8(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        const a_data = a.getData(i8);
        const b_data = b.getData(i8);
        const output_data = output.getMutableData(i8);

        if (Self.shapesEqual(a.shape, b.shape) and Self.shapesEqual(a.shape, output.shape)) {
            try addElementwise(i8, a_data, b_data, output_data, context);
        } else {
            try addWithBroadcasting(i8, a, b, output, context);
        }
    }

    // Helper functions
    fn addElementwise(comptime T: type, a_data: []const T, b_data: []const T, output_data: []T, context: *ExecutionContext) !void {
        _ = context;
        
        if (a_data.len != b_data.len or a_data.len != output_data.len) {
            return FrameworkError.ShapeMismatch;
        }

        // Vectorized addition when possible
        const simd_width = getSimdWidth(T);
        const vectorized_len = (output_data.len / simd_width) * simd_width;

        // SIMD loop
        var i: usize = 0;
        while (i < vectorized_len) : (i += simd_width) {
            // TODO: Implement SIMD operations
            var j: usize = 0;
            while (j < simd_width and i + j < output_data.len) : (j += 1) {
                output_data[i + j] = a_data[i + j] + b_data[i + j];
            }
        }

        // Remainder loop
        while (i < output_data.len) : (i += 1) {
            output_data[i] = a_data[i] + b_data[i];
        }
    }

    fn addWithBroadcasting(comptime T: type, a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const a_data = a.getData(T);
        const b_data = b.getData(T);
        const output_data = output.getMutableData(T);

        // Implement broadcasting logic
        const output_elements = framework.utils.calculateTotalElements(output.shape);
        
        for (0..output_elements) |i| {
            const a_idx = calculateBroadcastIndex(i, output.shape, a.shape, a.strides);
            const b_idx = calculateBroadcastIndex(i, output.shape, b.shape, b.strides);
            
            output_data[i] = a_data[a_idx] + b_data[b_idx];
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
            
            // Handle broadcasting
            const input_coord = if (i < input_shape.len and input_shape[i] > 1) coord % input_shape[i] else 0;
            if (i < input_strides.len) {
                idx += input_coord * input_strides[i];
            }
        }
        
        return idx;
    }

    fn getVectorizationFactor(dtype: Tensor.DataType) ?u32 {
        return switch (dtype) {
            .f32, .i32 => 8,  // 256-bit SIMD / 32-bit = 8 elements
            .f16, .i16 => 16, // 256-bit SIMD / 16-bit = 16 elements
            .i8 => 32,        // 256-bit SIMD / 8-bit = 32 elements
            else => null,
        };
    }

    fn getSimdWidth(comptime T: type) usize {
        return switch (T) {
            f32, i32 => 8,
            f16, i16 => 16,
            i8 => 32,
            else => 1,
        };
    }
});

// Tests for the Add operator
test "Add operator metadata" {
    const metadata = Add.getMetadata();
    try std.testing.expectEqualStrings("Add", metadata.name);
    try std.testing.expect(metadata.min_inputs == 2);
    try std.testing.expect(metadata.max_inputs == 2);
    try std.testing.expect(metadata.supports_broadcasting);
}

test "Add operator validation" {
    const allocator = std.testing.allocator;
    
    const shape1 = [_]usize{ 2, 3 };
    const shape2 = [_]usize{ 2, 3 };
    const input_shapes = [_][]const usize{ &shape1, &shape2 };
    const input_types = [_]Tensor.DataType{ .f32, .f32 };
    
    var attrs = framework.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    try Add.validate(&input_shapes, &input_types, &attrs);
}

test "Add operator shape inference" {
    const allocator = std.testing.allocator;
    
    const shape1 = [_]usize{ 2, 3 };
    const shape2 = [_]usize{ 2, 3 };
    const input_shapes = [_][]const usize{ &shape1, &shape2 };
    
    var attrs = framework.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    const output_shapes = try Add.inferShapes(&input_shapes, &attrs, allocator);
    defer {
        for (output_shapes) |shape| {
            allocator.free(shape);
        }
        allocator.free(output_shapes);
    }
    
    try std.testing.expect(output_shapes.len == 1);
    try std.testing.expect(output_shapes[0].len == 2);
    try std.testing.expect(output_shapes[0][0] == 2);
    try std.testing.expect(output_shapes[0][1] == 3);
}

test "Add operator compute" {
    const allocator = std.testing.allocator;
    
    // Create input tensors
    const shape = [_]usize{ 2, 2 };
    var tensor_a = try framework.utils.createTensor(allocator, &shape, .f32);
    defer tensor_a.deinit();
    var tensor_b = try framework.utils.createTensor(allocator, &shape, .f32);
    defer tensor_b.deinit();
    var output = try framework.utils.createTensor(allocator, &shape, .f32);
    defer output.deinit();
    
    // Set input data
    const data_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const data_b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    try framework.utils.setTensorData(&tensor_a, f32, &data_a);
    try framework.utils.setTensorData(&tensor_b, f32, &data_b);
    
    // Execute operator
    const inputs = [_]Tensor{ tensor_a, tensor_b };
    var outputs = [_]Tensor{output};
    
    var attrs = framework.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    var context = framework.utils.createExecutionContext(allocator);
    
    try Add.compute(&inputs, &outputs, &attrs, &context);
    
    // Verify results
    const expected = [_]f32{ 6.0, 8.0, 10.0, 12.0 };
    const result_data = framework.utils.getTensorData(&output, f32);
    try std.testing.expectEqualSlices(f32, &expected, result_data);
}

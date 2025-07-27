const std = @import("std");
const framework = @import("../../../framework/lib.zig");

const Tensor = framework.Tensor;
const Attributes = framework.Attributes;
const ExecutionContext = framework.ExecutionContext;
const FrameworkError = framework.FrameworkError;
const OperatorInterface = framework.OperatorInterface;
const BaseOperator = framework.BaseOperator;

/// ReLU activation function implementation
pub const ReLU = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Relu",
            .version = "1.0.0",
            .description = "Rectified Linear Unit activation function: f(x) = max(0, x)",
            .domain = "ai.onnx",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
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

        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        // ReLU supports floating point types
        switch (input_types[0]) {
            .f32, .f16 => {},
            else => return FrameworkError.DataTypeMismatch,
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, input_shapes[0]);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = attributes;

        if (inputs.len != 1 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        if (input.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        if (!framework.utils.shapesEqual(input.shape, output.shape)) {
            return FrameworkError.ShapeMismatch;
        }

        switch (input.dtype) {
            .f32 => try computeF32(input, output, context),
            .f16 => try computeF16(input, output, context),
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

    fn computeF32(input: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const input_data = input.getData(f32);
        const output_data = output.getMutableData(f32);

        if (input_data.len != output_data.len) {
            return FrameworkError.ShapeMismatch;
        }

        // Vectorized ReLU implementation
        const simd_width = 8; // AVX2 can process 8 f32 values at once
        const vectorized_len = (input_data.len / simd_width) * simd_width;

        // SIMD loop
        var i: usize = 0;
        while (i < vectorized_len) : (i += simd_width) {
            // TODO: Implement SIMD ReLU
            var j: usize = 0;
            while (j < simd_width and i + j < input_data.len) : (j += 1) {
                output_data[i + j] = @max(0.0, input_data[i + j]);
            }
        }

        // Remainder loop
        while (i < input_data.len) : (i += 1) {
            output_data[i] = @max(0.0, input_data[i]);
        }
    }

    fn computeF16(input: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = input;
        _ = output;
        _ = context;
        // TODO: Implement f16 ReLU
        return FrameworkError.UnsupportedOperation;
    }

    fn getVectorizationFactor(dtype: Tensor.DataType) ?u32 {
        return switch (dtype) {
            .f32 => 8,  // 256-bit SIMD / 32-bit = 8 elements
            .f16 => 16, // 256-bit SIMD / 16-bit = 16 elements
            else => null,
        };
    }
});

/// Sigmoid activation function implementation
pub const Sigmoid = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Sigmoid",
            .version = "1.0.0",
            .description = "Sigmoid activation function: f(x) = 1 / (1 + exp(-x))",
            .domain = "ai.onnx",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
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

        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        switch (input_types[0]) {
            .f32, .f16 => {},
            else => return FrameworkError.DataTypeMismatch,
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, input_shapes[0]);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = attributes;

        if (inputs.len != 1 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        if (input.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        if (!framework.utils.shapesEqual(input.shape, output.shape)) {
            return FrameworkError.ShapeMismatch;
        }

        switch (input.dtype) {
            .f32 => try sigmoidF32(input, output, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn sigmoidF32(input: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const input_data = input.getData(f32);
        const output_data = output.getMutableData(f32);

        if (input_data.len != output_data.len) {
            return FrameworkError.ShapeMismatch;
        }

        for (0..input_data.len) |i| {
            output_data[i] = 1.0 / (1.0 + @exp(-input_data[i]));
        }
    }
});

/// Tanh activation function implementation
pub const Tanh = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Tanh",
            .version = "1.0.0",
            .description = "Hyperbolic tangent activation function: f(x) = tanh(x)",
            .domain = "ai.onnx",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
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

        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        switch (input_types[0]) {
            .f32, .f16 => {},
            else => return FrameworkError.DataTypeMismatch,
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, input_shapes[0]);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = attributes;

        if (inputs.len != 1 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        if (input.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        if (!framework.utils.shapesEqual(input.shape, output.shape)) {
            return FrameworkError.ShapeMismatch;
        }

        switch (input.dtype) {
            .f32 => try tanhF32(input, output, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn tanhF32(input: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const input_data = input.getData(f32);
        const output_data = output.getMutableData(f32);

        if (input_data.len != output_data.len) {
            return FrameworkError.ShapeMismatch;
        }

        for (0..input_data.len) |i| {
            output_data[i] = std.math.tanh(input_data[i]);
        }
    }
});

/// GELU activation function implementation
pub const GELU = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Gelu",
            .version = "1.0.0",
            .description = "Gaussian Error Linear Unit activation function",
            .domain = "ai.onnx",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
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

        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        switch (input_types[0]) {
            .f32, .f16 => {},
            else => return FrameworkError.DataTypeMismatch,
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, input_shapes[0]);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = attributes;

        if (inputs.len != 1 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        if (input.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        if (!framework.utils.shapesEqual(input.shape, output.shape)) {
            return FrameworkError.ShapeMismatch;
        }

        switch (input.dtype) {
            .f32 => try geluF32(input, output, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn geluF32(input: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const input_data = input.getData(f32);
        const output_data = output.getMutableData(f32);

        if (input_data.len != output_data.len) {
            return FrameworkError.ShapeMismatch;
        }

        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const sqrt_2_over_pi = std.math.sqrt(2.0 / std.math.pi);
        
        for (0..input_data.len) |i| {
            const x = input_data[i];
            const x_cubed = x * x * x;
            const inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
            output_data[i] = 0.5 * x * (1.0 + std.math.tanh(inner));
        }
    }
});

// Tests
test "ReLU operator" {
    const allocator = std.testing.allocator;
    
    const shape = [_]usize{ 2, 2 };
    var input = try framework.utils.createTensor(allocator, &shape, .f32);
    defer input.deinit();
    var output = try framework.utils.createTensor(allocator, &shape, .f32);
    defer output.deinit();
    
    const input_data = [_]f32{ -1.0, 2.0, -3.0, 4.0 };
    try framework.utils.setTensorData(&input, f32, &input_data);
    
    const inputs = [_]Tensor{input};
    var outputs = [_]Tensor{output};
    
    var attrs = framework.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    var context = framework.utils.createExecutionContext(allocator);
    
    try ReLU.compute(&inputs, &outputs, &attrs, &context);
    
    const expected = [_]f32{ 0.0, 2.0, 0.0, 4.0 };
    const result_data = framework.utils.getTensorData(&output, f32);
    try std.testing.expectEqualSlices(f32, &expected, result_data);
}

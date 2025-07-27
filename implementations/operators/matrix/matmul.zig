const std = @import("std");
const framework = @import("../../../framework/lib.zig");

const Tensor = framework.Tensor;
const Attributes = framework.Attributes;
const ExecutionContext = framework.ExecutionContext;
const FrameworkError = framework.FrameworkError;
const OperatorInterface = framework.OperatorInterface;
const BaseOperator = framework.BaseOperator;

/// Matrix multiplication operator implementation
pub const MatMul = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "MatMul",
            .version = "1.0.0",
            .description = "Matrix multiplication with broadcasting support for batch dimensions",
            .domain = "ai.onnx",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
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

        const a_shape = input_shapes[0];
        const b_shape = input_shapes[1];

        // Check minimum dimensions
        if (a_shape.len < 2 or b_shape.len < 2) {
            return FrameworkError.ShapeMismatch;
        }

        // Check matrix multiplication compatibility
        const a_cols = a_shape[a_shape.len - 1];
        const b_rows = b_shape[b_shape.len - 2];
        
        if (a_cols != b_rows) {
            return FrameworkError.ShapeMismatch;
        }

        // Check batch dimension compatibility
        if (a_shape.len > 2 or b_shape.len > 2) {
            try validateBatchDimensions(a_shape, b_shape);
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

        const a_shape = input_shapes[0];
        const b_shape = input_shapes[1];

        const output_shapes = try allocator.alloc([]usize, 1);
        
        if (a_shape.len == 2 and b_shape.len == 2) {
            // Simple 2D matrix multiplication
            output_shapes[0] = try allocator.alloc(usize, 2);
            output_shapes[0][0] = a_shape[0];
            output_shapes[0][1] = b_shape[1];
        } else {
            // Batch matrix multiplication
            const max_dims = @max(a_shape.len, b_shape.len);
            output_shapes[0] = try allocator.alloc(usize, max_dims);
            
            // Broadcast batch dimensions
            for (0..max_dims - 2) |i| {
                const a_dim = if (i < a_shape.len - 2) a_shape[i] else 1;
                const b_dim = if (i < b_shape.len - 2) b_shape[i] else 1;
                output_shapes[0][i] = @max(a_dim, b_dim);
            }
            
            // Matrix dimensions
            output_shapes[0][max_dims - 2] = a_shape[a_shape.len - 2];
            output_shapes[0][max_dims - 1] = b_shape[b_shape.len - 1];
        }
        
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
            .f16 => try computeF16(a, b, output, context),
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

        const a = &inputs[0];
        const b = &inputs[1];
        
        // Determine optimal strategy based on matrix sizes
        const a_rows = a.shape[a.shape.len - 2];
        const a_cols = a.shape[a.shape.len - 1];
        const b_cols = b.shape[b.shape.len - 1];
        
        const hint = OperatorInterface.OptimizationHint{
            .can_fuse = false, // MatMul is typically not fused
            .preferred_memory_layout = if (a_cols > 1024) .blocked else .row_major,
            .vectorization_factor = 8, // AVX2 for f32
            .parallelization_strategy = if (a_rows > 64) .data_parallel else .none,
            .memory_access_pattern = .strided,
        };

        return hint;
    }

    fn computeF32(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const a_data = a.getData(f32);
        const b_data = b.getData(f32);
        const output_data = output.getMutableData(f32);

        if (a.shape.len == 2 and b.shape.len == 2) {
            // Simple 2D matrix multiplication
            try matmul2DF32(a_data, b_data, output_data, a.shape, b.shape, output.shape);
        } else {
            // Batch matrix multiplication
            try batchMatmulF32(a, b, output);
        }
    }

    fn computeF16(a: *const Tensor, b: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = a;
        _ = b;
        _ = output;
        _ = context;
        // TODO: Implement f16 matrix multiplication
        return FrameworkError.UnsupportedOperation;
    }

    fn matmul2DF32(
        a_data: []const f32,
        b_data: []const f32,
        output_data: []f32,
        a_shape: []const usize,
        b_shape: []const usize,
        output_shape: []const usize,
    ) !void {
        const M = a_shape[0];
        const K = a_shape[1];
        const N = b_shape[1];

        if (output_shape[0] != M or output_shape[1] != N) {
            return FrameworkError.ShapeMismatch;
        }

        // Initialize output to zero
        @memset(output_data, 0.0);

        // Basic matrix multiplication: C[i,j] = sum(A[i,k] * B[k,j])
        for (0..M) |i| {
            for (0..N) |j| {
                var sum: f32 = 0.0;
                for (0..K) |k| {
                    sum += a_data[i * K + k] * b_data[k * N + j];
                }
                output_data[i * N + j] = sum;
            }
        }
    }

    fn batchMatmulF32(a: *const Tensor, b: *const Tensor, output: *const Tensor) !void {
        const a_data = a.getData(f32);
        const b_data = b.getData(f32);
        const output_data = output.getMutableData(f32);

        // Calculate batch size and matrix dimensions
        const batch_size = calculateBatchSize(output.shape);
        const M = a.shape[a.shape.len - 2];
        const K = a.shape[a.shape.len - 1];
        const N = b.shape[b.shape.len - 1];

        // Initialize output to zero
        @memset(output_data, 0.0);

        for (0..batch_size) |batch_idx| {
            const a_offset = calculateBatchOffset(batch_idx, a.shape, M * K);
            const b_offset = calculateBatchOffset(batch_idx, b.shape, K * N);
            const output_offset = batch_idx * M * N;

            // Perform matrix multiplication for this batch
            for (0..M) |i| {
                for (0..N) |j| {
                    var sum: f32 = 0.0;
                    for (0..K) |k| {
                        const a_idx = a_offset + i * K + k;
                        const b_idx = b_offset + k * N + j;
                        sum += a_data[a_idx] * b_data[b_idx];
                    }
                    output_data[output_offset + i * N + j] = sum;
                }
            }
        }
    }

    fn validateBatchDimensions(a_shape: []const usize, b_shape: []const usize) !void {
        const a_batch_dims = a_shape.len - 2;
        const b_batch_dims = b_shape.len - 2;
        const max_batch_dims = @max(a_batch_dims, b_batch_dims);

        for (0..max_batch_dims) |i| {
            const a_dim = if (i < a_batch_dims) a_shape[i] else 1;
            const b_dim = if (i < b_batch_dims) b_shape[i] else 1;
            
            if (a_dim != b_dim and a_dim != 1 and b_dim != 1) {
                return FrameworkError.ShapeMismatch;
            }
        }
    }

    fn calculateBatchSize(shape: []const usize) usize {
        var batch_size: usize = 1;
        for (0..shape.len - 2) |i| {
            batch_size *= shape[i];
        }
        return batch_size;
    }

    fn calculateBatchOffset(batch_idx: usize, shape: []const usize, matrix_size: usize) usize {
        if (shape.len <= 2) {
            return 0; // No batch dimensions
        }

        var offset: usize = 0;
        var remaining_idx = batch_idx;
        
        // Calculate offset in batch dimensions
        var stride = matrix_size;
        var i = shape.len - 3;
        while (true) {
            const dim_size = shape[i];
            const coord = remaining_idx % dim_size;
            offset += coord * stride;
            stride *= dim_size;
            remaining_idx /= dim_size;
            
            if (i == 0) break;
            i -= 1;
        }
        
        return offset;
    }
});

/// Transpose operator implementation
pub const Transpose = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Transpose",
            .version = "1.0.0",
            .description = "Transpose the input tensor similar to numpy.transpose",
            .domain = "ai.onnx",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
            .supports_broadcasting = false,
            .type_constraints = &[_]OperatorInterface.TypeConstraint{
                OperatorInterface.TypeConstraint{
                    .name = "T",
                    .allowed_types = &[_]Tensor.DataType{ .f32, .f16, .i32, .i16, .i8, .u8 },
                    .description = "Constrain input and output types to all tensor types",
                },
            },
        };
    }

    pub fn validate(
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void {
        _ = input_types;

        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        // Validate perm attribute if provided
        if (attributes.get("perm")) |perm_attr| {
            switch (perm_attr) {
                .ints => |perm| {
                    if (perm.len != input_shapes[0].len) {
                        return FrameworkError.ValidationFailed;
                    }
                    
                    // Check that perm is a valid permutation
                    var seen = try std.testing.allocator.alloc(bool, perm.len);
                    defer std.testing.allocator.free(seen);
                    @memset(seen, false);
                    
                    for (perm) |axis| {
                        if (axis < 0 or axis >= perm.len) {
                            return FrameworkError.ValidationFailed;
                        }
                        if (seen[@intCast(axis)]) {
                            return FrameworkError.ValidationFailed; // Duplicate axis
                        }
                        seen[@intCast(axis)] = true;
                    }
                },
                else => return FrameworkError.ValidationFailed,
            }
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        if (input_shapes.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const input_shape = input_shapes[0];
        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.alloc(usize, input_shape.len);

        if (attributes.get("perm")) |perm_attr| {
            switch (perm_attr) {
                .ints => |perm| {
                    for (perm, 0..) |axis, i| {
                        output_shapes[0][i] = input_shape[@intCast(axis)];
                    }
                },
                else => return FrameworkError.ValidationFailed,
            }
        } else {
            // Default: reverse all dimensions
            for (0..input_shape.len) |i| {
                output_shapes[0][i] = input_shape[input_shape.len - 1 - i];
            }
        }
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        _ = context;

        if (inputs.len != 1 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        if (input.dtype != output.dtype) {
            return FrameworkError.DataTypeMismatch;
        }

        // Get permutation
        var perm: []const i64 = undefined;
        var default_perm: []i64 = undefined;
        var owns_perm = false;
        
        if (attributes.get("perm")) |perm_attr| {
            switch (perm_attr) {
                .ints => |p| perm = p,
                else => return FrameworkError.ValidationFailed,
            }
        } else {
            // Create default permutation (reverse dimensions)
            default_perm = try context.allocator.alloc(i64, input.shape.len);
            owns_perm = true;
            for (0..input.shape.len) |i| {
                default_perm[i] = @intCast(input.shape.len - 1 - i);
            }
            perm = default_perm;
        }
        defer if (owns_perm) context.allocator.free(default_perm);

        switch (input.dtype) {
            .f32 => try transposeF32(input, output, perm),
            .i32 => try transposeI32(input, output, perm),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn transposeF32(input: *const Tensor, output: *const Tensor, perm: []const i64) !void {
        const input_data = input.getData(f32);
        const output_data = output.getMutableData(f32);
        
        const total_elements = framework.utils.calculateTotalElements(input.shape);
        
        for (0..total_elements) |linear_idx| {
            const input_coords = try linearToCoords(linear_idx, input.shape, std.testing.allocator);
            defer std.testing.allocator.free(input_coords);
            
            var output_coords = try std.testing.allocator.alloc(usize, input_coords.len);
            defer std.testing.allocator.free(output_coords);
            
            for (perm, 0..) |axis, i| {
                output_coords[i] = input_coords[@intCast(axis)];
            }
            
            const output_idx = coordsToLinear(output_coords, output.shape);
            output_data[output_idx] = input_data[linear_idx];
        }
    }

    fn transposeI32(input: *const Tensor, output: *const Tensor, perm: []const i64) !void {
        const input_data = input.getData(i32);
        const output_data = output.getMutableData(i32);
        
        const total_elements = framework.utils.calculateTotalElements(input.shape);
        
        for (0..total_elements) |linear_idx| {
            const input_coords = try linearToCoords(linear_idx, input.shape, std.testing.allocator);
            defer std.testing.allocator.free(input_coords);
            
            var output_coords = try std.testing.allocator.alloc(usize, input_coords.len);
            defer std.testing.allocator.free(output_coords);
            
            for (perm, 0..) |axis, i| {
                output_coords[i] = input_coords[@intCast(axis)];
            }
            
            const output_idx = coordsToLinear(output_coords, output.shape);
            output_data[output_idx] = input_data[linear_idx];
        }
    }

    fn linearToCoords(linear_idx: usize, shape: []const usize, allocator: std.mem.Allocator) ![]usize {
        const coords = try allocator.alloc(usize, shape.len);
        var remaining = linear_idx;
        
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            coords[i] = remaining % shape[i];
            remaining /= shape[i];
        }
        
        return coords;
    }

    fn coordsToLinear(coords: []const usize, shape: []const usize) usize {
        var linear_idx: usize = 0;
        var stride: usize = 1;
        
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            linear_idx += coords[i] * stride;
            stride *= shape[i];
        }
        
        return linear_idx;
    }
});

// Tests
test "MatMul operator 2D" {
    const allocator = std.testing.allocator;
    
    // Create 2x3 and 3x2 matrices
    const a_shape = [_]usize{ 2, 3 };
    const b_shape = [_]usize{ 3, 2 };
    const output_shape = [_]usize{ 2, 2 };
    
    var tensor_a = try framework.utils.createTensor(allocator, &a_shape, .f32);
    defer tensor_a.deinit();
    var tensor_b = try framework.utils.createTensor(allocator, &b_shape, .f32);
    defer tensor_b.deinit();
    var output = try framework.utils.createTensor(allocator, &output_shape, .f32);
    defer output.deinit();
    
    // Set input data: A = [[1,2,3], [4,5,6]], B = [[1,2], [3,4], [5,6]]
    const data_a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const data_b = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try framework.utils.setTensorData(&tensor_a, f32, &data_a);
    try framework.utils.setTensorData(&tensor_b, f32, &data_b);
    
    const inputs = [_]Tensor{ tensor_a, tensor_b };
    var outputs = [_]Tensor{output};
    
    var attrs = framework.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    var context = framework.utils.createExecutionContext(allocator);
    
    try MatMul.compute(&inputs, &outputs, &attrs, &context);
    
    // Expected result: [[22,28], [49,64]]
    const expected = [_]f32{ 22.0, 28.0, 49.0, 64.0 };
    const result_data = framework.utils.getTensorData(&output, f32);
    try std.testing.expectEqualSlices(f32, &expected, result_data);
}

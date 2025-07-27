const std = @import("std");
const framework = @import("../../../framework/lib.zig");

const Tensor = framework.Tensor;
const Attributes = framework.Attributes;
const ExecutionContext = framework.ExecutionContext;
const FrameworkError = framework.FrameworkError;
const OperatorInterface = framework.OperatorInterface;
const BaseOperator = framework.BaseOperator;

/// Common transformer components and utilities

/// Layer Normalization operator for transformers
pub const LayerNorm = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "LayerNormalization",
            .version = "1.0.0",
            .description = "Layer normalization for transformer models",
            .domain = "ai.onnx",
            .min_inputs = 1,
            .max_inputs = 3, // input, scale, bias
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
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

        if (input_shapes.len < 1 or input_shapes.len > 3) {
            return FrameworkError.InvalidInput;
        }

        // All inputs should have the same data type
        for (input_types[1..]) |dtype| {
            if (dtype != input_types[0]) {
                return FrameworkError.DataTypeMismatch;
            }
        }

        // Validate scale and bias shapes if provided
        if (input_shapes.len >= 2) {
            const input_shape = input_shapes[0];
            const scale_shape = input_shapes[1];
            
            // Scale should match the last dimension of input
            if (scale_shape.len != 1 or scale_shape[0] != input_shape[input_shape.len - 1]) {
                return FrameworkError.ShapeMismatch;
            }
        }

        if (input_shapes.len == 3) {
            const scale_shape = input_shapes[1];
            const bias_shape = input_shapes[2];
            
            // Bias should have the same shape as scale
            if (!framework.utils.shapesEqual(scale_shape, bias_shape)) {
                return FrameworkError.ShapeMismatch;
            }
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len < 1) {
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
        if (inputs.len < 1 or inputs.len > 3 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const input = &inputs[0];
        const output = &outputs[0];
        
        const scale = if (inputs.len >= 2) &inputs[1] else null;
        const bias = if (inputs.len == 3) &inputs[2] else null;

        const epsilon = attributes.getFloat("epsilon", 1e-5);
        const axis = attributes.getInt("axis", -1);

        switch (input.dtype) {
            .f32 => try layerNormF32(input, output, scale, bias, epsilon, axis, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn layerNormF32(
        input: *const Tensor,
        output: *const Tensor,
        scale: ?*const Tensor,
        bias: ?*const Tensor,
        epsilon: f64,
        axis: i64,
        context: *ExecutionContext,
    ) !void {
        _ = context;
        
        const input_data = input.getData(f32);
        const output_data = output.getMutableData(f32);
        
        const scale_data = if (scale) |s| s.getData(f32) else null;
        const bias_data = if (bias) |b| b.getData(f32) else null;

        // Normalize axis
        const norm_axis = if (axis < 0) @as(usize, @intCast(@as(i64, @intCast(input.shape.len)) + axis)) else @as(usize, @intCast(axis));
        
        // Calculate dimensions
        const outer_size = calculateOuterSize(input.shape, norm_axis);
        const inner_size = input.shape[norm_axis];
        
        for (0..outer_size) |outer_idx| {
            const offset = outer_idx * inner_size;
            
            // Calculate mean
            var sum: f32 = 0.0;
            for (0..inner_size) |i| {
                sum += input_data[offset + i];
            }
            const mean = sum / @as(f32, @floatFromInt(inner_size));
            
            // Calculate variance
            var var_sum: f32 = 0.0;
            for (0..inner_size) |i| {
                const diff = input_data[offset + i] - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f32, @floatFromInt(inner_size));
            const std_dev = @sqrt(variance + @as(f32, @floatCast(epsilon)));
            
            // Normalize and apply scale/bias
            for (0..inner_size) |i| {
                var normalized = (input_data[offset + i] - mean) / std_dev;
                
                if (scale_data) |s| {
                    normalized *= s[i];
                }
                
                if (bias_data) |b| {
                    normalized += b[i];
                }
                
                output_data[offset + i] = normalized;
            }
        }
    }

    fn calculateOuterSize(shape: []const usize, axis: usize) usize {
        var size: usize = 1;
        for (0..axis) |i| {
            size *= shape[i];
        }
        for (axis + 1..shape.len) |i| {
            size *= shape[i];
        }
        return size;
    }
});

/// RMS Normalization (used in LLaMA and other modern models)
pub const RMSNorm = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "RMSNorm",
            .version = "1.0.0",
            .description = "Root Mean Square normalization for transformer models",
            .domain = "ai.onnx",
            .min_inputs = 1,
            .max_inputs = 2, // input, scale
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
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

        if (input_shapes.len < 1 or input_shapes.len > 2) {
            return FrameworkError.InvalidInput;
        }

        if (input_shapes.len == 2) {
            for (input_types[1..]) |dtype| {
                if (dtype != input_types[0]) {
                    return FrameworkError.DataTypeMismatch;
                }
            }

            const input_shape = input_shapes[0];
            const scale_shape = input_shapes[1];
            
            if (scale_shape.len != 1 or scale_shape[0] != input_shape[input_shape.len - 1]) {
                return FrameworkError.ShapeMismatch;
            }
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len < 1) {
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
        if (inputs.len < 1 or inputs.len > 2 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const input = &inputs[0];
        const output = &outputs[0];
        const scale = if (inputs.len == 2) &inputs[1] else null;

        const epsilon = attributes.getFloat("epsilon", 1e-6);

        switch (input.dtype) {
            .f32 => try rmsNormF32(input, output, scale, epsilon, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn rmsNormF32(
        input: *const Tensor,
        output: *const Tensor,
        scale: ?*const Tensor,
        epsilon: f64,
        context: *ExecutionContext,
    ) !void {
        _ = context;
        
        const input_data = input.getData(f32);
        const output_data = output.getMutableData(f32);
        const scale_data = if (scale) |s| s.getData(f32) else null;

        const last_dim = input.shape[input.shape.len - 1];
        const outer_size = framework.utils.calculateTotalElements(input.shape) / last_dim;
        
        for (0..outer_size) |outer_idx| {
            const offset = outer_idx * last_dim;
            
            // Calculate RMS
            var sum_squares: f32 = 0.0;
            for (0..last_dim) |i| {
                const val = input_data[offset + i];
                sum_squares += val * val;
            }
            const rms = @sqrt(sum_squares / @as(f32, @floatFromInt(last_dim)) + @as(f32, @floatCast(epsilon)));
            
            // Normalize and apply scale
            for (0..last_dim) |i| {
                var normalized = input_data[offset + i] / rms;
                
                if (scale_data) |s| {
                    normalized *= s[i];
                }
                
                output_data[offset + i] = normalized;
            }
        }
    }
});

/// Embedding lookup operator
pub const Embedding = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "Gather",
            .version = "1.0.0",
            .description = "Embedding lookup using Gather operation",
            .domain = "ai.onnx",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
            .supports_broadcasting = false,
            .type_constraints = &[_]OperatorInterface.TypeConstraint{
                OperatorInterface.TypeConstraint{
                    .name = "T",
                    .allowed_types = &[_]Tensor.DataType{ .f32, .f16 },
                    .description = "Constrain data types to floating point tensors",
                },
                OperatorInterface.TypeConstraint{
                    .name = "Tind",
                    .allowed_types = &[_]Tensor.DataType{ .i32, .i64 },
                    .description = "Constrain indices to integer tensors",
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

        // First input (data) should be floating point
        switch (input_types[0]) {
            .f32, .f16 => {},
            else => return FrameworkError.DataTypeMismatch,
        }

        // Second input (indices) should be integer
        switch (input_types[1]) {
            .i32, .i64 => {},
            else => return FrameworkError.DataTypeMismatch,
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

        const data_shape = input_shapes[0];
        const indices_shape = input_shapes[1];

        const output_shapes = try allocator.alloc([]usize, 1);
        
        // Output shape: indices_shape + data_shape[1:]
        const output_dims = indices_shape.len + data_shape.len - 1;
        output_shapes[0] = try allocator.alloc(usize, output_dims);
        
        // Copy indices dimensions
        for (0..indices_shape.len) |i| {
            output_shapes[0][i] = indices_shape[i];
        }
        
        // Copy data dimensions (skip first dimension which is the vocabulary size)
        for (1..data_shape.len) |i| {
            output_shapes[0][indices_shape.len + i - 1] = data_shape[i];
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

        const data = &inputs[0];
        const indices = &inputs[1];
        const output = &outputs[0];

        switch (data.dtype) {
            .f32 => {
                switch (indices.dtype) {
                    .i32 => try embeddingF32I32(data, indices, output, context),
                    .i64 => try embeddingF32I64(data, indices, output, context),
                    else => return FrameworkError.UnsupportedOperation,
                }
            },
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn embeddingF32I32(data: *const Tensor, indices: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const data_ptr = data.getData(f32);
        const indices_ptr = indices.getData(i32);
        const output_ptr = output.getMutableData(f32);

        const vocab_size = data.shape[0];
        const embedding_dim = data.shape[1];
        const num_indices = framework.utils.calculateTotalElements(indices.shape);

        for (0..num_indices) |i| {
            const idx = indices_ptr[i];
            
            if (idx < 0 or idx >= vocab_size) {
                return FrameworkError.ExecutionFailed; // Index out of bounds
            }

            const src_offset = @as(usize, @intCast(idx)) * embedding_dim;
            const dst_offset = i * embedding_dim;

            for (0..embedding_dim) |j| {
                output_ptr[dst_offset + j] = data_ptr[src_offset + j];
            }
        }
    }

    fn embeddingF32I64(data: *const Tensor, indices: *const Tensor, output: *const Tensor, context: *ExecutionContext) !void {
        _ = context;
        
        const data_ptr = data.getData(f32);
        const indices_ptr = indices.getData(i64);
        const output_ptr = output.getMutableData(f32);

        const vocab_size = data.shape[0];
        const embedding_dim = data.shape[1];
        const num_indices = framework.utils.calculateTotalElements(indices.shape);

        for (0..num_indices) |i| {
            const idx = indices_ptr[i];
            
            if (idx < 0 or idx >= vocab_size) {
                return FrameworkError.ExecutionFailed;
            }

            const src_offset = @as(usize, @intCast(idx)) * embedding_dim;
            const dst_offset = i * embedding_dim;

            for (0..embedding_dim) |j| {
                output_ptr[dst_offset + j] = data_ptr[src_offset + j];
            }
        }
    }
});

/// Rotary Position Embedding (RoPE) for modern transformers
pub const RotaryPositionalEmbedding = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "RotaryPositionalEmbedding",
            .version = "1.0.0",
            .description = "Rotary positional embedding for transformer models",
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

        // Input should be at least 3D: [batch, seq_len, hidden_dim]
        if (input_shapes[0].len < 3) {
            return FrameworkError.ShapeMismatch;
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
        if (inputs.len != 1 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        const base = attributes.getFloat("base", 10000.0);
        const max_seq_len = attributes.getInt("max_seq_len", 2048);

        switch (input.dtype) {
            .f32 => try ropeF32(input, output, base, max_seq_len, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn ropeF32(input: *const Tensor, output: *const Tensor, base: f64, max_seq_len: i64, context: *ExecutionContext) !void {
        _ = context;
        _ = max_seq_len;
        
        const input_data = input.getData(f32);
        const output_data = output.getMutableData(f32);

        // Copy input to output first
        @memcpy(output_data, input_data);

        // Apply RoPE
        const shape = input.shape;
        const batch_size = shape[0];
        const seq_len = shape[1];
        const hidden_dim = shape[2];
        
        const head_dim = hidden_dim; // Assuming single head for simplicity
        const half_dim = head_dim / 2;

        for (0..batch_size) |batch| {
            for (0..seq_len) |pos| {
                for (0..half_dim) |i| {
                    const theta = @as(f32, @floatCast(base)) * -(@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half_dim)));
                    const freq = @pow(f32, @as(f32, @floatCast(base)), theta);
                    const angle = @as(f32, @floatFromInt(pos)) * freq;
                    
                    const cos_val = @cos(angle);
                    const sin_val = @sin(angle);
                    
                    const offset = batch * seq_len * hidden_dim + pos * hidden_dim;
                    const x1 = output_data[offset + i * 2];
                    const x2 = output_data[offset + i * 2 + 1];
                    
                    output_data[offset + i * 2] = x1 * cos_val - x2 * sin_val;
                    output_data[offset + i * 2 + 1] = x1 * sin_val + x2 * cos_val;
                }
            }
        }
    }
});

// Tests
test "LayerNorm operator" {
    const allocator = std.testing.allocator;
    
    const shape = [_]usize{ 2, 4 };
    var input = try framework.utils.createTensor(allocator, &shape, .f32);
    defer input.deinit();
    var output = try framework.utils.createTensor(allocator, &shape, .f32);
    defer output.deinit();
    
    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    try framework.utils.setTensorData(&input, f32, &input_data);
    
    const inputs = [_]Tensor{input};
    var outputs = [_]Tensor{output};
    
    var attrs = framework.utils.createAttributes(allocator);
    defer attrs.deinit();
    try attrs.set("epsilon", framework.Attributes.AttributeValue{ .float = 1e-5 });
    try attrs.set("axis", framework.Attributes.AttributeValue{ .int = -1 });
    
    var context = framework.utils.createExecutionContext(allocator);
    
    try LayerNorm.compute(&inputs, &outputs, &attrs, &context);
    
    // Verify that the output has been normalized (mean ≈ 0, std ≈ 1)
    const result_data = framework.utils.getTensorData(&output, f32);
    
    // Check first row normalization
    var sum: f32 = 0.0;
    for (0..4) |i| {
        sum += result_data[i];
    }
    const mean = sum / 4.0;
    try std.testing.expect(@abs(mean) < 1e-5); // Mean should be close to 0
}

const std = @import("std");
const framework = @import("../../../framework/lib.zig");

const Tensor = framework.Tensor;
const Attributes = framework.Attributes;
const ExecutionContext = framework.ExecutionContext;
const FrameworkError = framework.FrameworkError;
const OperatorInterface = framework.OperatorInterface;
const BaseOperator = framework.BaseOperator;

/// Multi-Head Attention operator for transformer models
pub const MultiHeadAttention = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "MultiHeadAttention",
            .version = "1.0.0",
            .description = "Multi-head attention mechanism for transformer models",
            .domain = "ai.onnx",
            .min_inputs = 3,
            .max_inputs = 5, // query, key, value, [attention_mask], [key_padding_mask]
            .min_outputs = 1,
            .max_outputs = 2, // output, [attention_weights]
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
        if (input_shapes.len < 3 or input_shapes.len > 5) {
            return FrameworkError.InvalidInput;
        }

        // All inputs should have the same data type
        for (input_types[1..]) |dtype| {
            if (dtype != input_types[0]) {
                return FrameworkError.DataTypeMismatch;
            }
        }

        const query_shape = input_shapes[0];
        const key_shape = input_shapes[1];
        const value_shape = input_shapes[2];

        // Validate Q, K, V shapes
        if (query_shape.len != 3 or key_shape.len != 3 or value_shape.len != 3) {
            return FrameworkError.ShapeMismatch;
        }

        // Check dimension compatibility
        const num_heads = attributes.getInt("num_heads", 8);
        const embed_dim = query_shape[2];
        
        if (@mod(embed_dim, num_heads) != 0) {
            return FrameworkError.ValidationFailed;
        }

        // Key and value should have compatible dimensions
        if (key_shape[2] != value_shape[2]) {
            return FrameworkError.ShapeMismatch;
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len < 3) {
            return FrameworkError.InvalidInput;
        }

        const query_shape = input_shapes[0];
        
        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, query_shape);
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        if (inputs.len < 3 or inputs.len > 5 or outputs.len < 1 or outputs.len > 2) {
            return FrameworkError.InvalidInput;
        }

        const query = &inputs[0];
        const key = &inputs[1];
        const value = &inputs[2];
        const output = &outputs[0];

        const num_heads = attributes.getInt("num_heads", 8);
        const dropout_p = attributes.getFloat("dropout_p", 0.0);
        const is_causal = attributes.getInt("is_causal", 0) != 0;

        switch (query.dtype) {
            .f32 => try multiHeadAttentionF32(query, key, value, output, num_heads, dropout_p, is_causal, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn multiHeadAttentionF32(
        query: *const Tensor,
        key: *const Tensor,
        value: *const Tensor,
        output: *const Tensor,
        num_heads: i64,
        dropout_p: f64,
        is_causal: bool,
        context: *ExecutionContext,
    ) !void {
        _ = dropout_p; // TODO: Implement dropout
        
        const query_data = query.getData(f32);
        const key_data = key.getData(f32);
        const value_data = value.getData(f32);
        const output_data = output.getMutableData(f32);

        const batch_size = query.shape[0];
        const seq_len_q = query.shape[1];
        const seq_len_k = key.shape[1];
        const embed_dim = query.shape[2];
        
        const head_dim = embed_dim / @as(usize, @intCast(num_heads));
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Allocate temporary tensors for attention computation
        const attention_scores_size = batch_size * @as(usize, @intCast(num_heads)) * seq_len_q * seq_len_k;
        const attention_scores = try context.allocateMemory(attention_scores_size * @sizeOf(f32));
        defer context.freeMemory(attention_scores);
        const scores_data = std.mem.bytesAsSlice(f32, attention_scores);

        // Initialize output to zero
        @memset(output_data, 0.0);

        for (0..batch_size) |batch| {
            for (0..@as(usize, @intCast(num_heads))) |head| {
                // Calculate attention scores for this head
                try calculateAttentionScores(
                    query_data,
                    key_data,
                    scores_data,
                    batch,
                    head,
                    seq_len_q,
                    seq_len_k,
                    head_dim,
                    embed_dim,
                    @as(usize, @intCast(num_heads)),
                    scale,
                );

                // Apply causal mask if needed
                if (is_causal) {
                    applyCausalMask(scores_data, batch, head, seq_len_q, seq_len_k, @as(usize, @intCast(num_heads)));
                }

                // Apply softmax to attention scores
                try applySoftmax(scores_data, batch, head, seq_len_q, seq_len_k, @as(usize, @intCast(num_heads)));

                // Apply attention to values
                try applyAttentionToValues(
                    scores_data,
                    value_data,
                    output_data,
                    batch,
                    head,
                    seq_len_q,
                    seq_len_k,
                    head_dim,
                    embed_dim,
                    @as(usize, @intCast(num_heads)),
                );
            }
        }
    }

    fn calculateAttentionScores(
        query_data: []const f32,
        key_data: []const f32,
        scores_data: []f32,
        batch: usize,
        head: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        embed_dim: usize,
        num_heads: usize,
        scale: f32,
    ) !void {
        for (0..seq_len_q) |i| {
            for (0..seq_len_k) |j| {
                var score: f32 = 0.0;
                
                for (0..head_dim) |d| {
                    const q_idx = batch * seq_len_q * embed_dim + i * embed_dim + head * head_dim + d;
                    const k_idx = batch * seq_len_k * embed_dim + j * embed_dim + head * head_dim + d;
                    
                    score += query_data[q_idx] * key_data[k_idx];
                }
                
                const score_idx = batch * num_heads * seq_len_q * seq_len_k + 
                                 head * seq_len_q * seq_len_k + 
                                 i * seq_len_k + j;
                scores_data[score_idx] = score * scale;
            }
        }
    }

    fn applyCausalMask(
        scores_data: []f32,
        batch: usize,
        head: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        num_heads: usize,
    ) void {
        const neg_inf = -std.math.inf(f32);
        
        for (0..seq_len_q) |i| {
            for (0..seq_len_k) |j| {
                if (j > i) {
                    const score_idx = batch * num_heads * seq_len_q * seq_len_k + 
                                     head * seq_len_q * seq_len_k + 
                                     i * seq_len_k + j;
                    scores_data[score_idx] = neg_inf;
                }
            }
        }
    }

    fn applySoftmax(
        scores_data: []f32,
        batch: usize,
        head: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        num_heads: usize,
    ) !void {
        for (0..seq_len_q) |i| {
            const row_offset = batch * num_heads * seq_len_q * seq_len_k + 
                              head * seq_len_q * seq_len_k + 
                              i * seq_len_k;
            
            // Find max for numerical stability
            var max_val = scores_data[row_offset];
            for (1..seq_len_k) |j| {
                max_val = @max(max_val, scores_data[row_offset + j]);
            }
            
            // Compute exp and sum
            var sum: f32 = 0.0;
            for (0..seq_len_k) |j| {
                scores_data[row_offset + j] = @exp(scores_data[row_offset + j] - max_val);
                sum += scores_data[row_offset + j];
            }
            
            // Normalize
            for (0..seq_len_k) |j| {
                scores_data[row_offset + j] /= sum;
            }
        }
    }

    fn applyAttentionToValues(
        scores_data: []const f32,
        value_data: []const f32,
        output_data: []f32,
        batch: usize,
        head: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        embed_dim: usize,
        num_heads: usize,
    ) !void {
        for (0..seq_len_q) |i| {
            for (0..head_dim) |d| {
                var weighted_sum: f32 = 0.0;
                
                for (0..seq_len_k) |j| {
                    const score_idx = batch * num_heads * seq_len_q * seq_len_k + 
                                     head * seq_len_q * seq_len_k + 
                                     i * seq_len_k + j;
                    const value_idx = batch * seq_len_k * embed_dim + j * embed_dim + head * head_dim + d;
                    
                    weighted_sum += scores_data[score_idx] * value_data[value_idx];
                }
                
                const output_idx = batch * seq_len_q * embed_dim + i * embed_dim + head * head_dim + d;
                output_data[output_idx] += weighted_sum;
            }
        }
    }
});

/// Scaled Dot-Product Attention (simpler version)
pub const ScaledDotProductAttention = BaseOperator(struct {
    const Self = @This();

    pub fn getMetadata() OperatorInterface.Metadata {
        return OperatorInterface.Metadata{
            .name = "ScaledDotProductAttention",
            .version = "1.0.0",
            .description = "Scaled dot-product attention mechanism",
            .domain = "ai.onnx",
            .min_inputs = 3,
            .max_inputs = 4, // query, key, value, [mask]
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

        if (input_shapes.len < 3 or input_shapes.len > 4) {
            return FrameworkError.InvalidInput;
        }

        // All inputs should have the same data type
        for (input_types[1..]) |dtype| {
            if (dtype != input_types[0]) {
                return FrameworkError.DataTypeMismatch;
            }
        }

        const query_shape = input_shapes[0];
        const key_shape = input_shapes[1];
        const value_shape = input_shapes[2];

        // Basic shape validation
        if (query_shape.len < 2 or key_shape.len < 2 or value_shape.len < 2) {
            return FrameworkError.ShapeMismatch;
        }

        // Check that key and value have compatible sequence lengths
        if (key_shape[key_shape.len - 2] != value_shape[value_shape.len - 2]) {
            return FrameworkError.ShapeMismatch;
        }

        // Check that query and key have compatible dimensions
        if (query_shape[query_shape.len - 1] != key_shape[key_shape.len - 1]) {
            return FrameworkError.ShapeMismatch;
        }
    }

    pub fn inferShapes(
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: std.mem.Allocator,
    ) FrameworkError![][]usize {
        _ = attributes;

        if (input_shapes.len < 3) {
            return FrameworkError.InvalidInput;
        }

        const query_shape = input_shapes[0];
        const value_shape = input_shapes[2];

        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.alloc(usize, query_shape.len);
        
        // Output shape: [..., seq_len_q, value_dim]
        for (0..query_shape.len - 1) |i| {
            output_shapes[0][i] = query_shape[i];
        }
        output_shapes[0][query_shape.len - 1] = value_shape[value_shape.len - 1];
        
        return output_shapes;
    }

    pub fn compute(
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void {
        if (inputs.len < 3 or inputs.len > 4 or outputs.len != 1) {
            return FrameworkError.InvalidInput;
        }

        const query = &inputs[0];
        const key = &inputs[1];
        const value = &inputs[2];
        const output = &outputs[0];

        const dropout_p = attributes.getFloat("dropout_p", 0.0);
        const is_causal = attributes.getInt("is_causal", 0) != 0;

        switch (query.dtype) {
            .f32 => try scaledDotProductAttentionF32(query, key, value, output, dropout_p, is_causal, context),
            else => return FrameworkError.UnsupportedOperation,
        }
    }

    fn scaledDotProductAttentionF32(
        query: *const Tensor,
        key: *const Tensor,
        value: *const Tensor,
        output: *const Tensor,
        dropout_p: f64,
        is_causal: bool,
        context: *ExecutionContext,
    ) !void {
        _ = dropout_p; // TODO: Implement dropout
        
        const query_data = query.getData(f32);
        const key_data = key.getData(f32);
        const value_data = value.getData(f32);
        const output_data = output.getMutableData(f32);

        const batch_size = query.shape[0];
        const seq_len_q = query.shape[1];
        const seq_len_k = key.shape[1];
        const d_k = query.shape[2];
        const d_v = value.shape[2];
        
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_k)));

        // Allocate temporary tensor for attention scores
        const scores_size = batch_size * seq_len_q * seq_len_k;
        const scores_memory = try context.allocateMemory(scores_size * @sizeOf(f32));
        defer context.freeMemory(scores_memory);
        const scores_data = std.mem.bytesAsSlice(f32, scores_memory);

        for (0..batch_size) |batch| {
            // Compute attention scores: Q * K^T
            for (0..seq_len_q) |i| {
                for (0..seq_len_k) |j| {
                    var score: f32 = 0.0;
                    
                    for (0..d_k) |k| {
                        const q_idx = batch * seq_len_q * d_k + i * d_k + k;
                        const k_idx = batch * seq_len_k * d_k + j * d_k + k;
                        score += query_data[q_idx] * key_data[k_idx];
                    }
                    
                    const score_idx = batch * seq_len_q * seq_len_k + i * seq_len_k + j;
                    scores_data[score_idx] = score * scale;
                }
            }

            // Apply causal mask if needed
            if (is_causal) {
                const neg_inf = -std.math.inf(f32);
                for (0..seq_len_q) |i| {
                    for (0..seq_len_k) |j| {
                        if (j > i) {
                            const score_idx = batch * seq_len_q * seq_len_k + i * seq_len_k + j;
                            scores_data[score_idx] = neg_inf;
                        }
                    }
                }
            }

            // Apply softmax
            for (0..seq_len_q) |i| {
                const row_offset = batch * seq_len_q * seq_len_k + i * seq_len_k;
                
                // Find max for numerical stability
                var max_val = scores_data[row_offset];
                for (1..seq_len_k) |j| {
                    max_val = @max(max_val, scores_data[row_offset + j]);
                }
                
                // Compute exp and sum
                var sum: f32 = 0.0;
                for (0..seq_len_k) |j| {
                    scores_data[row_offset + j] = @exp(scores_data[row_offset + j] - max_val);
                    sum += scores_data[row_offset + j];
                }
                
                // Normalize
                for (0..seq_len_k) |j| {
                    scores_data[row_offset + j] /= sum;
                }
            }

            // Apply attention to values
            for (0..seq_len_q) |i| {
                for (0..d_v) |d| {
                    var weighted_sum: f32 = 0.0;
                    
                    for (0..seq_len_k) |j| {
                        const score_idx = batch * seq_len_q * seq_len_k + i * seq_len_k + j;
                        const value_idx = batch * seq_len_k * d_v + j * d_v + d;
                        
                        weighted_sum += scores_data[score_idx] * value_data[value_idx];
                    }
                    
                    const output_idx = batch * seq_len_q * d_v + i * d_v + d;
                    output_data[output_idx] = weighted_sum;
                }
            }
        }
    }
});

// Tests
test "ScaledDotProductAttention operator" {
    const allocator = std.testing.allocator;
    
    // Create small tensors for testing
    const batch_size = 1;
    const seq_len = 2;
    const d_model = 4;
    
    const qkv_shape = [_]usize{ batch_size, seq_len, d_model };
    var query = try framework.utils.createTensor(allocator, &qkv_shape, .f32);
    defer query.deinit();
    var key = try framework.utils.createTensor(allocator, &qkv_shape, .f32);
    defer key.deinit();
    var value = try framework.utils.createTensor(allocator, &qkv_shape, .f32);
    defer value.deinit();
    var output = try framework.utils.createTensor(allocator, &qkv_shape, .f32);
    defer output.deinit();
    
    // Set simple test data
    const q_data = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };
    const k_data = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };
    const v_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    
    try framework.utils.setTensorData(&query, f32, &q_data);
    try framework.utils.setTensorData(&key, f32, &k_data);
    try framework.utils.setTensorData(&value, f32, &v_data);
    
    const inputs = [_]Tensor{ query, key, value };
    var outputs = [_]Tensor{output};
    
    var attrs = framework.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    var context = framework.utils.createExecutionContext(allocator);
    
    try ScaledDotProductAttention.compute(&inputs, &outputs, &attrs, &context);
    
    // Verify that attention was computed (output should not be zero)
    const result_data = framework.utils.getTensorData(&output, f32);
    var has_nonzero = false;
    for (result_data) |val| {
        if (val != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero);
}

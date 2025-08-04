const std = @import("std");
const matrix = @import("matrix.zig");
const activations = @import("activations.zig");
const simd = @import("simd.zig");

const Matrix = matrix.Matrix;

/// Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / √d_k)V
pub fn scaledDotProductAttention(
    query: Matrix,           // [seq_len, d_k]
    key: Matrix,             // [seq_len, d_k] 
    value: Matrix,           // [seq_len, d_v]
    output: *Matrix,         // [seq_len, d_v]
    mask: ?Matrix,           // [seq_len, seq_len] optional causal mask
    allocator: std.mem.Allocator,
) !void {
    const seq_len = query.rows;
    const d_k = query.cols;
    const d_v = value.cols;
    
    std.debug.assert(key.rows == seq_len and key.cols == d_k);
    std.debug.assert(value.rows == seq_len);
    std.debug.assert(output.rows == seq_len and output.cols == d_v);
    
    // Compute QK^T
    var key_transposed = try Matrix.init(allocator, d_k, seq_len);
    defer key_transposed.deinit();
    try matrix.transpose(key, &key_transposed);
    
    var scores = try Matrix.init(allocator, seq_len, seq_len);
    defer scores.deinit();
    try matrix.matmul(query, key_transposed, &scores);
    
    // Scale by √d_k
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_k)));
    try matrix.scale(scores, scale, &scores);
    
    // Apply mask if provided
    if (mask) |m| {
        std.debug.assert(m.rows == seq_len and m.cols == seq_len);
        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                if (m.get(i, j) == 0.0) {
                    scores.set(i, j, -std.math.inf(f32));
                }
            }
        }
    }
    
    // Apply softmax to each row
    for (0..seq_len) |i| {
        const scores_row = scores.getRow(i);
        activations.softmax(scores_row, scores_row);
    }
    
    // Compute attention output: scores * V
    try matrix.matmul(scores, value, output);
}

/// Multi-head attention implementation
pub const MultiHeadAttention = struct {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    d_v: usize,
    
    // Weight matrices
    w_q: Matrix,  // [d_model, d_model]
    w_k: Matrix,  // [d_model, d_model] 
    w_v: Matrix,  // [d_model, d_model]
    w_o: Matrix,  // [d_model, d_model]
    
    allocator: std.mem.Allocator,
    
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        num_heads: usize,
    ) !MultiHeadAttention {
        std.debug.assert(d_model % num_heads == 0);
        
        const d_k = d_model / num_heads;
        const d_v = d_k; // Usually d_v = d_k
        
        var w_q = try Matrix.init(allocator, d_model, d_model);
        var w_k = try Matrix.init(allocator, d_model, d_model);
        var w_v = try Matrix.init(allocator, d_model, d_model);
        var w_o = try Matrix.init(allocator, d_model, d_model);
        
        // Initialize weights (Xavier/Glorot initialization)
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = rng.random();
        
        const xavier_std = @sqrt(2.0 / @as(f32, @floatFromInt(d_model + d_model)));
        
        for (w_q.data) |*val| val.* = random.floatNorm(f32) * xavier_std;
        for (w_k.data) |*val| val.* = random.floatNorm(f32) * xavier_std;
        for (w_v.data) |*val| val.* = random.floatNorm(f32) * xavier_std;
        for (w_o.data) |*val| val.* = random.floatNorm(f32) * xavier_std;
        
        return MultiHeadAttention{
            .num_heads = num_heads,
            .d_model = d_model,
            .d_k = d_k,
            .d_v = d_v,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MultiHeadAttention) void {
        self.w_q.deinit();
        self.w_k.deinit();
        self.w_v.deinit();
        self.w_o.deinit();
    }
    
    pub fn forward(
        self: *MultiHeadAttention,
        input: Matrix,           // [seq_len, d_model]
        output: *Matrix,         // [seq_len, d_model]
        mask: ?Matrix,           // [seq_len, seq_len] optional causal mask
    ) !void {
        const seq_len = input.rows;
        std.debug.assert(input.cols == self.d_model);
        std.debug.assert(output.rows == seq_len and output.cols == self.d_model);
        
        // Compute Q, K, V projections
        var q = try Matrix.init(self.allocator, seq_len, self.d_model);
        defer q.deinit();
        var k = try Matrix.init(self.allocator, seq_len, self.d_model);
        defer k.deinit();
        var v = try Matrix.init(self.allocator, seq_len, self.d_model);
        defer v.deinit();
        
        try matrix.matmul(input, self.w_q, &q);
        try matrix.matmul(input, self.w_k, &k);
        try matrix.matmul(input, self.w_v, &v);
        
        // Split into heads and compute attention
        var concat_heads = try Matrix.initZeros(self.allocator, seq_len, self.d_model);
        defer concat_heads.deinit();
        
        for (0..self.num_heads) |head| {
            const head_start = head * self.d_k;
            const head_end = head_start + self.d_k;
            
            // Extract head slices
            var q_head = try Matrix.init(self.allocator, seq_len, self.d_k);
            defer q_head.deinit();
            var k_head = try Matrix.init(self.allocator, seq_len, self.d_k);
            defer k_head.deinit();
            var v_head = try Matrix.init(self.allocator, seq_len, self.d_k);
            defer v_head.deinit();
            
            // Copy head data
            for (0..seq_len) |i| {
                const q_row = q.getRow(i);
                const k_row = k.getRow(i);
                const v_row = v.getRow(i);
                const q_head_row = q_head.getRow(i);
                const k_head_row = k_head.getRow(i);
                const v_head_row = v_head.getRow(i);
                
                @memcpy(q_head_row, q_row[head_start..head_end]);
                @memcpy(k_head_row, k_row[head_start..head_end]);
                @memcpy(v_head_row, v_row[head_start..head_end]);
            }
            
            // Compute attention for this head
            var head_output = try Matrix.init(self.allocator, seq_len, self.d_k);
            defer head_output.deinit();
            
            try scaledDotProductAttention(q_head, k_head, v_head, &head_output, mask, self.allocator);
            
            // Copy head output back to concatenated result
            for (0..seq_len) |i| {
                const concat_row = concat_heads.getRow(i);
                const head_row = head_output.getRow(i);
                @memcpy(concat_row[head_start..head_end], head_row);
            }
        }
        
        // Final linear projection
        try matrix.matmul(concat_heads, self.w_o, output);
    }
};

/// Create causal mask for autoregressive attention
pub fn createCausalMask(allocator: std.mem.Allocator, seq_len: usize) !Matrix {
    var mask = try Matrix.init(allocator, seq_len, seq_len);
    
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            // Allow attention to current and previous positions
            mask.set(i, j, if (j <= i) 1.0 else 0.0);
        }
    }
    
    return mask;
}

/// Rotary Position Embedding (RoPE) implementation
pub fn applyRotaryPositionEmbedding(
    tensor: *Matrix,        // [seq_len, d_model]
    position_ids: []const u32,
    theta: f32,
) void {
    const seq_len = tensor.rows;
    const d_model = tensor.cols;
    
    std.debug.assert(position_ids.len == seq_len);
    std.debug.assert(d_model % 2 == 0); // Must be even for rotation
    
    for (0..seq_len) |i| {
        const pos = @as(f32, @floatFromInt(position_ids[i]));
        const row = tensor.getRow(i);
        
        // Apply rotation to pairs of dimensions
        var j: usize = 0;
        while (j < d_model) : (j += 2) {
            const dim_idx = @as(f32, @floatFromInt(j / 2));
            const freq = 1.0 / std.math.pow(f32, theta, dim_idx / @as(f32, @floatFromInt(d_model / 2)));
            const angle = pos * freq;
            
            const cos_val = @cos(angle);
            const sin_val = @sin(angle);
            
            const x = row[j];
            const y = row[j + 1];
            
            row[j] = x * cos_val - y * sin_val;
            row[j + 1] = x * sin_val + y * cos_val;
        }
    }
}

/// Grouped Query Attention (GQA) for efficient inference
pub const GroupedQueryAttention = struct {
    num_query_heads: usize,
    num_key_value_heads: usize,
    d_model: usize,
    d_k: usize,
    
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
    
    allocator: std.mem.Allocator,
    
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        num_query_heads: usize,
        num_key_value_heads: usize,
    ) !GroupedQueryAttention {
        std.debug.assert(num_query_heads % num_key_value_heads == 0);
        std.debug.assert(d_model % num_query_heads == 0);
        
        const d_k = d_model / num_query_heads;
        
        var w_q = try Matrix.init(allocator, d_model, d_model);
        var w_k = try Matrix.init(allocator, d_model, num_key_value_heads * d_k);
        var w_v = try Matrix.init(allocator, d_model, num_key_value_heads * d_k);
        var w_o = try Matrix.init(allocator, d_model, d_model);
        
        // Initialize weights
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = rng.random();
        const xavier_std = @sqrt(2.0 / @as(f32, @floatFromInt(d_model + d_model)));
        
        for (w_q.data) |*val| val.* = random.floatNorm(f32) * xavier_std;
        for (w_k.data) |*val| val.* = random.floatNorm(f32) * xavier_std;
        for (w_v.data) |*val| val.* = random.floatNorm(f32) * xavier_std;
        for (w_o.data) |*val| val.* = random.floatNorm(f32) * xavier_std;
        
        return GroupedQueryAttention{
            .num_query_heads = num_query_heads,
            .num_key_value_heads = num_key_value_heads,
            .d_model = d_model,
            .d_k = d_k,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *GroupedQueryAttention) void {
        self.w_q.deinit();
        self.w_k.deinit();
        self.w_v.deinit();
        self.w_o.deinit();
    }
    
    // Forward pass implementation would be similar to MultiHeadAttention
    // but with key/value head repetition for efficiency
};

test "scaled dot product attention" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const seq_len = 4;
    const d_k = 8;
    const d_v = 8;
    
    var query = try Matrix.initRandom(allocator, seq_len, d_k, std.rand.DefaultPrng.init(42).random());
    defer query.deinit();
    var key = try Matrix.initRandom(allocator, seq_len, d_k, std.rand.DefaultPrng.init(43).random());
    defer key.deinit();
    var value = try Matrix.initRandom(allocator, seq_len, d_v, std.rand.DefaultPrng.init(44).random());
    defer value.deinit();
    
    var output = try Matrix.initZeros(allocator, seq_len, d_v);
    defer output.deinit();
    
    try scaledDotProductAttention(query, key, value, &output, null, allocator);
    
    // Basic sanity check - output should not be all zeros
    var has_nonzero = false;
    for (output.data) |val| {
        if (val != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "causal mask creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var mask = try createCausalMask(allocator, 3);
    defer mask.deinit();
    
    // Check causal pattern
    try testing.expect(mask.get(0, 0) == 1.0);
    try testing.expect(mask.get(0, 1) == 0.0);
    try testing.expect(mask.get(0, 2) == 0.0);
    
    try testing.expect(mask.get(1, 0) == 1.0);
    try testing.expect(mask.get(1, 1) == 1.0);
    try testing.expect(mask.get(1, 2) == 0.0);
    
    try testing.expect(mask.get(2, 0) == 1.0);
    try testing.expect(mask.get(2, 1) == 1.0);
    try testing.expect(mask.get(2, 2) == 1.0);
}

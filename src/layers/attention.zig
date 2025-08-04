const std = @import("std");
const matrix = @import("../math/matrix.zig");
const attention_math = @import("../math/attention.zig");
const linear = @import("linear.zig");

const Matrix = matrix.Matrix;
const Linear = linear.Linear;

/// Multi-head attention layer wrapper
pub const MultiHeadAttention = struct {
    d_model: usize,
    num_heads: usize,
    d_k: usize,
    d_v: usize,
    
    // Linear projections
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, d_model: usize, num_heads: usize) !MultiHeadAttention {
        std.debug.assert(d_model % num_heads == 0);
        
        const d_k = d_model / num_heads;
        const d_v = d_k;
        
        var w_q = try Linear.init(allocator, d_model, d_model, false);
        var w_k = try Linear.init(allocator, d_model, d_model, false);
        var w_v = try Linear.init(allocator, d_model, d_model, false);
        var w_o = try Linear.init(allocator, d_model, d_model, false);
        
        return MultiHeadAttention{
            .d_model = d_model,
            .num_heads = num_heads,
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
    
    /// Self-attention forward pass
    pub fn forward(
        self: *MultiHeadAttention,
        input: Matrix,
        output: *Matrix,
        mask: ?Matrix,
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
        
        try self.w_q.forward(input, &q);
        try self.w_k.forward(input, &k);
        try self.w_v.forward(input, &v);
        
        // Reshape and compute multi-head attention
        var concat_output = try Matrix.init(self.allocator, seq_len, self.d_model);
        defer concat_output.deinit();
        
        try self.computeMultiHeadAttention(q, k, v, &concat_output, mask);
        
        // Final output projection
        try self.w_o.forward(concat_output, output);
    }
    
    /// Cross-attention forward pass (for decoder)
    pub fn forwardCross(
        self: *MultiHeadAttention,
        query_input: Matrix,
        key_value_input: Matrix,
        output: *Matrix,
        mask: ?Matrix,
    ) !void {
        const seq_len = query_input.rows;
        const kv_seq_len = key_value_input.rows;
        
        std.debug.assert(query_input.cols == self.d_model);
        std.debug.assert(key_value_input.cols == self.d_model);
        std.debug.assert(output.rows == seq_len and output.cols == self.d_model);
        
        // Compute projections
        var q = try Matrix.init(self.allocator, seq_len, self.d_model);
        defer q.deinit();
        var k = try Matrix.init(self.allocator, kv_seq_len, self.d_model);
        defer k.deinit();
        var v = try Matrix.init(self.allocator, kv_seq_len, self.d_model);
        defer v.deinit();
        
        try self.w_q.forward(query_input, &q);
        try self.w_k.forward(key_value_input, &k);
        try self.w_v.forward(key_value_input, &v);
        
        // Compute cross-attention
        var concat_output = try Matrix.init(self.allocator, seq_len, self.d_model);
        defer concat_output.deinit();
        
        try self.computeMultiHeadCrossAttention(q, k, v, &concat_output, mask);
        
        // Final output projection
        try self.w_o.forward(concat_output, output);
    }
    
    fn computeMultiHeadAttention(
        self: *MultiHeadAttention,
        q: Matrix,
        k: Matrix,
        v: Matrix,
        output: *Matrix,
        mask: ?Matrix,
    ) !void {
        const seq_len = q.rows;
        
        // Initialize output
        output.fill(0.0);
        
        // Process each head
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
            
            try attention_math.scaledDotProductAttention(
                q_head,
                k_head,
                v_head,
                &head_output,
                mask,
                self.allocator,
            );
            
            // Copy head output back to concatenated result
            for (0..seq_len) |i| {
                const output_row = output.getRow(i);
                const head_row = head_output.getRow(i);
                @memcpy(output_row[head_start..head_end], head_row);
            }
        }
    }
    
    fn computeMultiHeadCrossAttention(
        self: *MultiHeadAttention,
        q: Matrix,
        k: Matrix,
        v: Matrix,
        output: *Matrix,
        mask: ?Matrix,
    ) !void {
        const seq_len = q.rows;
        const kv_seq_len = k.rows;
        
        // Initialize output
        output.fill(0.0);
        
        // Process each head
        for (0..self.num_heads) |head| {
            const head_start = head * self.d_k;
            const head_end = head_start + self.d_k;
            
            // Extract head slices
            var q_head = try Matrix.init(self.allocator, seq_len, self.d_k);
            defer q_head.deinit();
            var k_head = try Matrix.init(self.allocator, kv_seq_len, self.d_k);
            defer k_head.deinit();
            var v_head = try Matrix.init(self.allocator, kv_seq_len, self.d_k);
            defer v_head.deinit();
            
            // Copy head data
            for (0..seq_len) |i| {
                const q_row = q.getRow(i);
                const q_head_row = q_head.getRow(i);
                @memcpy(q_head_row, q_row[head_start..head_end]);
            }
            
            for (0..kv_seq_len) |i| {
                const k_row = k.getRow(i);
                const v_row = v.getRow(i);
                const k_head_row = k_head.getRow(i);
                const v_head_row = v_head.getRow(i);
                @memcpy(k_head_row, k_row[head_start..head_end]);
                @memcpy(v_head_row, v_row[head_start..head_end]);
            }
            
            // Compute attention for this head
            var head_output = try Matrix.init(self.allocator, seq_len, self.d_k);
            defer head_output.deinit();
            
            try attention_math.scaledDotProductAttention(
                q_head,
                k_head,
                v_head,
                &head_output,
                mask,
                self.allocator,
            );
            
            // Copy head output back to concatenated result
            for (0..seq_len) |i| {
                const output_row = output.getRow(i);
                const head_row = head_output.getRow(i);
                @memcpy(output_row[head_start..head_end], head_row);
            }
        }
    }
    
    /// Load weights from external data
    pub fn loadWeights(
        self: *MultiHeadAttention,
        q_weights: []const f32,
        k_weights: []const f32,
        v_weights: []const f32,
        o_weights: []const f32,
    ) !void {
        try self.w_q.loadWeights(q_weights, null);
        try self.w_k.loadWeights(k_weights, null);
        try self.w_v.loadWeights(v_weights, null);
        try self.w_o.loadWeights(o_weights, null);
    }
    
    /// Get parameter count
    pub fn getParameterCount(self: *MultiHeadAttention) usize {
        return self.w_q.getParameterCount() +
               self.w_k.getParameterCount() +
               self.w_v.getParameterCount() +
               self.w_o.getParameterCount();
    }
};

test "multi-head attention creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var mha = try MultiHeadAttention.init(allocator, 512, 8);
    defer mha.deinit();
    
    try testing.expect(mha.d_model == 512);
    try testing.expect(mha.num_heads == 8);
    try testing.expect(mha.d_k == 64);
}

test "multi-head attention forward" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var mha = try MultiHeadAttention.init(allocator, 64, 4);
    defer mha.deinit();
    
    var input = try Matrix.initRandom(allocator, 8, 64, std.rand.DefaultPrng.init(42).random());
    defer input.deinit();
    
    var output = try Matrix.init(allocator, 8, 64);
    defer output.deinit();
    
    try mha.forward(input, &output, null);
    
    // Basic sanity check
    var has_nonzero = false;
    for (output.data) |val| {
        if (val != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

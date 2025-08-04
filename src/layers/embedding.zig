const std = @import("std");
const matrix = @import("../math/matrix.zig");
const embeddings_math = @import("../math/embeddings.zig");

const Matrix = matrix.Matrix;

/// Token embedding layer wrapper
pub const TokenEmbedding = struct {
    vocab_size: usize,
    embedding_dim: usize,
    weights: Matrix,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embedding_dim: usize) !TokenEmbedding {
        var weights = try Matrix.init(allocator, vocab_size, embedding_dim);

        // Initialize with small random values
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = rng.random();
        const std_dev = @sqrt(1.0 / @as(f32, @floatFromInt(embedding_dim)));

        for (weights.data) |*val| {
            val.* = random.floatNorm(f32) * std_dev;
        }

        return TokenEmbedding{
            .vocab_size = vocab_size,
            .embedding_dim = embedding_dim,
            .weights = weights,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TokenEmbedding) void {
        self.weights.deinit();
    }

    /// Forward pass: look up embeddings for token IDs
    pub fn forward(self: *TokenEmbedding, token_ids: []const u32, output: *Matrix) !void {
        const seq_len = token_ids.len;
        std.debug.assert(output.rows == seq_len and output.cols == self.embedding_dim);

        for (token_ids, 0..) |token_id, i| {
            if (token_id >= self.vocab_size) {
                return error.TokenIdOutOfRange;
            }

            // Copy embedding vector
            const embedding_row = self.weights.getRow(token_id);
            const output_row = output.getRow(i);
            @memcpy(output_row, embedding_row);
        }
    }

    /// Load weights from external data
    pub fn loadWeights(self: *TokenEmbedding, weight_data: []const f32) !void {
        if (weight_data.len != self.weights.data.len) {
            return error.WeightSizeMismatch;
        }
        @memcpy(self.weights.data, weight_data);
    }

    /// Get parameter count
    pub fn getParameterCount(self: *TokenEmbedding) usize {
        return self.weights.data.len;
    }
};

/// Positional embedding layer wrapper
pub const PositionalEmbedding = struct {
    max_seq_len: usize,
    embedding_dim: usize,
    weights: Matrix,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_seq_len: usize, embedding_dim: usize) !PositionalEmbedding {
        var weights = try Matrix.init(allocator, max_seq_len, embedding_dim);

        // Initialize with small random values
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = rng.random();
        const std_dev = @sqrt(1.0 / @as(f32, @floatFromInt(embedding_dim)));

        for (weights.data) |*val| {
            val.* = random.floatNorm(f32) * std_dev;
        }

        return PositionalEmbedding{
            .max_seq_len = max_seq_len,
            .embedding_dim = embedding_dim,
            .weights = weights,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PositionalEmbedding) void {
        self.weights.deinit();
    }

    /// Forward pass: get positional embeddings for sequence
    pub fn forward(self: *PositionalEmbedding, seq_len: usize, output: *Matrix) !void {
        std.debug.assert(output.rows == seq_len and output.cols == self.embedding_dim);

        if (seq_len > self.max_seq_len) {
            return error.SequenceTooLong;
        }

        for (0..seq_len) |i| {
            const pos_embedding_row = self.weights.getRow(i);
            const output_row = output.getRow(i);
            @memcpy(output_row, pos_embedding_row);
        }
    }

    /// Load weights from external data
    pub fn loadWeights(self: *PositionalEmbedding, weight_data: []const f32) !void {
        if (weight_data.len != self.weights.data.len) {
            return error.WeightSizeMismatch;
        }
        @memcpy(self.weights.data, weight_data);
    }

    /// Get parameter count
    pub fn getParameterCount(self: *PositionalEmbedding) usize {
        return self.weights.data.len;
    }
};

/// Combined input embedding layer (token + positional)
pub const InputEmbedding = struct {
    token_embedding: TokenEmbedding,
    positional_embedding: ?PositionalEmbedding,
    sinusoidal_encoding: ?Matrix,
    dropout_rate: f32,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        vocab_size: usize,
        embedding_dim: usize,
        max_seq_len: usize,
        use_learnable_pos: bool,
        dropout_rate: f32,
    ) !InputEmbedding {
        var token_embedding = try TokenEmbedding.init(allocator, vocab_size, embedding_dim);

        var positional_embedding: ?PositionalEmbedding = null;
        var sinusoidal_encoding: ?Matrix = null;

        if (use_learnable_pos) {
            positional_embedding = try PositionalEmbedding.init(allocator, max_seq_len, embedding_dim);
        } else {
            sinusoidal_encoding = try embeddings_math.sinusoidalPositionalEncoding(
                allocator,
                max_seq_len,
                embedding_dim,
            );
        }

        return InputEmbedding{
            .token_embedding = token_embedding,
            .positional_embedding = positional_embedding,
            .sinusoidal_encoding = sinusoidal_encoding,
            .dropout_rate = dropout_rate,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *InputEmbedding) void {
        self.token_embedding.deinit();

        if (self.positional_embedding) |*pos_emb| {
            pos_emb.deinit();
        }

        if (self.sinusoidal_encoding) |*sin_enc| {
            sin_enc.deinit();
        }
    }

    /// Forward pass: token embeddings + positional embeddings
    pub fn forward(self: *InputEmbedding, token_ids: []const u32, output: *Matrix) !void {
        const seq_len = token_ids.len;
        std.debug.assert(output.rows == seq_len and output.cols == self.token_embedding.embedding_dim);

        // Get token embeddings
        try self.token_embedding.forward(token_ids, output);

        // Add positional embeddings
        if (self.positional_embedding) |*pos_emb| {
            var pos_output = try Matrix.init(self.allocator, seq_len, self.token_embedding.embedding_dim);
            defer pos_output.deinit();

            try pos_emb.forward(seq_len, &pos_output);

            // Add positional to token embeddings
            for (0..seq_len) |i| {
                const output_row = output.getRow(i);
                const pos_row = pos_output.getRow(i);

                for (output_row, pos_row) |*out_val, pos_val| {
                    out_val.* += pos_val;
                }
            }
        } else if (self.sinusoidal_encoding) |sin_enc| {
            // Add sinusoidal positional encoding
            for (0..seq_len) |i| {
                const output_row = output.getRow(i);
                const pos_row = sin_enc.getRow(i);

                for (output_row, pos_row) |*out_val, pos_val| {
                    out_val.* += pos_val;
                }
            }
        }

        // Apply dropout scaling (simplified for inference)
        if (self.dropout_rate > 0.0) {
            const scale = 1.0 - self.dropout_rate;
            for (output.data) |*val| {
                val.* *= scale;
            }
        }
    }

    /// Load weights from external data
    pub fn loadWeights(
        self: *InputEmbedding,
        token_weights: []const f32,
        pos_weights: ?[]const f32,
    ) !void {
        try self.token_embedding.loadWeights(token_weights);

        if (pos_weights) |weights| {
            if (self.positional_embedding) |*pos_emb| {
                try pos_emb.loadWeights(weights);
            } else {
                return error.PositionalWeightsNotExpected;
            }
        }
    }

    /// Get parameter count
    pub fn getParameterCount(self: *InputEmbedding) usize {
        var count = self.token_embedding.getParameterCount();
        if (self.positional_embedding) |*pos_emb| {
            count += pos_emb.getParameterCount();
        }
        return count;
    }
};

test "token embedding" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var embedding = try TokenEmbedding.init(allocator, 100, 64);
    defer embedding.deinit();

    const token_ids = [_]u32{ 0, 1, 2 };
    var output = try Matrix.init(allocator, 3, 64);
    defer output.deinit();

    try embedding.forward(&token_ids, &output);

    // Check that embeddings are different for different tokens
    const emb0 = output.getRow(0);
    const emb1 = output.getRow(1);

    var different = false;
    for (emb0, emb1) |val0, val1| {
        if (val0 != val1) {
            different = true;
            break;
        }
    }
    try testing.expect(different);
}

test "positional embedding" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var pos_emb = try PositionalEmbedding.init(allocator, 10, 8);
    defer pos_emb.deinit();

    var output = try Matrix.init(allocator, 5, 8);
    defer output.deinit();

    try pos_emb.forward(5, &output);

    // Check that different positions have different embeddings
    const pos0 = output.getRow(0);
    const pos1 = output.getRow(1);

    var different = false;
    for (pos0, pos1) |val0, val1| {
        if (@fabs(val0 - val1) > 1e-6) {
            different = true;
            break;
        }
    }
    try testing.expect(different);
}

test "input embedding with sinusoidal" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var input_emb = try InputEmbedding.init(
        allocator,
        100, // vocab_size
        64, // embedding_dim
        10, // max_seq_len
        false, // use_learnable_pos (false = sinusoidal)
        0.0, // dropout_rate
    );
    defer input_emb.deinit();

    const token_ids = [_]u32{ 5, 10, 15 };
    var output = try Matrix.init(allocator, 3, 64);
    defer output.deinit();

    try input_emb.forward(&token_ids, &output);

    // Should complete without error
    try testing.expect(true);
}

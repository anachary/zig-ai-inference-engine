const std = @import("std");
const matrix = @import("matrix.zig");

const Matrix = matrix.Matrix;

/// Token embedding layer
pub const TokenEmbedding = struct {
    vocab_size: usize,
    embedding_dim: usize,
    weights: Matrix, // [vocab_size, embedding_dim]
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

    /// Look up embeddings for a sequence of token IDs
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

    /// Get embedding for a single token
    pub fn getEmbedding(self: *TokenEmbedding, token_id: u32) ![]const f32 {
        if (token_id >= self.vocab_size) {
            return error.TokenIdOutOfRange;
        }
        return self.weights.getRow(token_id);
    }
};

/// Sinusoidal positional encoding as used in the original Transformer paper
pub fn sinusoidalPositionalEncoding(
    allocator: std.mem.Allocator,
    max_seq_len: usize,
    d_model: usize,
) !Matrix {
    var pos_encoding = try Matrix.init(allocator, max_seq_len, d_model);

    for (0..max_seq_len) |pos| {
        const position = @as(f32, @floatFromInt(pos));

        for (0..d_model) |i| {
            const dim = @as(f32, @floatFromInt(i));
            const angle = position / std.math.pow(f32, 10000.0, dim / @as(f32, @floatFromInt(d_model)));

            if (i % 2 == 0) {
                pos_encoding.set(pos, i, @sin(angle));
            } else {
                pos_encoding.set(pos, i, @cos(angle));
            }
        }
    }

    return pos_encoding;
}

/// Learnable positional embedding
pub const PositionalEmbedding = struct {
    max_seq_len: usize,
    embedding_dim: usize,
    weights: Matrix, // [max_seq_len, embedding_dim]
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

    /// Get positional embeddings for a sequence
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
};

/// Combined token + positional embeddings
pub const InputEmbedding = struct {
    token_embedding: TokenEmbedding,
    positional_encoding: ?Matrix, // Sinusoidal (fixed)
    positional_embedding: ?PositionalEmbedding, // Learnable
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

        var positional_encoding: ?Matrix = null;
        var positional_embedding: ?PositionalEmbedding = null;

        if (use_learnable_pos) {
            positional_embedding = try PositionalEmbedding.init(allocator, max_seq_len, embedding_dim);
        } else {
            positional_encoding = try sinusoidalPositionalEncoding(allocator, max_seq_len, embedding_dim);
        }

        return InputEmbedding{
            .token_embedding = token_embedding,
            .positional_encoding = positional_encoding,
            .positional_embedding = positional_embedding,
            .dropout_rate = dropout_rate,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *InputEmbedding) void {
        self.token_embedding.deinit();

        if (self.positional_encoding) |*pos_enc| {
            pos_enc.deinit();
        }

        if (self.positional_embedding) |*pos_emb| {
            pos_emb.deinit();
        }
    }

    /// Forward pass: token embeddings + positional embeddings
    pub fn forward(self: *InputEmbedding, token_ids: []const u32, output: *Matrix) !void {
        const seq_len = token_ids.len;
        std.debug.assert(output.rows == seq_len and output.cols == self.token_embedding.embedding_dim);

        // Get token embeddings
        try self.token_embedding.forward(token_ids, output);

        // Add positional embeddings
        if (self.positional_encoding) |pos_enc| {
            for (0..seq_len) |i| {
                const output_row = output.getRow(i);
                const pos_row = pos_enc.getRow(i);

                for (output_row, pos_row) |*out_val, pos_val| {
                    out_val.* += pos_val;
                }
            }
        } else if (self.positional_embedding) |*pos_emb| {
            var pos_output = try Matrix.init(self.allocator, seq_len, self.token_embedding.embedding_dim);
            defer pos_output.deinit();

            try pos_emb.forward(seq_len, &pos_output);

            for (0..seq_len) |i| {
                const output_row = output.getRow(i);
                const pos_row = pos_output.getRow(i);

                for (output_row, pos_row) |*out_val, pos_val| {
                    out_val.* += pos_val;
                }
            }
        }

        // Apply dropout during training (simplified - just scale for inference)
        if (self.dropout_rate > 0.0) {
            const scale = 1.0 - self.dropout_rate;
            for (output.data) |*val| {
                val.* *= scale;
            }
        }
    }
};

/// Rotary Position Embedding (RoPE) frequencies
pub fn computeRoPEFrequencies(allocator: std.mem.Allocator, d_model: usize, theta: f32) ![]f32 {
    std.debug.assert(d_model % 2 == 0);

    const freqs = try allocator.alloc(f32, d_model / 2);

    for (0..d_model / 2) |i| {
        const dim_idx = @as(f32, @floatFromInt(i));
        freqs[i] = 1.0 / std.math.pow(f32, theta, dim_idx / @as(f32, @floatFromInt(d_model / 2)));
    }

    return freqs;
}

/// Apply RoPE to query and key matrices
pub fn applyRoPE(
    tensor: *Matrix,
    position_ids: []const u32,
    freqs: []const f32,
) void {
    const seq_len = tensor.rows;
    const d_model = tensor.cols;

    std.debug.assert(position_ids.len == seq_len);
    std.debug.assert(freqs.len == d_model / 2);

    for (0..seq_len) |i| {
        const pos = @as(f32, @floatFromInt(position_ids[i]));
        const row = tensor.getRow(i);

        for (0..freqs.len) |j| {
            const angle = pos * freqs[j];
            const cos_val = @cos(angle);
            const sin_val = @sin(angle);

            const x = row[j * 2];
            const y = row[j * 2 + 1];

            row[j * 2] = x * cos_val - y * sin_val;
            row[j * 2 + 1] = x * sin_val + y * cos_val;
        }
    }
}

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

test "sinusoidal positional encoding" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var pos_enc = try sinusoidalPositionalEncoding(allocator, 10, 8);
    defer pos_enc.deinit();

    // Check that different positions have different encodings
    const pos0 = pos_enc.getRow(0);
    const pos1 = pos_enc.getRow(1);

    var different = false;
    for (pos0, pos1) |val0, val1| {
        if (@fabs(val0 - val1) > 1e-6) {
            different = true;
            break;
        }
    }
    try testing.expect(different);
}

test "rope frequencies" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const freqs = try computeRoPEFrequencies(allocator, 8, 10000.0);
    defer allocator.free(freqs);

    try testing.expect(freqs.len == 4);

    // Frequencies should be decreasing
    for (1..freqs.len) |i| {
        try testing.expect(freqs[i] < freqs[i - 1]);
    }
}

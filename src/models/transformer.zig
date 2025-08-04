const std = @import("std");
const ModelConfig = @import("mod.zig").ModelConfig;
const layers = @import("../layers/mod.zig");
const math = @import("../math/mod.zig");
const Matrix = math.matrix.Matrix;

/// Generic Transformer model implementation
/// Can be configured for different architectures (GPT, LLaMA, Qwen, etc.)
pub const TransformerModel = struct {
    config: ModelConfig,
    allocator: std.mem.Allocator,

    // Model weights (loaded from external source)
    token_embeddings: ?*@import("../core/tensor.zig").DynamicTensor,
    position_embeddings: ?*@import("../core/tensor.zig").DynamicTensor,
    layer_weights: []LayerWeights,
    output_weights: ?*@import("../core/tensor.zig").DynamicTensor,

    // Layer normalization weights
    input_layernorm: ?*@import("../core/tensor.zig").DynamicTensor,
    output_layernorm: ?*@import("../core/tensor.zig").DynamicTensor,

    // Working memory for inference
    hidden_states: []f32,
    attention_cache: ?[]f32,

    const LayerWeights = struct {
        // Attention weights
        wq: ?*@import("../core/tensor.zig").DynamicTensor, // Query projection
        wk: ?*@import("../core/tensor.zig").DynamicTensor, // Key projection
        wv: ?*@import("../core/tensor.zig").DynamicTensor, // Value projection
        wo: ?*@import("../core/tensor.zig").DynamicTensor, // Output projection

        // Feed-forward weights
        w1: ?*@import("../core/tensor.zig").DynamicTensor, // Gate projection
        w2: ?*@import("../core/tensor.zig").DynamicTensor, // Down projection
        w3: ?*@import("../core/tensor.zig").DynamicTensor, // Up projection

        // Layer normalization
        attention_norm: ?*@import("../core/tensor.zig").DynamicTensor,
        ffn_norm: ?*@import("../core/tensor.zig").DynamicTensor,
    };

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !TransformerModel {
        // Allocate layer weights
        const layer_weights = try allocator.alloc(LayerWeights, config.num_layers);
        for (layer_weights) |*layer| {
            layer.* = LayerWeights{
                .wq = null,
                .wk = null,
                .wv = null,
                .wo = null,
                .w1 = null,
                .w2 = null,
                .w3 = null,
                .attention_norm = null,
                .ffn_norm = null,
            };
        }

        // Allocate working memory
        const hidden_size = config.hidden_size * config.max_position_embeddings;
        const hidden_states = try allocator.alloc(f32, hidden_size);

        return TransformerModel{
            .config = config,
            .allocator = allocator,
            .token_embeddings = null,
            .position_embeddings = null,
            .layer_weights = layer_weights,
            .output_weights = null,
            .input_layernorm = null,
            .output_layernorm = null,
            .hidden_states = hidden_states,
            .attention_cache = null,
        };
    }

    pub fn deinit(self: *TransformerModel) void {
        self.allocator.free(self.layer_weights);
        self.allocator.free(self.hidden_states);
        if (self.attention_cache) |cache| {
            self.allocator.free(cache);
        }
    }

    /// Load weights from a model (GGUF, ONNX, etc.)
    pub fn loadWeights(self: *TransformerModel, model: *@import("../core/model.zig").Model) !void {
        // Load token embeddings
        self.token_embeddings = model.getTensor("token_embd.weight") orelse
            model.getTensor("tok_embeddings.weight");

        // Load output weights
        self.output_weights = model.getTensor("output.weight") orelse
            model.getTensor("lm_head.weight");

        // Load layer weights
        for (self.layer_weights, 0..) |*layer, i| {
            const layer_prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}", .{i});
            defer self.allocator.free(layer_prefix);

            // Attention weights
            layer.wq = model.getTensor(try std.fmt.allocPrint(self.allocator, "{s}.attn_q.weight", .{layer_prefix}));
            layer.wk = model.getTensor(try std.fmt.allocPrint(self.allocator, "{s}.attn_k.weight", .{layer_prefix}));
            layer.wv = model.getTensor(try std.fmt.allocPrint(self.allocator, "{s}.attn_v.weight", .{layer_prefix}));
            layer.wo = model.getTensor(try std.fmt.allocPrint(self.allocator, "{s}.attn_output.weight", .{layer_prefix}));

            // Feed-forward weights
            layer.w1 = model.getTensor(try std.fmt.allocPrint(self.allocator, "{s}.ffn_gate.weight", .{layer_prefix}));
            layer.w2 = model.getTensor(try std.fmt.allocPrint(self.allocator, "{s}.ffn_down.weight", .{layer_prefix}));
            layer.w3 = model.getTensor(try std.fmt.allocPrint(self.allocator, "{s}.ffn_up.weight", .{layer_prefix}));

            // Layer normalization
            layer.attention_norm = model.getTensor(try std.fmt.allocPrint(self.allocator, "{s}.attn_norm.weight", .{layer_prefix}));
            layer.ffn_norm = model.getTensor(try std.fmt.allocPrint(self.allocator, "{s}.ffn_norm.weight", .{layer_prefix}));
        }

        std.log.info("Loaded transformer weights for {} layers", .{self.config.num_layers});
    }

    /// Forward pass through the transformer
    pub fn forward(self: *TransformerModel, input_ids: []const u32, output: []f32, allocator: std.mem.Allocator) !void {
        _ = allocator; // May be needed for temporary allocations

        const seq_len = input_ids.len;
        if (seq_len > self.config.max_position_embeddings) {
            return error.SequenceTooLong;
        }

        // Token embedding lookup
        try self.embedTokens(input_ids);

        // Process through transformer layers
        for (self.layer_weights, 0..) |*layer, layer_idx| {
            try self.processLayer(layer, layer_idx, seq_len);
        }

        // Final layer normalization
        if (self.output_layernorm) |norm| {
            try self.applyLayerNorm(norm, seq_len);
        }

        // Output projection to vocabulary
        try self.computeLogits(output, seq_len);
    }

    fn embedTokens(self: *TransformerModel, input_ids: []const u32) !void {
        if (self.token_embeddings == null) {
            return error.MissingTokenEmbeddings;
        }

        const embeddings = self.token_embeddings.?;
        const emb_data = std.mem.bytesAsSlice(f32, embeddings.data);

        // Clear hidden states
        @memset(self.hidden_states, 0.0);

        // Look up embeddings for each token
        for (input_ids, 0..) |token_id, pos| {
            if (token_id >= embeddings.shape[0]) {
                std.log.warn("Token ID {} exceeds vocab size {}", .{ token_id, embeddings.shape[0] });
                continue;
            }

            const emb_offset = token_id * self.config.hidden_size;
            const pos_offset = pos * self.config.hidden_size;

            // Copy embedding to hidden states
            for (0..self.config.hidden_size) |i| {
                if (emb_offset + i < emb_data.len) {
                    self.hidden_states[pos_offset + i] = emb_data[emb_offset + i];
                }
            }
        }
    }

    fn processLayer(self: *TransformerModel, layer: *LayerWeights, layer_idx: usize, seq_len: usize) !void {
        _ = layer_idx; // For debugging

        // Pre-attention layer normalization
        if (layer.attention_norm) |norm| {
            try self.applyLayerNormToRange(norm, 0, seq_len);
        }

        // Multi-head self-attention with real matrix operations
        try self.realMultiHeadAttention(layer, seq_len);

        // Pre-FFN layer normalization
        if (layer.ffn_norm) |norm| {
            try self.applyLayerNormToRange(norm, 0, seq_len);
        }

        // Feed-forward network with real SwiGLU activation
        try self.realFeedForwardNetwork(layer, seq_len);
    }

    fn realMultiHeadAttention(self: *TransformerModel, layer: *LayerWeights, seq_len: usize) !void {
        if (layer.wq == null or layer.wk == null or layer.wv == null or layer.wo == null) {
            std.log.warn("Attention weights not loaded, skipping attention");
            return;
        }

        const hidden_size = self.config.hidden_size;
        const num_heads = self.config.num_heads;
        const head_dim = hidden_size / num_heads;
        _ = head_dim;

        // Create matrix views for current hidden states
        var input_matrix = try Matrix.fromSlice(
            self.allocator,
            self.hidden_states[0 .. seq_len * hidden_size],
            seq_len,
            hidden_size,
        );
        defer input_matrix.deinit();

        // Project to Q, K, V using real weights
        var q_matrix = try Matrix.init(self.allocator, seq_len, hidden_size);
        defer q_matrix.deinit();
        var k_matrix = try Matrix.init(self.allocator, seq_len, hidden_size);
        defer k_matrix.deinit();
        var v_matrix = try Matrix.init(self.allocator, seq_len, hidden_size);
        defer v_matrix.deinit();

        // Load weight matrices
        var wq_matrix = try Matrix.fromSlice(
            self.allocator,
            std.mem.bytesAsSlice(f32, layer.wq.?.data),
            hidden_size,
            hidden_size,
        );
        defer wq_matrix.deinit();

        var wk_matrix = try Matrix.fromSlice(
            self.allocator,
            std.mem.bytesAsSlice(f32, layer.wk.?.data),
            hidden_size,
            hidden_size,
        );
        defer wk_matrix.deinit();

        var wv_matrix = try Matrix.fromSlice(
            self.allocator,
            std.mem.bytesAsSlice(f32, layer.wv.?.data),
            hidden_size,
            hidden_size,
        );
        defer wv_matrix.deinit();

        // Compute Q = input * Wq, K = input * Wk, V = input * Wv
        try math.matrix.matmul(input_matrix, wq_matrix, &q_matrix);
        try math.matrix.matmul(input_matrix, wk_matrix, &k_matrix);
        try math.matrix.matmul(input_matrix, wv_matrix, &v_matrix);

        // Apply multi-head attention (simplified - real implementation would reshape for heads)
        var attention_output = try Matrix.init(self.allocator, seq_len, hidden_size);
        defer attention_output.deinit();

        // For now, use simplified attention computation
        // TODO: Implement proper multi-head reshaping and attention
        try math.attention.scaledDotProductAttention(
            q_matrix,
            k_matrix,
            v_matrix,
            &attention_output,
            null, // No causal mask for now
            self.allocator,
        );

        // Output projection
        if (layer.wo) |wo_tensor| {
            var wo_matrix = try Matrix.fromSlice(
                self.allocator,
                std.mem.bytesAsSlice(f32, wo_tensor.data),
                hidden_size,
                hidden_size,
            );
            defer wo_matrix.deinit();

            var final_output = try Matrix.init(self.allocator, seq_len, hidden_size);
            defer final_output.deinit();

            try math.matrix.matmul(attention_output, wo_matrix, &final_output);

            // Copy result back to hidden states with residual connection
            for (0..seq_len) |i| {
                for (0..hidden_size) |j| {
                    const idx = i * hidden_size + j;
                    self.hidden_states[idx] += final_output.get(i, j);
                }
            }
        } else {
            // Copy attention output back to hidden states with residual connection
            for (0..seq_len) |i| {
                for (0..hidden_size) |j| {
                    const idx = i * hidden_size + j;
                    self.hidden_states[idx] += attention_output.get(i, j);
                }
            }
        }

        std.log.debug("Real multi-head attention complete for {} tokens", .{seq_len});
    }

    fn realFeedForwardNetwork(self: *TransformerModel, layer: *LayerWeights, seq_len: usize) !void {
        if (layer.w1 == null or layer.w2 == null or layer.w3 == null) {
            std.log.warn("FFN weights not loaded, skipping feed-forward");
            return;
        }

        const hidden_size = self.config.hidden_size;
        const intermediate_size = self.config.intermediate_size;

        // Create matrix view for current hidden states
        var input_matrix = try Matrix.fromSlice(
            self.allocator,
            self.hidden_states[0 .. seq_len * hidden_size],
            seq_len,
            hidden_size,
        );
        defer input_matrix.deinit();

        // Load weight matrices
        var w1_matrix = try Matrix.fromSlice(
            self.allocator,
            std.mem.bytesAsSlice(f32, layer.w1.?.data),
            hidden_size,
            intermediate_size,
        );
        defer w1_matrix.deinit();

        var w2_matrix = try Matrix.fromSlice(
            self.allocator,
            std.mem.bytesAsSlice(f32, layer.w2.?.data),
            intermediate_size,
            hidden_size,
        );
        defer w2_matrix.deinit();

        var w3_matrix = try Matrix.fromSlice(
            self.allocator,
            std.mem.bytesAsSlice(f32, layer.w3.?.data),
            hidden_size,
            intermediate_size,
        );
        defer w3_matrix.deinit();

        // Gate projection: gate = input * W1
        var gate_matrix = try Matrix.init(self.allocator, seq_len, intermediate_size);
        defer gate_matrix.deinit();
        try math.matrix.matmul(input_matrix, w1_matrix, &gate_matrix);

        // Up projection: up = input * W3
        var up_matrix = try Matrix.init(self.allocator, seq_len, intermediate_size);
        defer up_matrix.deinit();
        try math.matrix.matmul(input_matrix, w3_matrix, &up_matrix);

        // Apply SwiGLU activation: SiLU(gate) ⊙ up
        for (0..seq_len) |i| {
            for (0..intermediate_size) |j| {
                const gate_val = gate_matrix.get(i, j);
                const up_val = up_matrix.get(i, j);

                // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                const silu_gate = gate_val / (1.0 + @exp(-gate_val));
                const swiglu_result = silu_gate * up_val;

                gate_matrix.set(i, j, swiglu_result);
            }
        }

        // Down projection: output = SwiGLU_result * W2
        var ffn_output = try Matrix.init(self.allocator, seq_len, hidden_size);
        defer ffn_output.deinit();
        try math.matrix.matmul(gate_matrix, w2_matrix, &ffn_output);

        // Add residual connection
        for (0..seq_len) |i| {
            for (0..hidden_size) |j| {
                const idx = i * hidden_size + j;
                self.hidden_states[idx] += ffn_output.get(i, j);
            }
        }

        std.log.debug("Real feed-forward network complete for {} tokens", .{seq_len});
    }

    fn applyLayerNorm(self: *TransformerModel, norm_weights: *@import("../core/tensor.zig").DynamicTensor, seq_len: usize) !void {
        try self.applyLayerNormToRange(norm_weights, 0, seq_len);
    }

    fn applyLayerNormToRange(self: *TransformerModel, norm_weights: *@import("../core/tensor.zig").DynamicTensor, start_pos: usize, end_pos: usize) !void {
        const hidden_size = self.config.hidden_size;
        const weight_data = std.mem.bytesAsSlice(f32, norm_weights.data);

        // Apply layer normalization with learned weights
        for (start_pos..end_pos) |pos| {
            const pos_offset = pos * hidden_size;
            const hidden_slice = self.hidden_states[pos_offset .. pos_offset + hidden_size];

            // Use the math library's layer normalization
            try math.normalization.layerNorm(
                hidden_slice,
                weight_data[0..hidden_size],
                null, // No bias for most transformer variants
                self.config.layer_norm_eps,
            );
        }

        std.log.debug("Applied layer normalization to positions {}-{}", .{ start_pos, end_pos });
    }

    fn computeLogits(self: *TransformerModel, output: []f32, seq_len: usize) !void {
        if (self.output_weights == null) {
            return error.MissingOutputWeights;
        }

        const hidden_size = self.config.hidden_size;
        const vocab_size = self.config.vocab_size;

        // Use the last token's hidden state for next token prediction
        const last_pos = seq_len - 1;
        const last_hidden = self.hidden_states[last_pos * hidden_size .. (last_pos + 1) * hidden_size];

        // Create matrix views
        var hidden_matrix = try Matrix.fromSlice(
            self.allocator,
            last_hidden,
            1, // Single token
            hidden_size,
        );
        defer hidden_matrix.deinit();

        var output_weight_matrix = try Matrix.fromSlice(
            self.allocator,
            std.mem.bytesAsSlice(f32, self.output_weights.?.data),
            hidden_size,
            vocab_size,
        );
        defer output_weight_matrix.deinit();

        var logits_matrix = try Matrix.fromSlice(
            self.allocator,
            output,
            1, // Single token output
            vocab_size,
        );
        defer logits_matrix.deinit();

        // Compute logits = last_hidden * output_weights
        try math.matrix.matmul(hidden_matrix, output_weight_matrix, &logits_matrix);

        std.log.debug("Computed real logits for {} vocabulary items", .{vocab_size});
    }

    pub fn getConfig(self: *TransformerModel) ModelConfig {
        return self.config;
    }
};

test "transformer model creation" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = ModelConfig{
        .vocab_size = 1000,
        .hidden_size = 128,
        .num_layers = 2,
        .num_heads = 4,
        .intermediate_size = 512,
        .max_position_embeddings = 64,
    };

    var transformer = try TransformerModel.init(allocator, config);
    defer transformer.deinit();

    try testing.expect(transformer.config.vocab_size == 1000);
    try testing.expect(transformer.layer_weights.len == 2);
}

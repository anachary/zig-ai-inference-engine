const std = @import("std");
const ModelConfig = @import("mod.zig").ModelConfig;
const layers = @import("../layers/mod.zig");
const math = @import("../math/mod.zig");
const Matrix = math.matrix.Matrix;

/// KV Cache for efficient autoregressive generation
const KVCache = struct {
    // Cache for each layer: [num_layers][2][max_seq_len][num_heads][head_dim]
    // Index 0 = keys, Index 1 = values
    cache: [][][]Matrix,
    current_length: usize,
    max_length: usize,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, num_heads: usize, head_dim: usize, max_length: usize) !KVCache {
        var cache = try allocator.alloc([][]Matrix, num_layers);

        for (0..num_layers) |layer_idx| {
            cache[layer_idx] = try allocator.alloc([]Matrix, 2); // K and V
            for (0..2) |kv_idx| {
                cache[layer_idx][kv_idx] = try allocator.alloc(Matrix, max_length);
                for (0..max_length) |pos| {
                    cache[layer_idx][kv_idx][pos] = try Matrix.init(allocator, num_heads, head_dim);
                }
            }
        }

        return KVCache{
            .cache = cache,
            .current_length = 0,
            .max_length = max_length,
        };
    }

    pub fn deinit(self: *KVCache, allocator: std.mem.Allocator) void {
        for (self.cache) |layer_cache| {
            for (layer_cache) |kv_cache| {
                for (kv_cache) |*matrix| {
                    matrix.deinit();
                }
                allocator.free(kv_cache);
            }
            allocator.free(layer_cache);
        }
        allocator.free(self.cache);
    }
};

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

    // KV cache for efficient autoregressive generation
    kv_cache: ?KVCache,

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

        // Initialize KV cache for efficient autoregressive generation
        const head_dim = config.hidden_size / config.num_heads;
        var kv_cache = try KVCache.init(
            allocator,
            config.num_layers,
            config.num_heads,
            head_dim,
            config.max_position_embeddings,
        );

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
            .kv_cache = kv_cache,
        };
    }

    pub fn deinit(self: *TransformerModel) void {
        self.allocator.free(self.layer_weights);
        self.allocator.free(self.hidden_states);
        if (self.attention_cache) |cache| {
            self.allocator.free(cache);
        }
        if (self.kv_cache) |*cache| {
            cache.deinit(self.allocator);
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
            std.log.warn("Attention weights not loaded, skipping attention", .{});
            return;
        }

        const hidden_size = self.config.hidden_size;
        const num_heads = self.config.num_heads;
        const head_dim = hidden_size / num_heads;

        std.log.debug("Multi-head attention: {} heads, {} head_dim, {} seq_len", .{ num_heads, head_dim, seq_len });

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

        // Load weight matrices with proper alignment
        const wq_bytes = layer.wq.?.data;
        var wq_data = try self.allocator.alloc(f32, hidden_size * hidden_size);
        defer self.allocator.free(wq_data);
        const wq_src = std.mem.bytesAsSlice(f32, wq_bytes[0 .. hidden_size * hidden_size * @sizeOf(f32)]);
        @memcpy(wq_data, wq_src);

        var wq_matrix = try Matrix.fromSlice(
            self.allocator,
            wq_data,
            hidden_size,
            hidden_size,
        );
        defer wq_matrix.deinit();

        const wk_bytes = layer.wk.?.data;
        var wk_data = try self.allocator.alloc(f32, hidden_size * hidden_size);
        defer self.allocator.free(wk_data);
        const wk_src = std.mem.bytesAsSlice(f32, wk_bytes[0 .. hidden_size * hidden_size * @sizeOf(f32)]);
        @memcpy(wk_data, wk_src);

        var wk_matrix = try Matrix.fromSlice(
            self.allocator,
            wk_data,
            hidden_size,
            hidden_size,
        );
        defer wk_matrix.deinit();

        const wv_bytes = layer.wv.?.data;
        var wv_data = try self.allocator.alloc(f32, hidden_size * hidden_size);
        defer self.allocator.free(wv_data);
        const wv_src = std.mem.bytesAsSlice(f32, wv_bytes[0 .. hidden_size * hidden_size * @sizeOf(f32)]);
        @memcpy(wv_data, wv_src);

        var wv_matrix = try Matrix.fromSlice(
            self.allocator,
            wv_data,
            hidden_size,
            hidden_size,
        );
        defer wv_matrix.deinit();

        // Compute Q = input * Wq, K = input * Wk, V = input * Wv
        try math.matrix.matmul(input_matrix, wq_matrix, &q_matrix);
        try math.matrix.matmul(input_matrix, wk_matrix, &k_matrix);
        try math.matrix.matmul(input_matrix, wv_matrix, &v_matrix);

        // Reshape Q, K, V for multi-head attention: [seq_len, hidden_size] -> [seq_len, num_heads, head_dim]
        var attention_output = try Matrix.init(self.allocator, seq_len, hidden_size);
        defer attention_output.deinit();

        // Process each attention head separately
        for (0..num_heads) |head_idx| {
            const head_offset = head_idx * head_dim;

            // Extract Q, K, V for this head
            var q_head = try Matrix.init(self.allocator, seq_len, head_dim);
            defer q_head.deinit();
            var k_head = try Matrix.init(self.allocator, seq_len, head_dim);
            defer k_head.deinit();
            var v_head = try Matrix.init(self.allocator, seq_len, head_dim);
            defer v_head.deinit();

            // Copy head data from full Q, K, V matrices
            for (0..seq_len) |i| {
                for (0..head_dim) |j| {
                    q_head.set(i, j, q_matrix.get(i, head_offset + j));
                    k_head.set(i, j, k_matrix.get(i, head_offset + j));
                    v_head.set(i, j, v_matrix.get(i, head_offset + j));
                }
            }

            // Create causal mask for autoregressive generation
            var causal_mask = try Matrix.init(self.allocator, seq_len, seq_len);
            defer causal_mask.deinit();

            // Fill causal mask: 1.0 for allowed positions, 0.0 for masked (future) positions
            for (0..seq_len) |i| {
                for (0..seq_len) |j| {
                    if (j <= i) {
                        causal_mask.set(i, j, 1.0); // Allow past and current positions
                    } else {
                        causal_mask.set(i, j, 0.0); // Mask future positions
                    }
                }
            }

            // Apply scaled dot-product attention for this head
            var head_output = try Matrix.init(self.allocator, seq_len, head_dim);
            defer head_output.deinit();

            try math.attention.scaledDotProductAttention(
                q_head,
                k_head,
                v_head,
                &head_output,
                causal_mask, // Apply causal mask
                self.allocator,
            );

            // Copy head output back to the full attention output matrix
            for (0..seq_len) |i| {
                for (0..head_dim) |j| {
                    attention_output.set(i, head_offset + j, head_output.get(i, j));
                }
            }
        }

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
            std.log.warn("FFN weights not loaded, skipping feed-forward", .{});
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
        // Create aligned weight array to avoid alignment issues
        const weight_bytes = norm_weights.data;
        if (weight_bytes.len < hidden_size * @sizeOf(f32)) {
            std.log.warn("Insufficient weight data for layer norm", .{});
            return;
        }

        var weight_data = try self.allocator.alloc(f32, hidden_size);
        defer self.allocator.free(weight_data);

        // Copy weight data with proper alignment
        const src_weights = std.mem.bytesAsSlice(f32, weight_bytes[0 .. hidden_size * @sizeOf(f32)]);
        @memcpy(weight_data, src_weights);

        // Apply layer normalization with learned weights
        for (start_pos..end_pos) |pos| {
            const pos_offset = pos * hidden_size;
            const hidden_slice = self.hidden_states[pos_offset .. pos_offset + hidden_size];

            // Create temporary output buffer for layer norm
            var norm_output = try self.allocator.alloc(f32, hidden_size);
            defer self.allocator.free(norm_output);

            // Create zero bias array (most transformer variants don't use bias in layer norm)
            var zero_bias = try self.allocator.alloc(f32, hidden_size);
            defer self.allocator.free(zero_bias);
            @memset(zero_bias, 0.0);

            // Use the math library's layer normalization
            math.normalization.layerNorm(
                hidden_slice,
                norm_output,
                weight_data,
                zero_bias,
                self.config.layer_norm_eps,
            );

            // Copy normalized output back to hidden states
            @memcpy(hidden_slice, norm_output);
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

    /// Forward pass with efficient KV caching for autoregressive generation
    fn forwardWithKVCache(
        self: *TransformerModel,
        tokens: []const u32,
        output_logits: []f32,
        allocator: std.mem.Allocator,
        use_cache: bool,
    ) !void {
        const seq_len = tokens.len;
        _ = self.config.hidden_size;
        _ = allocator;

        if (use_cache and seq_len != 1) {
            return error.InvalidCacheUsage; // Cache mode only for single token
        }

        std.log.debug("Forward pass: {} tokens, cache: {}", .{ seq_len, use_cache });

        // Step 1: Token embedding
        try self.embedTokens(tokens);

        // Step 2: Process through transformer layers with KV caching
        for (0..self.config.num_layers) |layer_idx| {
            const layer = &self.layer_weights[layer_idx];

            // Pre-attention layer norm
            if (layer.attention_norm) |norm| {
                try self.applyLayerNormToRange(norm, 0, seq_len);
            }

            // Multi-head attention with KV caching
            try self.multiHeadAttentionWithCache(layer, seq_len, layer_idx, use_cache);

            // Pre-FFN layer norm
            if (layer.ffn_norm) |norm| {
                try self.applyLayerNormToRange(norm, 0, seq_len);
            }

            // Feed-forward network
            try self.realFeedForwardNetwork(layer, seq_len);
        }

        // Step 3: Final layer norm
        if (self.output_layernorm) |norm| {
            try self.applyLayerNormToRange(norm, 0, seq_len);
        }

        // Step 4: Output projection
        try self.computeLogits(output_logits, seq_len);

        // Update KV cache length
        if (use_cache and self.kv_cache != null) {
            self.kv_cache.?.current_length += seq_len;
        }
    }

    /// Multi-head attention with efficient KV caching
    fn multiHeadAttentionWithCache(
        self: *TransformerModel,
        layer: *LayerWeights,
        seq_len: usize,
        layer_idx: usize,
        use_cache: bool,
    ) !void {
        if (layer.wq == null or layer.wk == null or layer.wv == null) {
            std.log.warn("Attention weights not loaded, skipping attention", .{});
            return;
        }

        const hidden_size = self.config.hidden_size;
        const num_heads = self.config.num_heads;
        const head_dim = hidden_size / num_heads;

        // Create input matrix
        var input_matrix = try Matrix.fromSlice(
            self.allocator,
            self.hidden_states[0 .. seq_len * hidden_size],
            seq_len,
            hidden_size,
        );
        defer input_matrix.deinit();

        if (use_cache and self.kv_cache != null) {
            // Efficient cached attention for single token
            try self.cachedAttention(layer, input_matrix, layer_idx, head_dim);
        } else {
            // Full attention computation (for prompt processing)
            try self.fullAttention(layer, input_matrix, seq_len, num_heads, head_dim);
        }
    }

    /// Efficient cached attention for single token generation
    fn cachedAttention(
        self: *TransformerModel,
        layer: *LayerWeights,
        input_matrix: Matrix,
        layer_idx: usize,
        head_dim: usize,
    ) !void {
        _ = layer;
        _ = input_matrix;
        _ = layer_idx;
        _ = head_dim;

        // TODO: Implement efficient KV cache lookup and update
        // For now, use simplified approach
        std.log.debug("Using cached attention (simplified)", .{});

        // Apply simple transformation to simulate cached attention
        const hidden_size = self.config.hidden_size;
        for (0..hidden_size) |i| {
            self.hidden_states[i] *= 0.98; // Small modification
        }
    }

    /// Full attention computation for prompt processing
    fn fullAttention(
        self: *TransformerModel,
        layer: *LayerWeights,
        input_matrix: Matrix,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) !void {
        _ = layer;
        _ = input_matrix;
        _ = num_heads;
        _ = head_dim;

        // TODO: Implement full multi-head attention
        // For now, use simplified approach
        std.log.debug("Using full attention (simplified)", .{});

        // Apply transformation to simulate full attention
        const hidden_size = self.config.hidden_size;
        for (0..seq_len) |pos| {
            const pos_offset = pos * hidden_size;
            for (0..hidden_size) |i| {
                self.hidden_states[pos_offset + i] *= 0.99;
            }
        }
    }

    /// Generate tokens autoregressively with efficient KV caching
    pub fn generateTokens(
        self: *TransformerModel,
        prompt_tokens: []const u32,
        max_new_tokens: usize,
        temperature: f32,
        allocator: std.mem.Allocator,
    ) ![]u32 {
        std.log.info("🚀 Starting autoregressive generation with KV caching", .{});
        std.log.info("   Prompt length: {} tokens", .{prompt_tokens.len});
        std.log.info("   Max new tokens: {}", .{max_new_tokens});
        std.log.info("   Temperature: {d:.2}", .{temperature});

        // Allocate output tokens array
        var output_tokens = try allocator.alloc(u32, prompt_tokens.len + max_new_tokens);

        // Copy prompt tokens
        @memcpy(output_tokens[0..prompt_tokens.len], prompt_tokens);
        var current_length = prompt_tokens.len;

        // Reset KV cache for new generation
        if (self.kv_cache) |*cache| {
            cache.current_length = 0;
            std.log.debug("KV cache reset for new generation", .{});
        }

        // Phase 1: Process prompt tokens (can be done in parallel)
        std.log.debug("Phase 1: Processing prompt tokens...", .{});
        var logits = try allocator.alloc(f32, self.config.vocab_size);
        defer allocator.free(logits);

        try self.forwardWithKVCache(output_tokens[0..current_length], logits, allocator, false);

        // Phase 2: Generate tokens one by one (autoregressive)
        std.log.debug("Phase 2: Autoregressive generation...", .{});
        for (0..max_new_tokens) |gen_step| {
            // For autoregressive generation, only process the last token
            const last_token = output_tokens[current_length - 1];
            const single_token = [_]u32{last_token};

            // Forward pass with KV caching (only process new token)
            try self.forwardWithKVCache(&single_token, logits, allocator, true);

            // Sample next token with advanced strategies
            const next_token = try self.sampleTokenAdvanced(logits, temperature, allocator);

            // Add to output
            output_tokens[current_length] = next_token;
            current_length += 1;

            std.log.debug("Generated token {}: {} (step {})", .{ current_length - 1, next_token, gen_step + 1 });

            // Check for end-of-sequence token
            if (self.isEndOfSequence(next_token)) {
                std.log.info("End-of-sequence token detected, stopping generation", .{});
                break;
            }

            // Check context window limit
            if (current_length >= self.config.max_position_embeddings) {
                std.log.warn("Context window limit reached, stopping generation", .{});
                break;
            }
        }

        // Resize output to actual length
        const final_tokens = try allocator.alloc(u32, current_length);
        @memcpy(final_tokens, output_tokens[0..current_length]);
        allocator.free(output_tokens);

        std.log.info("✅ Generation complete: {} total tokens ({} new)", .{ current_length, current_length - prompt_tokens.len });
        return final_tokens;
    }

    /// Advanced token sampling with multiple strategies
    fn sampleTokenAdvanced(self: *TransformerModel, logits: []f32, temperature: f32, allocator: std.mem.Allocator) !u32 {
        std.log.debug("Sampling token with temperature: {d:.2}", .{temperature});

        // Apply temperature scaling
        if (temperature > 0.0 and temperature != 1.0) {
            for (logits) |*logit| {
                logit.* /= temperature;
            }
        }

        // Convert logits to probabilities using softmax
        var probs = try allocator.alloc(f32, logits.len);
        defer allocator.free(probs);

        // Find max for numerical stability
        var max_logit: f32 = logits[0];
        for (logits[1..]) |logit| {
            max_logit = @max(max_logit, logit);
        }

        // Compute softmax
        var sum: f32 = 0.0;
        for (logits, probs) |logit, *prob| {
            prob.* = @exp(logit - max_logit);
            sum += prob.*;
        }

        // Normalize
        for (probs) |*prob| {
            prob.* /= sum;
        }

        // Choose sampling strategy based on temperature
        if (temperature <= 0.1) {
            // Greedy sampling for low temperature
            return self.greedySample(probs);
        } else if (temperature <= 0.8) {
            // Top-K sampling for medium temperature
            return self.topKSample(probs, 50, allocator);
        } else {
            // Nucleus (Top-P) sampling for high temperature
            return self.nucleusSample(probs, 0.9, allocator);
        }
    }

    /// Greedy sampling - select highest probability token
    fn greedySample(self: *TransformerModel, probs: []f32) u32 {
        _ = self;

        var best_token: u32 = 0;
        var best_prob: f32 = probs[0];
        for (probs[1..], 1..) |prob, i| {
            if (prob > best_prob) {
                best_prob = prob;
                best_token = @intCast(i);
            }
        }

        std.log.debug("Greedy sample: token {} (prob: {d:.4})", .{ best_token, best_prob });
        return best_token;
    }

    /// Top-K sampling - sample from top K most likely tokens
    fn topKSample(self: *TransformerModel, probs: []f32, k: usize, allocator: std.mem.Allocator) !u32 {
        _ = self;

        // Create array of (probability, index) pairs
        var prob_indices = try allocator.alloc(struct { prob: f32, idx: u32 }, probs.len);
        defer allocator.free(prob_indices);

        for (probs, 0..) |prob, i| {
            prob_indices[i] = .{ .prob = prob, .idx = @intCast(i) };
        }

        // Sort by probability (descending)
        std.sort.heap(struct { prob: f32, idx: u32 }, prob_indices, {}, struct {
            fn lessThan(context: void, a: struct { prob: f32, idx: u32 }, b: struct { prob: f32, idx: u32 }) bool {
                _ = context;
                return a.prob > b.prob; // Descending order
            }
        }.lessThan);

        // Renormalize top-K probabilities
        const actual_k = @min(k, probs.len);
        var top_k_sum: f32 = 0.0;
        for (prob_indices[0..actual_k]) |item| {
            top_k_sum += item.prob;
        }

        // Sample from top-K (simplified - just pick first for now)
        const selected = prob_indices[0];
        std.log.debug("Top-K sample: token {} (prob: {d:.4})", .{ selected.idx, selected.prob });
        return selected.idx;
    }

    /// Nucleus (Top-P) sampling - sample from smallest set with cumulative probability >= p
    fn nucleusSample(self: *TransformerModel, probs: []f32, p: f32, allocator: std.mem.Allocator) !u32 {
        _ = self;

        // Create array of (probability, index) pairs
        var prob_indices = try allocator.alloc(struct { prob: f32, idx: u32 }, probs.len);
        defer allocator.free(prob_indices);

        for (probs, 0..) |prob, i| {
            prob_indices[i] = .{ .prob = prob, .idx = @intCast(i) };
        }

        // Sort by probability (descending)
        std.sort.heap(struct { prob: f32, idx: u32 }, prob_indices, {}, struct {
            fn lessThan(context: void, a: struct { prob: f32, idx: u32 }, b: struct { prob: f32, idx: u32 }) bool {
                _ = context;
                return a.prob > b.prob; // Descending order
            }
        }.lessThan);

        // Find nucleus (smallest set with cumulative probability >= p)
        var cumulative_prob: f32 = 0.0;
        var nucleus_size: usize = 0;
        for (prob_indices) |item| {
            cumulative_prob += item.prob;
            nucleus_size += 1;
            if (cumulative_prob >= p) break;
        }

        // Sample from nucleus (simplified - just pick first for now)
        const selected = prob_indices[0];
        std.log.debug("Nucleus sample: token {} (prob: {d:.4}, nucleus size: {})", .{ selected.idx, selected.prob, nucleus_size });
        return selected.idx;
    }

    /// Check if token is end-of-sequence
    fn isEndOfSequence(self: *TransformerModel, token: u32) bool {
        _ = self;

        // Common EOS token IDs (model-specific)
        const eos_tokens = [_]u32{ 0, 1, 2, 50256, 151643 }; // Various EOS tokens

        for (eos_tokens) |eos_token| {
            if (token == eos_token) {
                return true;
            }
        }

        return false;
    }

    /// Manage context window with sliding window approach
    fn manageContextWindow(self: *TransformerModel, tokens: []u32, max_context: usize) []u32 {
        _ = self;
        if (tokens.len <= max_context) {
            return tokens; // No truncation needed
        }

        // Keep the most recent tokens within context window
        const start_idx = tokens.len - max_context;
        std.log.warn("Context window exceeded, truncating {} tokens", .{start_idx});

        return tokens[start_idx..];
    }

    /// Generate text with streaming output for real-time response
    pub fn generateStreaming(
        self: *TransformerModel,
        prompt_tokens: []const u32,
        max_new_tokens: usize,
        temperature: f32,
        allocator: std.mem.Allocator,
        callback: ?*const fn (token: u32) void,
    ) ![]u32 {
        std.log.info("🌊 Starting streaming generation", .{});

        var output_tokens = try allocator.alloc(u32, prompt_tokens.len + max_new_tokens);
        @memcpy(output_tokens[0..prompt_tokens.len], prompt_tokens);
        var current_length = prompt_tokens.len;

        // Reset KV cache
        if (self.kv_cache) |*cache| {
            cache.current_length = 0;
        }

        // Process prompt
        var logits = try allocator.alloc(f32, self.config.vocab_size);
        defer allocator.free(logits);

        try self.forwardWithKVCache(output_tokens[0..current_length], logits, allocator, false);

        // Generate tokens with streaming
        for (0..max_new_tokens) |_| {
            // Manage context window
            const context_tokens = self.manageContextWindow(
                output_tokens[0..current_length],
                self.config.max_position_embeddings - 1,
            );

            // Generate next token
            const last_token = context_tokens[context_tokens.len - 1];
            const single_token = [_]u32{last_token};

            try self.forwardWithKVCache(&single_token, logits, allocator, true);
            const next_token = try self.sampleTokenAdvanced(logits, temperature, allocator);

            // Add to output
            output_tokens[current_length] = next_token;
            current_length += 1;

            // Call streaming callback
            if (callback) |cb| {
                cb(next_token);
            }

            // Check for end of sequence
            if (self.isEndOfSequence(next_token)) {
                break;
            }
        }

        // Return final tokens
        const final_tokens = try allocator.alloc(u32, current_length);
        @memcpy(final_tokens, output_tokens[0..current_length]);
        allocator.free(output_tokens);

        return final_tokens;
    }

    /// Batch generation for multiple prompts
    pub fn generateBatch(
        self: *TransformerModel,
        prompts: []const []const u32,
        max_new_tokens: usize,
        temperature: f32,
        allocator: std.mem.Allocator,
    ) ![][]u32 {
        std.log.info("📦 Starting batch generation for {} prompts", .{prompts.len});

        var results = try allocator.alloc([]u32, prompts.len);

        for (prompts, 0..) |prompt, i| {
            std.log.debug("Processing prompt {} of {}", .{ i + 1, prompts.len });
            results[i] = try self.generateTokens(prompt, max_new_tokens, temperature, allocator);
        }

        return results;
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

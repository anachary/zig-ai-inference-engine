const std = @import("std");
const matrix = @import("../math/matrix.zig");
const attention_math = @import("../math/attention.zig");
const normalization_math = @import("../math/normalization.zig");
const embeddings_math = @import("../math/embeddings.zig");
const linear = @import("linear.zig");
const feedforward = @import("feedforward.zig");
const normalization = @import("normalization.zig");
const attention = @import("attention.zig");
const embedding = @import("embedding.zig");

const Matrix = matrix.Matrix;
const MultiHeadAttention = attention.MultiHeadAttention;
const FeedForward = feedforward.FeedForward;
const LayerNorm = normalization.LayerNorm;
const RMSNorm = normalization.RMSNorm;

/// Single transformer block (encoder or decoder layer)
pub const TransformerBlock = struct {
    d_model: usize,
    
    // Attention components
    self_attention: MultiHeadAttention,
    cross_attention: ?MultiHeadAttention, // For decoder blocks
    
    // Feed-forward network
    feed_forward: FeedForward,
    
    // Normalization layers
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: ?LayerNorm, // For cross-attention in decoder
    
    // Configuration
    is_decoder: bool,
    pre_norm: bool, // Pre-norm vs post-norm
    
    allocator: std.mem.Allocator,
    
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        is_decoder: bool,
        pre_norm: bool,
        ff_activation: FeedForward.ActivationType,
    ) !TransformerBlock {
        var self_attention = try MultiHeadAttention.init(allocator, d_model, num_heads);
        
        var cross_attention: ?MultiHeadAttention = null;
        if (is_decoder) {
            cross_attention = try MultiHeadAttention.init(allocator, d_model, num_heads);
        }
        
        var feed_forward = try FeedForward.init(allocator, d_model, d_ff, ff_activation, true);
        
        var norm1 = try LayerNorm.init(allocator, d_model);
        var norm2 = try LayerNorm.init(allocator, d_model);
        var norm3: ?LayerNorm = null;
        if (is_decoder) {
            norm3 = try LayerNorm.init(allocator, d_model);
        }
        
        return TransformerBlock{
            .d_model = d_model,
            .self_attention = self_attention,
            .cross_attention = cross_attention,
            .feed_forward = feed_forward,
            .norm1 = norm1,
            .norm2 = norm2,
            .norm3 = norm3,
            .is_decoder = is_decoder,
            .pre_norm = pre_norm,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TransformerBlock) void {
        self.self_attention.deinit();
        if (self.cross_attention) |*cross_attn| {
            cross_attn.deinit();
        }
        self.feed_forward.deinit();
        self.norm1.deinit();
        self.norm2.deinit();
        if (self.norm3) |*norm| {
            norm.deinit();
        }
    }
    
    /// Forward pass for encoder block
    pub fn forwardEncoder(
        self: *TransformerBlock,
        input: Matrix,
        output: *Matrix,
        mask: ?Matrix,
    ) !void {
        std.debug.assert(!self.is_decoder);
        
        const batch_size = input.rows;
        
        // Temporary buffers
        var attn_output = try Matrix.init(self.allocator, batch_size, self.d_model);
        defer attn_output.deinit();
        var norm_output = try Matrix.init(self.allocator, batch_size, self.d_model);
        defer norm_output.deinit();
        var residual = try Matrix.init(self.allocator, batch_size, self.d_model);
        defer residual.deinit();
        
        if (self.pre_norm) {
            // Pre-norm: LayerNorm -> Attention -> Residual
            try self.norm1.forward(input, &norm_output);
            try self.self_attention.forward(norm_output, &attn_output, mask);
            try matrix.add(input, attn_output, &residual);
            
            // Pre-norm: LayerNorm -> FFN -> Residual
            try self.norm2.forward(residual, &norm_output);
            try self.feed_forward.forward(norm_output, &attn_output);
            try matrix.add(residual, attn_output, output);
        } else {
            // Post-norm: Attention -> Residual -> LayerNorm
            try self.self_attention.forward(input, &attn_output, mask);
            try matrix.add(input, attn_output, &residual);
            try self.norm1.forward(residual, &norm_output);
            
            // Post-norm: FFN -> Residual -> LayerNorm
            try self.feed_forward.forward(norm_output, &attn_output);
            try matrix.add(norm_output, attn_output, &residual);
            try self.norm2.forward(residual, output);
        }
    }
    
    /// Forward pass for decoder block
    pub fn forwardDecoder(
        self: *TransformerBlock,
        input: Matrix,
        encoder_output: ?Matrix,
        output: *Matrix,
        self_mask: ?Matrix,
        cross_mask: ?Matrix,
    ) !void {
        std.debug.assert(self.is_decoder);
        
        const batch_size = input.rows;
        
        // Temporary buffers
        var attn_output = try Matrix.init(self.allocator, batch_size, self.d_model);
        defer attn_output.deinit();
        var norm_output = try Matrix.init(self.allocator, batch_size, self.d_model);
        defer norm_output.deinit();
        var residual1 = try Matrix.init(self.allocator, batch_size, self.d_model);
        defer residual1.deinit();
        var residual2 = try Matrix.init(self.allocator, batch_size, self.d_model);
        defer residual2.deinit();
        
        if (self.pre_norm) {
            // Self-attention
            try self.norm1.forward(input, &norm_output);
            try self.self_attention.forward(norm_output, &attn_output, self_mask);
            try matrix.add(input, attn_output, &residual1);
            
            // Cross-attention (if encoder output provided)
            if (encoder_output != null and self.cross_attention != null) {
                try self.norm3.?.forward(residual1, &norm_output);
                try self.cross_attention.?.forwardCross(norm_output, encoder_output.?, &attn_output, cross_mask);
                try matrix.add(residual1, attn_output, &residual2);
            } else {
                residual2 = residual1;
            }
            
            // Feed-forward
            try self.norm2.forward(residual2, &norm_output);
            try self.feed_forward.forward(norm_output, &attn_output);
            try matrix.add(residual2, attn_output, output);
        } else {
            // Post-norm version (similar structure but norm after residual)
            try self.self_attention.forward(input, &attn_output, self_mask);
            try matrix.add(input, attn_output, &residual1);
            try self.norm1.forward(residual1, &norm_output);
            
            if (encoder_output != null and self.cross_attention != null) {
                try self.cross_attention.?.forwardCross(norm_output, encoder_output.?, &attn_output, cross_mask);
                try matrix.add(norm_output, attn_output, &residual2);
                try self.norm3.?.forward(residual2, &norm_output);
            } else {
                norm_output = residual1;
            }
            
            try self.feed_forward.forward(norm_output, &attn_output);
            try matrix.add(norm_output, attn_output, &residual1);
            try self.norm2.forward(residual1, output);
        }
    }
};

/// Complete transformer model
pub const TransformerModel = struct {
    config: Config,
    
    // Embedding layers
    token_embedding: embedding.TokenEmbedding,
    positional_embedding: ?embedding.PositionalEmbedding,
    
    // Transformer blocks
    encoder_blocks: ?[]TransformerBlock,
    decoder_blocks: ?[]TransformerBlock,
    
    // Output layers
    final_norm: LayerNorm,
    output_projection: linear.Linear,
    
    allocator: std.mem.Allocator,
    
    pub const Config = struct {
        vocab_size: usize,
        d_model: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        num_heads: usize,
        d_ff: usize,
        max_seq_len: usize,
        dropout_rate: f32,
        pre_norm: bool,
        ff_activation: FeedForward.ActivationType,
        use_positional_embedding: bool,
        
        pub fn encoderOnly(
            vocab_size: usize,
            d_model: usize,
            num_layers: usize,
            num_heads: usize,
            d_ff: usize,
            max_seq_len: usize,
        ) Config {
            return Config{
                .vocab_size = vocab_size,
                .d_model = d_model,
                .num_encoder_layers = num_layers,
                .num_decoder_layers = 0,
                .num_heads = num_heads,
                .d_ff = d_ff,
                .max_seq_len = max_seq_len,
                .dropout_rate = 0.1,
                .pre_norm = true,
                .ff_activation = .gelu,
                .use_positional_embedding = true,
            };
        }
        
        pub fn decoderOnly(
            vocab_size: usize,
            d_model: usize,
            num_layers: usize,
            num_heads: usize,
            d_ff: usize,
            max_seq_len: usize,
        ) Config {
            return Config{
                .vocab_size = vocab_size,
                .d_model = d_model,
                .num_encoder_layers = 0,
                .num_decoder_layers = num_layers,
                .num_heads = num_heads,
                .d_ff = d_ff,
                .max_seq_len = max_seq_len,
                .dropout_rate = 0.1,
                .pre_norm = true,
                .ff_activation = .gelu,
                .use_positional_embedding = true,
            };
        }
    };
    
    pub fn init(allocator: std.mem.Allocator, config: Config) !TransformerModel {
        var token_embedding = try embedding.TokenEmbedding.init(
            allocator,
            config.vocab_size,
            config.d_model,
        );
        
        var positional_embedding: ?embedding.PositionalEmbedding = null;
        if (config.use_positional_embedding) {
            positional_embedding = try embedding.PositionalEmbedding.init(
                allocator,
                config.max_seq_len,
                config.d_model,
            );
        }
        
        // Create encoder blocks
        var encoder_blocks: ?[]TransformerBlock = null;
        if (config.num_encoder_layers > 0) {
            encoder_blocks = try allocator.alloc(TransformerBlock, config.num_encoder_layers);
            for (encoder_blocks.?) |*block| {
                block.* = try TransformerBlock.init(
                    allocator,
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    false, // is_decoder
                    config.pre_norm,
                    config.ff_activation,
                );
            }
        }
        
        // Create decoder blocks
        var decoder_blocks: ?[]TransformerBlock = null;
        if (config.num_decoder_layers > 0) {
            decoder_blocks = try allocator.alloc(TransformerBlock, config.num_decoder_layers);
            for (decoder_blocks.?) |*block| {
                block.* = try TransformerBlock.init(
                    allocator,
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    true, // is_decoder
                    config.pre_norm,
                    config.ff_activation,
                );
            }
        }
        
        var final_norm = try LayerNorm.init(allocator, config.d_model);
        var output_projection = try linear.Linear.init(
            allocator,
            config.d_model,
            config.vocab_size,
            false, // no bias for output projection
        );
        
        return TransformerModel{
            .config = config,
            .token_embedding = token_embedding,
            .positional_embedding = positional_embedding,
            .encoder_blocks = encoder_blocks,
            .decoder_blocks = decoder_blocks,
            .final_norm = final_norm,
            .output_projection = output_projection,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TransformerModel) void {
        self.token_embedding.deinit();
        if (self.positional_embedding) |*pos_emb| {
            pos_emb.deinit();
        }
        
        if (self.encoder_blocks) |blocks| {
            for (blocks) |*block| {
                block.deinit();
            }
            self.allocator.free(blocks);
        }
        
        if (self.decoder_blocks) |blocks| {
            for (blocks) |*block| {
                block.deinit();
            }
            self.allocator.free(blocks);
        }
        
        self.final_norm.deinit();
        self.output_projection.deinit();
    }
    
    /// Forward pass for encoder-only model (like BERT)
    pub fn forwardEncoder(
        self: *TransformerModel,
        token_ids: []const u32,
        output: *Matrix,
        mask: ?Matrix,
    ) !void {
        const seq_len = token_ids.len;
        
        // Input embeddings
        var embeddings = try Matrix.init(self.allocator, seq_len, self.config.d_model);
        defer embeddings.deinit();
        
        try self.token_embedding.forward(token_ids, &embeddings);
        
        if (self.positional_embedding) |*pos_emb| {
            var pos_embeddings = try Matrix.init(self.allocator, seq_len, self.config.d_model);
            defer pos_embeddings.deinit();
            
            try pos_emb.forward(seq_len, &pos_embeddings);
            try matrix.add(embeddings, pos_embeddings, &embeddings);
        }
        
        // Pass through encoder blocks
        var current_input = embeddings;
        var temp_output = try Matrix.init(self.allocator, seq_len, self.config.d_model);
        defer temp_output.deinit();
        
        if (self.encoder_blocks) |blocks| {
            for (blocks, 0..) |*block, i| {
                const block_output = if (i % 2 == 0) &temp_output else &embeddings;
                const block_input = if (i % 2 == 0) embeddings else temp_output;
                
                try block.forwardEncoder(block_input, block_output, mask);
                current_input = block_output.*;
            }
        }
        
        // Final normalization and output projection
        try self.final_norm.forward(current_input, &temp_output);
        try self.output_projection.forward(temp_output, output);
    }
    
    /// Get total parameter count
    pub fn getParameterCount(self: *TransformerModel) usize {
        var count: usize = 0;
        
        count += self.token_embedding.getParameterCount();
        if (self.positional_embedding) |*pos_emb| {
            count += pos_emb.getParameterCount();
        }
        
        if (self.encoder_blocks) |blocks| {
            for (blocks) |*block| {
                count += block.self_attention.getParameterCount();
                count += block.feed_forward.getParameterCount();
                count += block.norm1.getParameterCount();
                count += block.norm2.getParameterCount();
            }
        }
        
        if (self.decoder_blocks) |blocks| {
            for (blocks) |*block| {
                count += block.self_attention.getParameterCount();
                if (block.cross_attention) |*cross_attn| {
                    count += cross_attn.getParameterCount();
                }
                count += block.feed_forward.getParameterCount();
                count += block.norm1.getParameterCount();
                count += block.norm2.getParameterCount();
                if (block.norm3) |*norm| {
                    count += norm.getParameterCount();
                }
            }
        }
        
        count += self.final_norm.getParameterCount();
        count += self.output_projection.getParameterCount();
        
        return count;
    }
};

test "transformer block creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var block = try TransformerBlock.init(
        allocator,
        512, // d_model
        8,   // num_heads
        2048, // d_ff
        false, // is_decoder
        true,  // pre_norm
        .gelu,
    );
    defer block.deinit();
    
    try testing.expect(block.d_model == 512);
    try testing.expect(!block.is_decoder);
    try testing.expect(block.cross_attention == null);
}

test "transformer model config" {
    const testing = std.testing;
    
    const config = TransformerModel.Config.encoderOnly(
        30000, // vocab_size
        768,   // d_model
        12,    // num_layers
        12,    // num_heads
        3072,  // d_ff
        512,   // max_seq_len
    );
    
    try testing.expect(config.vocab_size == 30000);
    try testing.expect(config.num_encoder_layers == 12);
    try testing.expect(config.num_decoder_layers == 0);
}

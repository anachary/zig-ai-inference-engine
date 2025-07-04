const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;

const ShardManager = @import("shard_manager.zig").ShardManager;
const DistributedModelConfig = @import("shard_manager.zig").DistributedModelConfig;

/// Universal model adapter for different architectures
pub const ModelAdapter = struct {
    allocator: Allocator,
    model_type: ModelType,
    architecture_config: ArchitectureConfig,
    sharding_strategy: ShardingStrategy,
    
    const Self = @This();
    
    pub const ModelType = enum {
        // Language Models
        gpt_family,           // GPT-2, GPT-3, GPT-4, GPT-J, GPT-NeoX
        llama_family,         // LLaMA, LLaMA 2, Code Llama, Alpaca
        t5_family,            // T5, UL2, PaLM, Flan-T5
        bert_family,          // BERT, RoBERTa, DeBERTa, ELECTRA
        opt_family,           // OPT, BLOOM, BigScience
        
        // Multimodal Models
        clip_family,          // CLIP, ALIGN, DALL-E
        whisper_family,       // Whisper, SpeechT5
        llava_family,         // LLaVA, GPT-4V, Flamingo
        
        // Computer Vision
        resnet_family,        // ResNet, ResNeXt, RegNet
        efficientnet_family,  // EfficientNet, EfficientDet
        vision_transformer,   // ViT, DeiT, Swin Transformer
        diffusion_models,     // Stable Diffusion, DALL-E 2
        
        // Specialized
        mixture_of_experts,   // Switch Transformer, GLaM, PaLM-2
        retrieval_augmented,  // RAG, FiD, REALM
        code_generation,      // Codex, CodeT5, StarCoder
        
        // Generic fallback
        custom_architecture,
        
        pub fn toString(self: ModelType) []const u8 {
            return switch (self) {
                .gpt_family => "GPT Family",
                .llama_family => "LLaMA Family",
                .t5_family => "T5 Family",
                .bert_family => "BERT Family",
                .opt_family => "OPT Family",
                .clip_family => "CLIP Family",
                .whisper_family => "Whisper Family",
                .llava_family => "LLaVA Family",
                .resnet_family => "ResNet Family",
                .efficientnet_family => "EfficientNet Family",
                .vision_transformer => "Vision Transformer",
                .diffusion_models => "Diffusion Models",
                .mixture_of_experts => "Mixture of Experts",
                .retrieval_augmented => "Retrieval Augmented",
                .code_generation => "Code Generation",
                .custom_architecture => "Custom Architecture",
            };
        }
    };
    
    pub const ArchitectureConfig = struct {
        // Model dimensions
        hidden_size: u32,
        num_layers: u32,
        num_attention_heads: u32,
        intermediate_size: u32,
        vocab_size: u32,
        max_position_embeddings: u32,
        
        // Architecture-specific parameters
        num_key_value_heads: ?u32 = null,  // For grouped-query attention (LLaMA 2)
        rope_theta: ?f32 = null,           // For RoPE embeddings
        sliding_window: ?u32 = null,       // For sliding window attention
        num_experts: ?u32 = null,          // For MoE models
        expert_capacity: ?u32 = null,      // For MoE models
        
        // Multimodal parameters
        vision_config: ?VisionConfig = null,
        audio_config: ?AudioConfig = null,
        
        // Memory optimization
        use_gradient_checkpointing: bool = false,
        use_flash_attention: bool = true,
        use_kv_cache: bool = true,
        
        pub const VisionConfig = struct {
            image_size: u32,
            patch_size: u32,
            num_channels: u32,
            projection_dim: u32,
        };
        
        pub const AudioConfig = struct {
            sample_rate: u32,
            n_fft: u32,
            hop_length: u32,
            n_mels: u32,
        };
    };
    
    pub const ShardingStrategy = enum {
        layer_wise,           // Split by transformer layers (most common)
        tensor_parallel,      // Split attention/MLP within layers
        pipeline_parallel,    // Sequential layer execution
        expert_parallel,      // Split experts in MoE models
        hybrid_parallel,      // Combination of strategies
        custom_sharding,      // User-defined sharding
        
        pub fn toString(self: ShardingStrategy) []const u8 {
            return switch (self) {
                .layer_wise => "Layer-wise Sharding",
                .tensor_parallel => "Tensor Parallel",
                .pipeline_parallel => "Pipeline Parallel",
                .expert_parallel => "Expert Parallel",
                .hybrid_parallel => "Hybrid Parallel",
                .custom_sharding => "Custom Sharding",
            };
        }
    };
    
    pub fn init(allocator: Allocator, model_type: ModelType) Self {
        return Self{
            .allocator = allocator,
            .model_type = model_type,
            .architecture_config = getDefaultConfig(model_type),
            .sharding_strategy = getOptimalShardingStrategy(model_type),
        };
    }
    
    /// Get default configuration for model type
    fn getDefaultConfig(model_type: ModelType) ArchitectureConfig {
        return switch (model_type) {
            .gpt_family => ArchitectureConfig{
                .hidden_size = 12288,      // GPT-3 175B
                .num_layers = 96,
                .num_attention_heads = 96,
                .intermediate_size = 49152,
                .vocab_size = 50257,
                .max_position_embeddings = 2048,
                .use_flash_attention = true,
                .use_kv_cache = true,
            },
            .llama_family => ArchitectureConfig{
                .hidden_size = 8192,       // LLaMA 2 70B
                .num_layers = 80,
                .num_attention_heads = 64,
                .intermediate_size = 28672,
                .vocab_size = 32000,
                .max_position_embeddings = 4096,
                .num_key_value_heads = 8,  // Grouped-query attention
                .rope_theta = 10000.0,
                .use_flash_attention = true,
            },
            .t5_family => ArchitectureConfig{
                .hidden_size = 4096,       // T5-11B
                .num_layers = 24,
                .num_attention_heads = 64,
                .intermediate_size = 10240,
                .vocab_size = 32128,
                .max_position_embeddings = 512,
                .use_flash_attention = true,
            },
            .clip_family => ArchitectureConfig{
                .hidden_size = 1024,       // CLIP ViT-L/14
                .num_layers = 24,
                .num_attention_heads = 16,
                .intermediate_size = 4096,
                .vocab_size = 49408,
                .max_position_embeddings = 77,
                .vision_config = ArchitectureConfig.VisionConfig{
                    .image_size = 224,
                    .patch_size = 14,
                    .num_channels = 3,
                    .projection_dim = 768,
                },
            },
            .mixture_of_experts => ArchitectureConfig{
                .hidden_size = 4096,       // Switch Transformer
                .num_layers = 32,
                .num_attention_heads = 32,
                .intermediate_size = 16384,
                .vocab_size = 32000,
                .max_position_embeddings = 2048,
                .num_experts = 128,
                .expert_capacity = 64,
            },
            else => ArchitectureConfig{
                .hidden_size = 4096,       // Generic large model
                .num_layers = 32,
                .num_attention_heads = 32,
                .intermediate_size = 16384,
                .vocab_size = 50000,
                .max_position_embeddings = 2048,
            },
        };
    }
    
    /// Get optimal sharding strategy for model type
    fn getOptimalShardingStrategy(model_type: ModelType) ShardingStrategy {
        return switch (model_type) {
            .gpt_family, .llama_family, .t5_family, .bert_family => .layer_wise,
            .mixture_of_experts => .expert_parallel,
            .clip_family, .llava_family => .hybrid_parallel,
            .diffusion_models => .tensor_parallel,
            else => .layer_wise,
        };
    }
    
    /// Create distributed model configuration
    pub fn createDistributedConfig(self: *Self, total_shards: u32, memory_per_shard_gb: u64) !DistributedModelConfig {
        const layers_per_shard = self.calculateLayersPerShard(total_shards);
        
        return DistributedModelConfig{
            .model_path = "", // To be set by caller
            .total_layers = self.architecture_config.num_layers,
            .shards_count = total_shards,
            .max_shard_memory_mb = memory_per_shard_gb * 1024,
            .replication_factor = self.getReplicationFactor(),
            .load_balancing_strategy = self.getLoadBalancingStrategy(),
        };
    }
    
    /// Calculate optimal layers per shard
    fn calculateLayersPerShard(self: *Self, total_shards: u32) u32 {
        return switch (self.sharding_strategy) {
            .layer_wise => self.architecture_config.num_layers / total_shards,
            .tensor_parallel => 1, // Each shard handles part of each layer
            .pipeline_parallel => self.architecture_config.num_layers / total_shards,
            .expert_parallel => {
                if (self.architecture_config.num_experts) |experts| {
                    return experts / total_shards;
                }
                return self.architecture_config.num_layers / total_shards;
            },
            .hybrid_parallel => self.architecture_config.num_layers / (total_shards / 2),
            .custom_sharding => self.architecture_config.num_layers / total_shards,
        };
    }
    
    /// Get replication factor based on model criticality
    fn getReplicationFactor(self: *Self) u8 {
        return switch (self.model_type) {
            .gpt_family, .llama_family => 2, // High-value models need replication
            .mixture_of_experts => 1,        // MoE models are naturally redundant
            .custom_architecture => 1,       // Conservative for unknown models
            else => 2,                       // Default to replication
        };
    }
    
    /// Get load balancing strategy
    fn getLoadBalancingStrategy(self: *Self) DistributedModelConfig.LoadBalancingStrategy {
        return switch (self.model_type) {
            .mixture_of_experts => .weighted,      // MoE needs expert-aware balancing
            .diffusion_models => .least_loaded,    // Diffusion has variable compute
            else => .round_robin,                  // Simple and effective for most
        };
    }
    
    /// Estimate memory requirements
    pub fn estimateMemoryRequirements(self: *Self) MemoryEstimate {
        const params_per_layer = self.estimateParametersPerLayer();
        const total_params = params_per_layer * self.architecture_config.num_layers;
        
        // Memory calculation (rough estimates)
        const model_memory_gb = (total_params * 4) / (1024 * 1024 * 1024); // FP32
        const activation_memory_gb = model_memory_gb / 4; // Rough estimate
        const total_memory_gb = model_memory_gb + activation_memory_gb;
        
        return MemoryEstimate{
            .total_parameters = total_params,
            .model_memory_gb = model_memory_gb,
            .activation_memory_gb = activation_memory_gb,
            .total_memory_gb = total_memory_gb,
            .recommended_shards = @max(1, total_memory_gb / 32), // 32GB per shard
        };
    }
    
    /// Estimate parameters per layer
    fn estimateParametersPerLayer(self: *Self) u64 {
        const config = &self.architecture_config;
        
        // Attention parameters
        const attention_params = 4 * config.hidden_size * config.hidden_size; // Q, K, V, O projections
        
        // MLP parameters
        const mlp_params = 2 * config.hidden_size * config.intermediate_size; // Up and down projections
        
        // Layer norm parameters
        const norm_params = 2 * config.hidden_size; // Weight and bias
        
        var total_per_layer = attention_params + mlp_params + norm_params;
        
        // Add embedding parameters (only for first layer estimate)
        if (self.model_type == .gpt_family or self.model_type == .llama_family) {
            total_per_layer += config.vocab_size * config.hidden_size / config.num_layers;
        }
        
        return total_per_layer;
    }
    
    pub const MemoryEstimate = struct {
        total_parameters: u64,
        model_memory_gb: u64,
        activation_memory_gb: u64,
        total_memory_gb: u64,
        recommended_shards: u64,
    };
    
    /// Get model-specific optimization hints
    pub fn getOptimizationHints(self: *Self) OptimizationHints {
        return switch (self.model_type) {
            .gpt_family => OptimizationHints{
                .use_kv_cache = true,
                .use_flash_attention = true,
                .use_gradient_checkpointing = true,
                .preferred_precision = .fp16,
                .batch_size_hint = 1,
            },
            .llama_family => OptimizationHints{
                .use_kv_cache = true,
                .use_flash_attention = true,
                .use_gradient_checkpointing = false, // RMSNorm is memory efficient
                .preferred_precision = .fp16,
                .batch_size_hint = 1,
            },
            .mixture_of_experts => OptimizationHints{
                .use_kv_cache = true,
                .use_flash_attention = true,
                .use_gradient_checkpointing = true,
                .preferred_precision = .fp16,
                .batch_size_hint = 8, // MoE benefits from larger batches
            },
            .diffusion_models => OptimizationHints{
                .use_kv_cache = false,
                .use_flash_attention = true,
                .use_gradient_checkpointing = true,
                .preferred_precision = .fp32, // Diffusion needs higher precision
                .batch_size_hint = 4,
            },
            else => OptimizationHints{
                .use_kv_cache = true,
                .use_flash_attention = true,
                .use_gradient_checkpointing = true,
                .preferred_precision = .fp16,
                .batch_size_hint = 1,
            },
        };
    }
    
    pub const OptimizationHints = struct {
        use_kv_cache: bool,
        use_flash_attention: bool,
        use_gradient_checkpointing: bool,
        preferred_precision: Precision,
        batch_size_hint: u32,
        
        pub const Precision = enum {
            fp32,
            fp16,
            bf16,
            int8,
            int4,
        };
    };
};

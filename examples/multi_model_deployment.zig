const std = @import("std");
const print = std.debug.print;

const ModelAdapter = @import("../src/distributed/model_adapter.zig").ModelAdapter;
const ShardManager = @import("../src/distributed/shard_manager.zig").ShardManager;
const InferenceCoordinator = @import("../src/distributed/inference_coordinator.zig").InferenceCoordinator;

/// Example: Deploying different massive model architectures
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🌟 Multi-Model Distributed Deployment Examples\n");
    print("==============================================\n\n");

    // Example 1: LLaMA 2 70B
    try deployLLaMA2(allocator);
    
    // Example 2: Stable Diffusion XL
    try deployStableDiffusion(allocator);
    
    // Example 3: Whisper Large V3
    try deployWhisper(allocator);
    
    // Example 4: Mixture of Experts (Switch Transformer)
    try deployMixtureOfExperts(allocator);
    
    // Example 5: Custom Architecture
    try deployCustomModel(allocator);

    print("\n🎉 All model deployment examples completed!\n");
}

/// Deploy LLaMA 2 70B model
fn deployLLaMA2(allocator: std.mem.Allocator) !void {
    print("🦙 Deploying LLaMA 2 70B Model\n");
    print("==============================\n");

    // Initialize model adapter for LLaMA family
    var adapter = ModelAdapter.init(allocator, .llama_family);
    
    // Configure for LLaMA 2 70B specifically
    adapter.architecture_config = .{
        .hidden_size = 8192,
        .num_layers = 80,
        .num_attention_heads = 64,
        .intermediate_size = 28672,
        .vocab_size = 32000,
        .max_position_embeddings = 4096,
        .num_key_value_heads = 8,  // Grouped-query attention
        .rope_theta = 10000.0,
        .use_flash_attention = true,
        .use_kv_cache = true,
    };

    // Estimate memory requirements
    const memory_estimate = adapter.estimateMemoryRequirements();
    print("📊 Memory Requirements:\n");
    print("   - Total parameters: {d:.1}B\n", .{@as(f64, @floatFromInt(memory_estimate.total_parameters)) / 1e9});
    print("   - Model memory: {d} GB\n", .{memory_estimate.model_memory_gb});
    print("   - Total memory: {d} GB\n", .{memory_estimate.total_memory_gb});
    print("   - Recommended shards: {d}\n", .{memory_estimate.recommended_shards});

    // Create distributed configuration
    const total_shards = 8;
    const memory_per_shard = 32; // 32GB per shard
    var config = try adapter.createDistributedConfig(total_shards, memory_per_shard);
    config.model_path = "models/llama2-70b.onnx";

    print("\n🔧 Distributed Configuration:\n");
    print("   - Sharding strategy: {s}\n", .{adapter.sharding_strategy.toString()});
    print("   - Total shards: {d}\n", .{config.shards_count});
    print("   - Layers per shard: {d}\n", .{config.total_layers / config.shards_count});
    print("   - Memory per shard: {d} GB\n", .{config.max_shard_memory_mb / 1024});

    // Get optimization hints
    const hints = adapter.getOptimizationHints();
    print("\n⚡ Optimization Hints:\n");
    print("   - Use KV cache: {s}\n", .{if (hints.use_kv_cache) "Yes" else "No"});
    print("   - Use Flash Attention: {s}\n", .{if (hints.use_flash_attention) "Yes" else "No"});
    print("   - Preferred precision: {s}\n", .{@tagName(hints.preferred_precision)});
    print("   - Recommended batch size: {d}\n", .{hints.batch_size_hint});

    // Simulate deployment
    print("\n🚀 Deployment Commands:\n");
    print("   helm install llama2-70b ./deploy/aks/helm/zig-ai \\\n");
    print("     --set model.type=llama_family \\\n");
    print("     --set model.path=models/llama2-70b.onnx \\\n");
    print("     --set shard.replicas={d} \\\n", .{total_shards});
    print("     --set shard.resources.requests.memory=32Gi \\\n");
    print("     --set shard.resources.requests.nvidia\\.com/gpu=1\n");

    print("\n✅ LLaMA 2 70B deployment configured!\n\n");
}

/// Deploy Stable Diffusion XL model
fn deployStableDiffusion(allocator: std.mem.Allocator) !void {
    print("🎨 Deploying Stable Diffusion XL Model\n");
    print("=======================================\n");

    var adapter = ModelAdapter.init(allocator, .diffusion_models);
    
    // Configure for Stable Diffusion XL
    adapter.architecture_config = .{
        .hidden_size = 2048,
        .num_layers = 32,
        .num_attention_heads = 32,
        .intermediate_size = 8192,
        .vocab_size = 49408,
        .max_position_embeddings = 77,
        .vision_config = .{
            .image_size = 1024,
            .patch_size = 16,
            .num_channels = 4,
            .projection_dim = 2048,
        },
        .use_flash_attention = true,
        .use_gradient_checkpointing = true,
    };

    // Use tensor parallel sharding for diffusion models
    adapter.sharding_strategy = .tensor_parallel;

    const memory_estimate = adapter.estimateMemoryRequirements();
    print("📊 Memory Requirements:\n");
    print("   - Total parameters: {d:.1}B\n", .{@as(f64, @floatFromInt(memory_estimate.total_parameters)) / 1e9});
    print("   - Model memory: {d} GB\n", .{memory_estimate.model_memory_gb});
    print("   - Recommended shards: {d}\n", .{memory_estimate.recommended_shards});

    const total_shards = 4;
    const memory_per_shard = 16;
    var config = try adapter.createDistributedConfig(total_shards, memory_per_shard);
    config.model_path = "models/stable-diffusion-xl.onnx";

    print("\n🔧 Configuration:\n");
    print("   - Sharding strategy: {s}\n", .{adapter.sharding_strategy.toString()});
    print("   - Image resolution: {d}x{d}\n", .{ adapter.architecture_config.vision_config.?.image_size, adapter.architecture_config.vision_config.?.image_size });
    print("   - Shards: {d}\n", .{total_shards});

    print("\n🚀 Deployment Commands:\n");
    print("   helm install stable-diffusion ./deploy/aks/helm/zig-ai \\\n");
    print("     --set model.type=diffusion_models \\\n");
    print("     --set model.path=models/stable-diffusion-xl.onnx \\\n");
    print("     --set shard.replicas={d} \\\n", .{total_shards});
    print("     --set shard.resources.requests.memory=16Gi\n");

    print("\n✅ Stable Diffusion XL deployment configured!\n\n");
}

/// Deploy Whisper Large V3 model
fn deployWhisper(allocator: std.mem.Allocator) !void {
    print("🎤 Deploying Whisper Large V3 Model\n");
    print("====================================\n");

    var adapter = ModelAdapter.init(allocator, .whisper_family);
    
    // Configure for Whisper Large V3
    adapter.architecture_config = .{
        .hidden_size = 1280,
        .num_layers = 32,
        .num_attention_heads = 20,
        .intermediate_size = 5120,
        .vocab_size = 51865,
        .max_position_embeddings = 1500,
        .audio_config = .{
            .sample_rate = 16000,
            .n_fft = 400,
            .hop_length = 160,
            .n_mels = 80,
        },
        .use_flash_attention = true,
    };

    const memory_estimate = adapter.estimateMemoryRequirements();
    print("📊 Memory Requirements:\n");
    print("   - Total parameters: {d:.1}B\n", .{@as(f64, @floatFromInt(memory_estimate.total_parameters)) / 1e9});
    print("   - Model memory: {d} GB\n", .{memory_estimate.model_memory_gb});

    const total_shards = 2;
    const memory_per_shard = 8;
    var config = try adapter.createDistributedConfig(total_shards, memory_per_shard);
    config.model_path = "models/whisper-large-v3.onnx";

    print("\n🔧 Configuration:\n");
    print("   - Audio sample rate: {d} Hz\n", .{adapter.architecture_config.audio_config.?.sample_rate});
    print("   - Mel spectrogram bins: {d}\n", .{adapter.architecture_config.audio_config.?.n_mels});
    print("   - Shards: {d}\n", .{total_shards});

    print("\n🚀 Deployment Commands:\n");
    print("   helm install whisper-large ./deploy/aks/helm/zig-ai \\\n");
    print("     --set model.type=whisper_family \\\n");
    print("     --set model.path=models/whisper-large-v3.onnx \\\n");
    print("     --set shard.replicas={d} \\\n", .{total_shards});
    print("     --set shard.resources.requests.memory=8Gi\n");

    print("\n✅ Whisper Large V3 deployment configured!\n\n");
}

/// Deploy Mixture of Experts model
fn deployMixtureOfExperts(allocator: std.mem.Allocator) !void {
    print("🧠 Deploying Switch Transformer (MoE) Model\n");
    print("============================================\n");

    var adapter = ModelAdapter.init(allocator, .mixture_of_experts);
    
    // Configure for Switch Transformer
    adapter.architecture_config = .{
        .hidden_size = 4096,
        .num_layers = 32,
        .num_attention_heads = 32,
        .intermediate_size = 16384,
        .vocab_size = 32000,
        .max_position_embeddings = 2048,
        .num_experts = 128,
        .expert_capacity = 64,
        .use_flash_attention = true,
    };

    // Use expert parallel sharding for MoE
    adapter.sharding_strategy = .expert_parallel;

    const memory_estimate = adapter.estimateMemoryRequirements();
    print("📊 Memory Requirements:\n");
    print("   - Total parameters: {d:.1}B\n", .{@as(f64, @floatFromInt(memory_estimate.total_parameters)) / 1e9});
    print("   - Model memory: {d} GB\n", .{memory_estimate.model_memory_gb});

    const total_shards = 16; // More shards for expert parallelism
    const memory_per_shard = 24;
    var config = try adapter.createDistributedConfig(total_shards, memory_per_shard);
    config.model_path = "models/switch-transformer.onnx";

    print("\n🔧 Configuration:\n");
    print("   - Sharding strategy: {s}\n", .{adapter.sharding_strategy.toString()});
    print("   - Number of experts: {d}\n", .{adapter.architecture_config.num_experts.?});
    print("   - Experts per shard: {d}\n", .{adapter.architecture_config.num_experts.? / total_shards});
    print("   - Shards: {d}\n", .{total_shards});

    print("\n🚀 Deployment Commands:\n");
    print("   helm install switch-transformer ./deploy/aks/helm/zig-ai \\\n");
    print("     --set model.type=mixture_of_experts \\\n");
    print("     --set model.path=models/switch-transformer.onnx \\\n");
    print("     --set shard.replicas={d} \\\n", .{total_shards});
    print("     --set shard.resources.requests.memory=24Gi\n");

    print("\n✅ Switch Transformer (MoE) deployment configured!\n\n");
}

/// Deploy custom architecture model
fn deployCustomModel(allocator: std.mem.Allocator) !void {
    print("🔧 Deploying Custom Architecture Model\n");
    print("=======================================\n");

    var adapter = ModelAdapter.init(allocator, .custom_architecture);
    
    // Configure custom architecture (example: large vision-language model)
    adapter.architecture_config = .{
        .hidden_size = 6144,
        .num_layers = 48,
        .num_attention_heads = 48,
        .intermediate_size = 24576,
        .vocab_size = 65536,
        .max_position_embeddings = 4096,
        .vision_config = .{
            .image_size = 512,
            .patch_size = 16,
            .num_channels = 3,
            .projection_dim = 6144,
        },
        .use_flash_attention = true,
        .use_kv_cache = true,
    };

    // Use hybrid parallel for multimodal models
    adapter.sharding_strategy = .hybrid_parallel;

    const memory_estimate = adapter.estimateMemoryRequirements();
    print("📊 Memory Requirements:\n");
    print("   - Total parameters: {d:.1}B\n", .{@as(f64, @floatFromInt(memory_estimate.total_parameters)) / 1e9});
    print("   - Model memory: {d} GB\n", .{memory_estimate.model_memory_gb});

    const total_shards = 12;
    const memory_per_shard = 40;
    var config = try adapter.createDistributedConfig(total_shards, memory_per_shard);
    config.model_path = "models/custom-multimodal.onnx";

    print("\n🔧 Configuration:\n");
    print("   - Sharding strategy: {s}\n", .{adapter.sharding_strategy.toString()});
    print("   - Hidden size: {d}\n", .{adapter.architecture_config.hidden_size});
    print("   - Vision support: {s}\n", .{if (adapter.architecture_config.vision_config != null) "Yes" else "No"});
    print("   - Shards: {d}\n", .{total_shards});

    print("\n🚀 Deployment Commands:\n");
    print("   helm install custom-model ./deploy/aks/helm/zig-ai \\\n");
    print("     --set model.type=custom_architecture \\\n");
    print("     --set model.path=models/custom-multimodal.onnx \\\n");
    print("     --set shard.replicas={d} \\\n", .{total_shards});
    print("     --set shard.resources.requests.memory=40Gi\n");

    print("\n✅ Custom architecture deployment configured!\n\n");
}

/// Example usage patterns for different models
fn demonstrateUsagePatterns() !void {
    print("🎯 Usage Patterns for Different Models\n");
    print("======================================\n");

    print("\n1. 🦙 LLaMA 2 70B - Text Generation:\n");
    print("   curl -X POST http://coordinator/api/v1/inference \\\n");
    print("     -d '{{\"prompt\": \"Explain quantum computing\", \"max_tokens\": 500}}'\n");

    print("\n2. 🎨 Stable Diffusion XL - Image Generation:\n");
    print("   curl -X POST http://coordinator/api/v1/generate \\\n");
    print("     -d '{{\"prompt\": \"A futuristic city at sunset\", \"width\": 1024, \"height\": 1024}}'\n");

    print("\n3. 🎤 Whisper Large V3 - Speech Recognition:\n");
    print("   curl -X POST http://coordinator/api/v1/transcribe \\\n");
    print("     -F \"audio=@speech.wav\" -F \"language=en\"\n");

    print("\n4. 🧠 Switch Transformer - Efficient Text Generation:\n");
    print("   curl -X POST http://coordinator/api/v1/inference \\\n");
    print("     -d '{{\"prompt\": \"Write code for\", \"max_tokens\": 200, \"use_experts\": true}}'\n");

    print("\n5. 🔧 Custom Multimodal - Vision + Language:\n");
    print("   curl -X POST http://coordinator/api/v1/multimodal \\\n");
    print("     -F \"image=@photo.jpg\" -F \"prompt=Describe this image\"\n");
}

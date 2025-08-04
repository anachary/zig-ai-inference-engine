const std = @import("std");

// Memory management modules
pub const arena = @import("arena.zig");
pub const pool = @import("pool.zig");
pub const cache = @import("cache.zig");

// Re-export commonly used types
pub const ArenaAllocator = arena.ArenaAllocator;
pub const MemoryPool = pool.MemoryPool;
pub const TensorCache = cache.TensorCache;

/// Memory configuration for inference
pub const MemoryConfig = struct {
    arena_size: usize,
    pool_size: usize,
    cache_size: usize,
    alignment: usize,

    pub fn default() MemoryConfig {
        return MemoryConfig{
            .arena_size = 1024 * 1024 * 1024, // 1GB
            .pool_size = 512 * 1024 * 1024, // 512MB
            .cache_size = 256 * 1024 * 1024, // 256MB
            .alignment = 32, // 32-byte alignment for SIMD
        };
    }

    pub fn forModel(vocab_size: usize, d_model: usize, num_layers: usize, max_seq_len: usize) MemoryConfig {
        // Estimate memory requirements based on model size
        const embedding_size = vocab_size * d_model * @sizeOf(f32);
        const layer_size = d_model * d_model * 4 * @sizeOf(f32); // Rough estimate
        const kv_cache_size = num_layers * max_seq_len * d_model * 2 * @sizeOf(f32);

        const total_model_size = embedding_size + layer_size * num_layers;
        const arena_size = total_model_size * 2; // 2x for intermediate computations
        const cache_size = kv_cache_size * 2; // 2x for safety

        return MemoryConfig{
            .arena_size = @max(arena_size, 512 * 1024 * 1024),
            .pool_size = @max(total_model_size, 256 * 1024 * 1024),
            .cache_size = @max(cache_size, 128 * 1024 * 1024),
            .alignment = 32,
        };
    }
};

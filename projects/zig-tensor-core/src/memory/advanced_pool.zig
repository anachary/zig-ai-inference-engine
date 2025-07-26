const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");

/// Advanced memory pool with defragmentation and smart allocation
pub const AdvancedTensorPool = struct {
    allocator: Allocator,
    
    // Size-based pools with different strategies
    small_pool: SizePool,      // < 1KB tensors
    medium_pool: SizePool,     // 1KB - 1MB tensors  
    large_pool: SizePool,      // > 1MB tensors
    
    // Memory tracking
    total_allocated: usize,
    total_freed: usize,
    peak_usage: usize,
    fragmentation_ratio: f32,
    
    // Configuration
    max_pool_size: usize,
    defrag_threshold: f32,
    enable_defragmentation: bool,
    
    // Statistics
    stats: AdvancedPoolStats,
    
    const Self = @This();
    
    const SMALL_THRESHOLD = 1024;        // 1KB
    const MEDIUM_THRESHOLD = 1024 * 1024; // 1MB
    
    pub fn init(allocator: Allocator, config: PoolConfig) Self {
        return Self{
            .allocator = allocator,
            .small_pool = SizePool.init(allocator, config.small_pool_size),
            .medium_pool = SizePool.init(allocator, config.medium_pool_size),
            .large_pool = SizePool.init(allocator, config.large_pool_size),
            .total_allocated = 0,
            .total_freed = 0,
            .peak_usage = 0,
            .fragmentation_ratio = 0.0,
            .max_pool_size = config.max_pool_size,
            .defrag_threshold = config.defrag_threshold,
            .enable_defragmentation = config.enable_defragmentation,
            .stats = AdvancedPoolStats{},
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.small_pool.deinit();
        self.medium_pool.deinit();
        self.large_pool.deinit();
    }
    
    /// Get tensor with smart allocation strategy
    pub fn getTensor(self: *Self, shape: []const usize, dtype: tensor.DataType) !tensor.Tensor {
        const size_bytes = self.calculateTensorSize(shape, dtype);
        
        // Choose appropriate pool based on size
        var pool = self.selectPool(size_bytes);
        
        // Try to get from pool first
        if (try pool.getTensor(shape, dtype)) |pooled_tensor| {
            self.stats.cache_hits += 1;
            self.updateUsageStats(size_bytes, true);
            return pooled_tensor;
        }
        
        // Check if defragmentation is needed
        if (self.enable_defragmentation and self.shouldDefragment()) {
            try self.defragmentPool(pool);
            
            // Try again after defragmentation
            if (try pool.getTensor(shape, dtype)) |pooled_tensor| {
                self.stats.cache_hits += 1;
                self.stats.defragmentations += 1;
                self.updateUsageStats(size_bytes, true);
                return pooled_tensor;
            }
        }
        
        // Create new tensor
        self.stats.cache_misses += 1;
        self.updateUsageStats(size_bytes, true);
        return tensor.Tensor.init(self.allocator, shape, dtype);
    }
    
    /// Return tensor to appropriate pool
    pub fn returnTensor(self: *Self, t: tensor.Tensor) !void {
        const size_bytes = self.calculateTensorSizeFromTensor(&t);
        var pool = self.selectPool(size_bytes);
        
        try pool.returnTensor(t);
        self.updateUsageStats(size_bytes, false);
        
        // Update fragmentation ratio
        self.updateFragmentationRatio();
    }
    
    /// Force defragmentation of all pools
    pub fn defragmentAll(self: *Self) !void {
        try self.defragmentPool(&self.small_pool);
        try self.defragmentPool(&self.medium_pool);
        try self.defragmentPool(&self.large_pool);
        self.stats.defragmentations += 1;
    }
    
    /// Get detailed statistics
    pub fn getStats(self: *Self) AdvancedPoolStats {
        var stats = self.stats;
        stats.total_allocated = self.total_allocated;
        stats.total_freed = self.total_freed;
        stats.peak_usage = self.peak_usage;
        stats.fragmentation_ratio = self.fragmentation_ratio;
        stats.current_usage = self.total_allocated - self.total_freed;
        
        // Pool-specific stats
        stats.small_pool_usage = self.small_pool.getCurrentUsage();
        stats.medium_pool_usage = self.medium_pool.getCurrentUsage();
        stats.large_pool_usage = self.large_pool.getCurrentUsage();
        
        return stats;
    }
    
    // Private methods
    
    fn selectPool(self: *Self, size_bytes: usize) *SizePool {
        if (size_bytes < SMALL_THRESHOLD) {
            return &self.small_pool;
        } else if (size_bytes < MEDIUM_THRESHOLD) {
            return &self.medium_pool;
        } else {
            return &self.large_pool;
        }
    }
    
    fn shouldDefragment(self: *Self) bool {
        return self.fragmentation_ratio > self.defrag_threshold;
    }
    
    fn defragmentPool(self: *Self, pool: *SizePool) !void {
        // Compact pool by removing empty slots and reorganizing
        try pool.compact();
        std.log.info("Defragmented pool, reclaimed memory", .{});
    }
    
    fn updateUsageStats(self: *Self, size_bytes: usize, is_allocation: bool) void {
        if (is_allocation) {
            self.total_allocated += size_bytes;
            self.peak_usage = @max(self.peak_usage, self.total_allocated - self.total_freed);
        } else {
            self.total_freed += size_bytes;
        }
    }
    
    fn updateFragmentationRatio(self: *Self) void {
        const current_usage = self.total_allocated - self.total_freed;
        if (current_usage > 0) {
            const wasted_space = self.calculateWastedSpace();
            self.fragmentation_ratio = @as(f32, @floatFromInt(wasted_space)) / @as(f32, @floatFromInt(current_usage));
        } else {
            self.fragmentation_ratio = 0.0;
        }
    }
    
    fn calculateWastedSpace(self: *Self) usize {
        return self.small_pool.getWastedSpace() + 
               self.medium_pool.getWastedSpace() + 
               self.large_pool.getWastedSpace();
    }
    
    fn calculateTensorSize(self: *Self, shape: []const usize, dtype: tensor.DataType) usize {
        _ = self;
        var total_elements: usize = 1;
        for (shape) |dim| {
            total_elements *= dim;
        }
        
        const element_size: usize = switch (dtype) {
            .f32 => 4,
            .f16 => 2,
            .i32 => 4,
            .i16 => 2,
            .i8 => 1,
            .u8 => 1,
        };
        
        return total_elements * element_size;
    }
    
    fn calculateTensorSizeFromTensor(self: *Self, t: *const tensor.Tensor) usize {
        return self.calculateTensorSize(t.shape, t.dtype);
    }
};

/// Size-specific pool implementation
const SizePool = struct {
    allocator: Allocator,
    tensors: std.ArrayList(tensor.Tensor),
    max_size: usize,
    current_usage: usize,
    wasted_space: usize,
    
    fn init(allocator: Allocator, max_size: usize) SizePool {
        return SizePool{
            .allocator = allocator,
            .tensors = std.ArrayList(tensor.Tensor).init(allocator),
            .max_size = max_size,
            .current_usage = 0,
            .wasted_space = 0,
        };
    }
    
    fn deinit(self: *SizePool) void {
        for (self.tensors.items) |*t| {
            var mutable_t = t.*;
            mutable_t.deinit();
        }
        self.tensors.deinit();
    }
    
    fn getTensor(self: *SizePool, shape: []const usize, dtype: tensor.DataType) !?tensor.Tensor {
        // Look for compatible tensor in pool
        for (self.tensors.items, 0..) |pooled_tensor, i| {
            if (self.isCompatible(&pooled_tensor, shape, dtype)) {
                const result = self.tensors.swapRemove(i);
                self.current_usage -= 1;
                return result;
            }
        }
        return null;
    }
    
    fn returnTensor(self: *SizePool, t: tensor.Tensor) !void {
        if (self.tensors.items.len < self.max_size) {
            // Zero out data for security
            @memset(t.data, 0);
            try self.tensors.append(t);
            self.current_usage += 1;
        } else {
            // Pool is full, free the tensor
            var mutable_t = t;
            mutable_t.deinit();
        }
    }
    
    fn compact(self: *SizePool) !void {
        // Remove any invalid tensors and compact the array
        var write_index: usize = 0;
        for (self.tensors.items) |pooled_tensor| {
            if (pooled_tensor.data.len > 0) { // Valid tensor
                self.tensors.items[write_index] = pooled_tensor;
                write_index += 1;
            }
        }
        self.tensors.shrinkRetainingCapacity(write_index);
        self.wasted_space = 0;
    }
    
    fn getCurrentUsage(self: *SizePool) usize {
        return self.current_usage;
    }
    
    fn getWastedSpace(self: *SizePool) usize {
        return self.wasted_space;
    }
    
    fn isCompatible(self: *SizePool, pooled_tensor: *const tensor.Tensor, shape: []const usize, dtype: tensor.DataType) bool {
        _ = self;
        
        // Check data type
        if (pooled_tensor.dtype != dtype) {
            return false;
        }
        
        // Check shape compatibility
        if (pooled_tensor.shape.len != shape.len) {
            return false;
        }
        
        for (pooled_tensor.shape, shape) |pool_dim, req_dim| {
            if (pool_dim != req_dim) {
                return false;
            }
        }
        
        return true;
    }
};

/// Configuration for advanced pool
pub const PoolConfig = struct {
    max_pool_size: usize = 1000,
    small_pool_size: usize = 200,
    medium_pool_size: usize = 100,
    large_pool_size: usize = 50,
    defrag_threshold: f32 = 0.3, // 30% fragmentation triggers defrag
    enable_defragmentation: bool = true,
};

/// Advanced pool statistics
pub const AdvancedPoolStats = struct {
    cache_hits: usize = 0,
    cache_misses: usize = 0,
    defragmentations: usize = 0,
    total_allocated: usize = 0,
    total_freed: usize = 0,
    peak_usage: usize = 0,
    current_usage: usize = 0,
    fragmentation_ratio: f32 = 0.0,
    small_pool_usage: usize = 0,
    medium_pool_usage: usize = 0,
    large_pool_usage: usize = 0,
    
    pub fn getCacheHitRatio(self: *const AdvancedPoolStats) f32 {
        const total_requests = self.cache_hits + self.cache_misses;
        if (total_requests == 0) return 0.0;
        return @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(total_requests));
    }
    
    pub fn getMemoryEfficiency(self: *const AdvancedPoolStats) f32 {
        if (self.total_allocated == 0) return 1.0;
        return @as(f32, @floatFromInt(self.current_usage)) / @as(f32, @floatFromInt(self.total_allocated));
    }
};

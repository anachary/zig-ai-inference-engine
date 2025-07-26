const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");

/// Pool statistics
pub const PoolStats = struct {
    num_pools: usize,
    total_tensors: usize,
    active_tensors: usize,
    peak_usage: usize,
    cache_hits: usize,
    cache_misses: usize,

    pub fn hitRatio(self: *const PoolStats) f32 {
        const total = self.cache_hits + self.cache_misses;
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(total));
    }
};

/// Size-based tensor pool for efficient memory reuse
pub const TensorPool = struct {
    pools: std.AutoHashMap(u64, std.ArrayList(tensor.Tensor)),
    allocator: Allocator,
    max_pool_size: usize,
    stats: PoolStats,

    const Self = @This();

    /// Initialize tensor pool with maximum pool size per tensor configuration
    pub fn init(allocator: Allocator, max_pool_size: usize) Self {
        return Self{
            .pools = std.AutoHashMap(u64, std.ArrayList(tensor.Tensor)).init(allocator),
            .allocator = allocator,
            .max_pool_size = max_pool_size,
            .stats = PoolStats{
                .num_pools = 0,
                .total_tensors = 0,
                .active_tensors = 0,
                .peak_usage = 0,
                .cache_hits = 0,
                .cache_misses = 0,
            },
        };
    }

    /// Deinitialize tensor pool and free all tensors
    pub fn deinit(self: *Self) void {
        var iterator = self.pools.iterator();
        while (iterator.next()) |entry| {
            for (entry.value_ptr.items) |*t| {
                t.deinit();
            }
            entry.value_ptr.deinit();
        }
        self.pools.deinit();
    }

    /// Get a tensor from the pool or create a new one
    pub fn getTensor(self: *Self, shape: []const usize, dtype: tensor.DataType) !tensor.Tensor {
        const key = self.computeKey(shape, dtype);

        if (self.pools.getPtr(key)) |pool| {
            if (pool.items.len > 0) {
                self.stats.cache_hits += 1;
                self.stats.active_tensors += 1;
                return pool.pop();
            }
        }

        // No available tensor in pool, create new one
        self.stats.cache_misses += 1;
        self.stats.active_tensors += 1;
        self.stats.peak_usage = @max(self.stats.peak_usage, self.stats.active_tensors);

        return tensor.Tensor.init(self.allocator, shape, dtype);
    }

    /// Return a tensor to the pool for reuse
    pub fn returnTensor(self: *Self, t: tensor.Tensor) !void {
        const key = self.computeKey(t.shape, t.dtype);

        const result = try self.pools.getOrPut(key);
        if (!result.found_existing) {
            result.value_ptr.* = std.ArrayList(tensor.Tensor).init(self.allocator);
            self.stats.num_pools += 1;
        }

        // Limit pool size to prevent unbounded growth
        if (result.value_ptr.items.len < self.max_pool_size) {
            // Zero out tensor data for security
            @memset(t.data, 0);
            try result.value_ptr.append(t);
            self.stats.total_tensors += 1;
        } else {
            // Pool is full, just deinit the tensor
            var mutable_t = t;
            mutable_t.deinit();
        }

        self.stats.active_tensors -= 1;
    }

    /// Force cleanup of a tensor (for immediate deallocation)
    pub fn cleanupTensor(self: *Self, t: tensor.Tensor) void {
        _ = self; // unused parameter
        var mutable_t = t;
        mutable_t.deinit();
    }

    /// Get pool statistics
    pub fn getStats(self: *const Self) PoolStats {
        return self.stats;
    }

    /// Clear all pools and free tensors
    pub fn clear(self: *Self) void {
        var iterator = self.pools.iterator();
        while (iterator.next()) |entry| {
            for (entry.value_ptr.items) |*t| {
                t.deinit();
            }
            entry.value_ptr.clearAndFree();
        }
        self.pools.clearAndFree();

        self.stats = PoolStats{
            .num_pools = 0,
            .total_tensors = 0,
            .active_tensors = 0,
            .peak_usage = self.stats.peak_usage, // Keep peak usage
            .cache_hits = 0,
            .cache_misses = 0,
        };
    }

    /// Compute hash key for tensor configuration
    fn computeKey(self: *Self, shape: []const usize, dtype: tensor.DataType) u64 {
        _ = self;

        // Simple hash computation
        var hash: u64 = 0;
        for (shape) |dim| {
            hash = hash *% 31 +% dim;
        }

        // Include data type in hash
        hash = hash *% 31 +% @intFromEnum(dtype);

        return hash;
    }
};

/// Memory tracker for monitoring allocations
pub const MemoryTracker = struct {
    total_allocated: std.atomic.Atomic(usize),
    peak_usage: std.atomic.Atomic(usize),
    current_usage: std.atomic.Atomic(usize),
    allocations: std.AutoHashMap(usize, AllocationInfo),
    mutex: std.Thread.Mutex,
    allocator: Allocator,

    const AllocationInfo = struct {
        size: usize,
        timestamp: i64,
        location: []const u8,
    };

    const Self = @This();

    /// Initialize memory tracker
    pub fn init(allocator: Allocator) Self {
        return Self{
            .total_allocated = std.atomic.Atomic(usize).init(0),
            .peak_usage = std.atomic.Atomic(usize).init(0),
            .current_usage = std.atomic.Atomic(usize).init(0),
            .allocations = std.AutoHashMap(usize, AllocationInfo).init(allocator),
            .mutex = std.Thread.Mutex{},
            .allocator = allocator,
        };
    }

    /// Deinitialize memory tracker
    pub fn deinit(self: *Self) void {
        self.allocations.deinit();
    }

    /// Track an allocation
    pub fn trackAllocation(self: *Self, ptr: usize, size: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const info = AllocationInfo{
            .size = size,
            .timestamp = std.time.timestamp(),
            .location = "unknown", // Could be enhanced with stack trace
        };

        self.allocations.put(ptr, info) catch {};

        _ = self.total_allocated.fetchAdd(size, .Monotonic);
        const current = self.current_usage.fetchAdd(size, .Monotonic) + size;

        // Update peak usage
        var peak = self.peak_usage.load(.Monotonic);
        while (current > peak) {
            peak = self.peak_usage.cmpxchgWeak(peak, current, .Monotonic, .Monotonic) orelse break;
        }
    }

    /// Track a deallocation
    pub fn trackDeallocation(self: *Self, ptr: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.allocations.fetchRemove(ptr)) |entry| {
            _ = self.current_usage.fetchSub(entry.value.size, .Monotonic);
        }
    }

    /// Get memory statistics
    pub fn getStats(self: *Self) MemoryStats {
        return MemoryStats{
            .total_allocated = self.total_allocated.load(.Monotonic),
            .peak_usage = self.peak_usage.load(.Monotonic),
            .current_usage = self.current_usage.load(.Monotonic),
            .active_allocations = self.allocations.count(),
        };
    }

    /// Reset statistics
    pub fn reset(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.total_allocated.store(0, .Monotonic);
        self.peak_usage.store(0, .Monotonic);
        self.current_usage.store(0, .Monotonic);
        self.allocations.clearAndFree();
    }
};

/// Memory statistics
pub const MemoryStats = struct {
    total_allocated: usize,
    peak_usage: usize,
    current_usage: usize,
    active_allocations: usize,
};

/// Tracked allocator that wraps another allocator with memory tracking
pub const TrackedAllocator = struct {
    backing_allocator: Allocator,
    tracker: *MemoryTracker,

    const Self = @This();

    /// Initialize tracked allocator
    pub fn init(backing_allocator: Allocator, tracker: *MemoryTracker) Self {
        return Self{
            .backing_allocator = backing_allocator,
            .tracker = tracker,
        };
    }

    /// Get Zig allocator interface
    pub fn allocator(self: *Self) Allocator {
        return Allocator.init(self, alloc, resize, free);
    }

    fn alloc(self: *Self, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const result = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr);
        if (result) |ptr| {
            self.tracker.trackAllocation(@intFromPtr(ptr), len);
        }
        return result;
    }

    fn resize(self: *Self, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const old_len = buf.len;
        const result = self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr);

        if (result) {
            if (new_len > old_len) {
                self.tracker.trackAllocation(@intFromPtr(buf.ptr), new_len - old_len);
            } else if (new_len < old_len) {
                self.tracker.trackDeallocation(@intFromPtr(buf.ptr));
                self.tracker.trackAllocation(@intFromPtr(buf.ptr), new_len);
            }
        }

        return result;
    }

    fn free(self: *Self, buf: []u8, buf_align: u8, ret_addr: usize) void {
        self.tracker.trackDeallocation(@intFromPtr(buf.ptr));
        self.backing_allocator.rawFree(buf, buf_align, ret_addr);
    }
};

// Tests
test "tensor pool operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var pool = TensorPool.init(allocator, 5);
    defer pool.deinit();

    // Get a tensor from empty pool (should create new)
    const shape = [_]usize{ 2, 3 };
    var t1 = try pool.getTensor(&shape, .f32);

    // Return tensor to pool
    try pool.returnTensor(t1);

    // Get tensor again (should reuse from pool)
    var t2 = try pool.getTensor(&shape, .f32);
    defer t2.deinit();

    const stats = pool.getStats();
    try testing.expect(stats.num_pools >= 1);
    try testing.expect(stats.cache_hits >= 1);
}

test "memory tracker" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var tracker = MemoryTracker.init(allocator);
    defer tracker.deinit();

    // Track some allocations
    tracker.trackAllocation(0x1000, 100);
    tracker.trackAllocation(0x2000, 200);

    const stats = tracker.getStats();
    try testing.expect(stats.total_allocated == 300);
    try testing.expect(stats.current_usage == 300);
    try testing.expect(stats.active_allocations == 2);

    // Track deallocation
    tracker.trackDeallocation(0x1000);

    const stats2 = tracker.getStats();
    try testing.expect(stats2.current_usage == 200);
    try testing.expect(stats2.active_allocations == 1);
}

const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");

pub const PoolError = error{
    OutOfMemory,
    InvalidSize,
    PoolExhausted,
};

/// Size-based tensor pool for efficient memory reuse
pub const TensorPool = struct {
    pools: std.AutoHashMap(u64, std.ArrayList(tensor.Tensor)),
    allocator: Allocator,
    max_pool_size: usize,
    total_tensors: usize,

    const Self = @This();

    pub fn init(allocator: Allocator, max_pool_size: usize) Self {
        return Self{
            .pools = std.AutoHashMap(u64, std.ArrayList(tensor.Tensor)).init(allocator),
            .allocator = allocator,
            .max_pool_size = max_pool_size,
            .total_tensors = 0,
        };
    }

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
    pub fn get_tensor(self: *Self, shape: []const usize, dtype: tensor.DataType) !tensor.Tensor {
        const key = self.compute_key(shape, dtype);

        if (self.pools.getPtr(key)) |pool| {
            if (pool.items.len > 0) {
                return pool.pop();
            }
        }

        // No available tensor in pool, create new one
        return tensor.Tensor.init(self.allocator, shape, dtype);
    }

    /// Return a tensor to the pool for reuse
    pub fn return_tensor(self: *Self, t: tensor.Tensor) !void {
        const key = self.compute_key(t.shape, t.dtype);

        const result = try self.pools.getOrPut(key);
        if (!result.found_existing) {
            result.value_ptr.* = std.ArrayList(tensor.Tensor).init(self.allocator);
        }

        // Limit pool size to prevent unbounded growth
        if (result.value_ptr.items.len < self.max_pool_size) {
            try result.value_ptr.append(t);
            self.total_tensors += 1;
        } else {
            // Pool is full, just deinit the tensor
            var mutable_t = t;
            mutable_t.deinit();
        }
    }

    /// Force cleanup of a tensor (for immediate deallocation)
    pub fn cleanup_tensor(self: *Self, t: tensor.Tensor) void {
        _ = self; // unused parameter
        var mutable_t = t;
        mutable_t.deinit();
    }

    /// Compute a hash key for shape and dtype combination
    fn compute_key(self: *Self, shape: []const usize, dtype: tensor.DataType) u64 {
        _ = self;
        var hash: u64 = 0;

        // Hash the shape
        for (shape) |dim| {
            hash = std.hash.Wyhash.hash(hash, std.mem.asBytes(&dim));
        }

        // Hash the dtype
        hash = std.hash.Wyhash.hash(hash, std.mem.asBytes(&dtype));

        return hash;
    }

    /// Get pool statistics
    pub fn get_stats(self: *const Self) PoolStats {
        var total_pooled: usize = 0;
        var num_pools: usize = 0;

        var iterator = self.pools.iterator();
        while (iterator.next()) |entry| {
            total_pooled += entry.value_ptr.items.len;
            num_pools += 1;
        }

        return PoolStats{
            .total_pooled = total_pooled,
            .num_pools = num_pools,
            .total_tensors = self.total_tensors,
        };
    }

    /// Clear all pools
    pub fn clear(self: *Self) void {
        var iterator = self.pools.iterator();
        while (iterator.next()) |entry| {
            for (entry.value_ptr.items) |*t| {
                t.deinit();
            }
            entry.value_ptr.clearAndFree();
        }
        self.total_tensors = 0;
    }
};

pub const PoolStats = struct {
    total_pooled: usize,
    num_pools: usize,
    total_tensors: usize,
};

/// Memory tracker for monitoring allocations
pub const MemoryTracker = struct {
    total_allocated: std.atomic.Atomic(usize),
    peak_usage: std.atomic.Atomic(usize),
    current_usage: std.atomic.Atomic(usize),
    allocations: std.AutoHashMap(usize, AllocationInfo),
    mutex: std.Thread.Mutex,
    allocator: Allocator,

    const Self = @This();

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

    pub fn deinit(self: *Self) void {
        self.allocations.deinit();
    }

    /// Track a memory allocation
    pub fn track_allocation(self: *Self, ptr: usize, size: usize) void {
        _ = self.total_allocated.fetchAdd(size, .Monotonic);
        const current = self.current_usage.fetchAdd(size, .Monotonic) + size;

        // Update peak usage
        var peak = self.peak_usage.load(.Monotonic);
        while (current > peak) {
            peak = self.peak_usage.compareAndSwap(peak, current, .Monotonic, .Monotonic) orelse break;
        }

        // Track individual allocation
        self.mutex.lock();
        defer self.mutex.unlock();

        self.allocations.put(ptr, AllocationInfo{
            .size = size,
            .timestamp = std.time.milliTimestamp(),
        }) catch {}; // Ignore errors in tracking
    }

    /// Track a memory deallocation
    pub fn track_deallocation(self: *Self, ptr: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.allocations.fetchRemove(ptr)) |entry| {
            _ = self.current_usage.fetchSub(entry.value.size, .Monotonic);
        }
    }

    /// Get memory usage statistics
    pub fn get_stats(self: *const Self) MemoryStats {
        return MemoryStats{
            .total_allocated = self.total_allocated.load(.Monotonic),
            .peak_usage = self.peak_usage.load(.Monotonic),
            .current_usage = self.current_usage.load(.Monotonic),
            .active_allocations = self.allocations.count(),
        };
    }

    /// Reset statistics
    pub fn reset(self: *Self) void {
        self.total_allocated.store(0, .Monotonic);
        self.peak_usage.store(0, .Monotonic);
        self.current_usage.store(0, .Monotonic);

        self.mutex.lock();
        defer self.mutex.unlock();
        self.allocations.clearAndFree();
    }
};

const AllocationInfo = struct {
    size: usize,
    timestamp: i64,
};

pub const MemoryStats = struct {
    total_allocated: usize,
    peak_usage: usize,
    current_usage: usize,
    active_allocations: u32,
};

/// Tracked allocator that wraps another allocator with memory tracking
pub const TrackedAllocator = struct {
    backing_allocator: Allocator,
    tracker: *MemoryTracker,

    const Self = @This();

    pub fn init(backing_allocator: Allocator, tracker: *MemoryTracker) Self {
        return Self{
            .backing_allocator = backing_allocator,
            .tracker = tracker,
        };
    }

    pub fn allocator(self: *Self) Allocator {
        return Allocator.init(self, alloc, resize, free);
    }

    fn alloc(self: *Self, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const result = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr);
        if (result) |ptr| {
            self.tracker.track_allocation(@intFromPtr(ptr), len);
        }
        return result;
    }

    fn resize(self: *Self, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const old_len = buf.len;
        const result = self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr);

        if (result) {
            if (new_len > old_len) {
                self.tracker.track_allocation(@intFromPtr(buf.ptr), new_len - old_len);
            } else if (new_len < old_len) {
                self.tracker.track_deallocation(@intFromPtr(buf.ptr));
                self.tracker.track_allocation(@intFromPtr(buf.ptr), new_len);
            }
        }

        return result;
    }

    fn free(self: *Self, buf: []u8, buf_align: u8, ret_addr: usize) void {
        self.tracker.track_deallocation(@intFromPtr(buf.ptr));
        self.backing_allocator.rawFree(buf, buf_align, ret_addr);
    }
};

test "tensor pool operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var pool = TensorPool.init(allocator, 5);
    defer pool.deinit();

    // Get a tensor from empty pool (should create new)
    const shape = [_]usize{ 2, 3 };
    var t1 = try pool.get_tensor(&shape, .f32);

    // Return tensor to pool
    try pool.return_tensor(t1);

    // Get tensor again (should reuse from pool)
    var t2 = try pool.get_tensor(&shape, .f32);
    defer t2.deinit();

    const stats = pool.get_stats();
    try testing.expect(stats.num_pools >= 1);
}

test "memory tracker" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var tracker = MemoryTracker.init(allocator);
    defer tracker.deinit();

    // Simulate allocations
    tracker.track_allocation(0x1000, 100);
    tracker.track_allocation(0x2000, 200);

    var stats = tracker.get_stats();
    try testing.expect(stats.total_allocated == 300);
    try testing.expect(stats.current_usage == 300);
    try testing.expect(stats.peak_usage == 300);

    // Simulate deallocation
    tracker.track_deallocation(0x1000);

    stats = tracker.get_stats();
    try testing.expect(stats.current_usage == 200);
    try testing.expect(stats.peak_usage == 300); // Peak should remain
}

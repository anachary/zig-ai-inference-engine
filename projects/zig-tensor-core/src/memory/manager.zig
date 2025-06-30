const std = @import("std");
const Allocator = std.mem.Allocator;

/// Memory management strategies for tensor operations
pub const MemoryStrategy = enum {
    general_purpose,    // Standard GPA
    arena,             // Arena allocator for batch operations
    pool,              // Pool allocator for fixed-size objects
    stack,             // Stack allocator for temporary objects
    ring_buffer,       // Ring buffer for streaming data
};

/// Memory statistics
pub const MemoryStats = struct {
    total_allocated: usize,
    peak_allocated: usize,
    current_allocated: usize,
    allocation_count: usize,
    deallocation_count: usize,
    fragmentation_ratio: f32,
};

/// Multi-strategy memory manager for tensor operations
pub const MemoryManager = struct {
    backing_allocator: Allocator,
    permanent_arena: std.heap.ArenaAllocator,
    temporary_arena: std.heap.ArenaAllocator,
    scratch_arena: std.heap.ArenaAllocator,
    stats: MemoryStats,
    
    const Self = @This();
    
    /// Initialize memory manager with backing allocator
    pub fn init(backing_allocator: Allocator) Self {
        return Self{
            .backing_allocator = backing_allocator,
            .permanent_arena = std.heap.ArenaAllocator.init(backing_allocator),
            .temporary_arena = std.heap.ArenaAllocator.init(backing_allocator),
            .scratch_arena = std.heap.ArenaAllocator.init(backing_allocator),
            .stats = MemoryStats{
                .total_allocated = 0,
                .peak_allocated = 0,
                .current_allocated = 0,
                .allocation_count = 0,
                .deallocation_count = 0,
                .fragmentation_ratio = 0.0,
            },
        };
    }
    
    /// Deinitialize memory manager and free all arenas
    pub fn deinit(self: *Self) void {
        self.permanent_arena.deinit();
        self.temporary_arena.deinit();
        self.scratch_arena.deinit();
    }
    
    /// Get allocator for permanent allocations (model weights, etc.)
    pub fn permanentAllocator(self: *Self) Allocator {
        return self.permanent_arena.allocator();
    }
    
    /// Get allocator for temporary allocations (intermediate results)
    pub fn temporaryAllocator(self: *Self) Allocator {
        return self.temporary_arena.allocator();
    }
    
    /// Get allocator for scratch space (very short-lived allocations)
    pub fn scratchAllocator(self: *Self) Allocator {
        return self.scratch_arena.allocator();
    }
    
    /// Get backing allocator for direct use
    pub fn backingAllocator(self: *Self) Allocator {
        return self.backing_allocator;
    }
    
    /// Reset temporary arena (free all temporary allocations)
    pub fn resetTemporary(self: *Self) void {
        _ = self.temporary_arena.reset(.retain_capacity);
    }
    
    /// Reset scratch arena (free all scratch allocations)
    pub fn resetScratch(self: *Self) void {
        _ = self.scratch_arena.reset(.retain_capacity);
    }
    
    /// Reset both temporary and scratch arenas
    pub fn resetAll(self: *Self) void {
        self.resetTemporary();
        self.resetScratch();
    }
    
    /// Get current memory statistics
    pub fn getStats(self: *const Self) MemoryStats {
        return self.stats;
    }
    
    /// Update memory statistics (called by tracked allocators)
    pub fn updateStats(self: *Self, allocated: usize, deallocated: usize) void {
        self.stats.total_allocated += allocated;
        self.stats.current_allocated = self.stats.current_allocated + allocated - deallocated;
        self.stats.peak_allocated = @max(self.stats.peak_allocated, self.stats.current_allocated);
        
        if (allocated > 0) self.stats.allocation_count += 1;
        if (deallocated > 0) self.stats.deallocation_count += 1;
        
        // Simple fragmentation estimate
        if (self.stats.total_allocated > 0) {
            self.stats.fragmentation_ratio = 1.0 - (@as(f32, @floatFromInt(self.stats.current_allocated)) / @as(f32, @floatFromInt(self.stats.total_allocated)));
        }
    }
};

/// Arena allocator wrapper with size limits
pub const ArenaManager = struct {
    arena: std.heap.ArenaAllocator,
    max_size: usize,
    current_size: usize,
    
    const Self = @This();
    
    /// Initialize arena manager with size limit
    pub fn init(backing_allocator: Allocator, max_size: usize) Self {
        return Self{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .max_size = max_size,
            .current_size = 0,
        };
    }
    
    /// Deinitialize arena manager
    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }
    
    /// Get arena allocator
    pub fn allocator(self: *Self) Allocator {
        return self.arena.allocator();
    }
    
    /// Reset arena and clear size tracking
    pub fn reset(self: *Self) void {
        _ = self.arena.reset(.retain_capacity);
        self.current_size = 0;
    }
    
    /// Check if allocation would exceed size limit
    pub fn canAllocate(self: *const Self, size: usize) bool {
        return self.current_size + size <= self.max_size;
    }
    
    /// Get current usage
    pub fn getCurrentSize(self: *const Self) usize {
        return self.current_size;
    }
    
    /// Get maximum size
    pub fn getMaxSize(self: *const Self) usize {
        return self.max_size;
    }
    
    /// Get usage ratio (0.0 to 1.0)
    pub fn getUsageRatio(self: *const Self) f32 {
        if (self.max_size == 0) return 0.0;
        return @as(f32, @floatFromInt(self.current_size)) / @as(f32, @floatFromInt(self.max_size));
    }
};

/// Stack allocator for very fast temporary allocations
pub const StackAllocator = struct {
    buffer: []u8,
    offset: usize,
    allocator: Allocator,
    
    const Self = @This();
    
    /// Initialize stack allocator with fixed buffer size
    pub fn init(backing_allocator: Allocator, size: usize) !Self {
        const buffer = try backing_allocator.alloc(u8, size);
        return Self{
            .buffer = buffer,
            .offset = 0,
            .allocator = backing_allocator,
        };
    }
    
    /// Deinitialize stack allocator
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buffer);
    }
    
    /// Allocate from stack
    pub fn alloc(self: *Self, comptime T: type, n: usize) ![]T {
        const size = n * @sizeOf(T);
        const alignment = @alignOf(T);
        
        // Align offset
        const aligned_offset = std.mem.alignForward(usize, self.offset, alignment);
        
        if (aligned_offset + size > self.buffer.len) {
            return error.OutOfMemory;
        }
        
        const result = @as([*]T, @ptrCast(@alignCast(self.buffer.ptr + aligned_offset)))[0..n];
        self.offset = aligned_offset + size;
        
        return result;
    }
    
    /// Reset stack (free all allocations)
    pub fn reset(self: *Self) void {
        self.offset = 0;
    }
    
    /// Get current usage
    pub fn getCurrentUsage(self: *const Self) usize {
        return self.offset;
    }
    
    /// Get total capacity
    pub fn getCapacity(self: *const Self) usize {
        return self.buffer.len;
    }
    
    /// Get usage ratio
    pub fn getUsageRatio(self: *const Self) f32 {
        return @as(f32, @floatFromInt(self.offset)) / @as(f32, @floatFromInt(self.buffer.len));
    }
};

/// Ring buffer allocator for streaming data
pub const RingBufferAllocator = struct {
    buffer: []u8,
    head: usize,
    tail: usize,
    full: bool,
    allocator: Allocator,
    
    const Self = @This();
    
    /// Initialize ring buffer allocator
    pub fn init(backing_allocator: Allocator, size: usize) !Self {
        const buffer = try backing_allocator.alloc(u8, size);
        return Self{
            .buffer = buffer,
            .head = 0,
            .tail = 0,
            .full = false,
            .allocator = backing_allocator,
        };
    }
    
    /// Deinitialize ring buffer allocator
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buffer);
    }
    
    /// Allocate from ring buffer
    pub fn alloc(self: *Self, comptime T: type, n: usize) ![]T {
        const size = n * @sizeOf(T);
        const alignment = @alignOf(T);
        
        // Align head
        const aligned_head = std.mem.alignForward(usize, self.head, alignment);
        
        // Check if we have enough space
        const available = if (self.full) 0 else if (self.tail >= aligned_head) self.buffer.len - aligned_head else self.tail - aligned_head;
        
        if (size > available) {
            return error.OutOfMemory;
        }
        
        const result = @as([*]T, @ptrCast(@alignCast(self.buffer.ptr + aligned_head)))[0..n];
        self.head = (aligned_head + size) % self.buffer.len;
        
        if (self.head == self.tail) {
            self.full = true;
        }
        
        return result;
    }
    
    /// Free from ring buffer (advance tail)
    pub fn free(self: *Self, size: usize) void {
        if (self.full) {
            self.full = false;
        }
        self.tail = (self.tail + size) % self.buffer.len;
    }
    
    /// Reset ring buffer
    pub fn reset(self: *Self) void {
        self.head = 0;
        self.tail = 0;
        self.full = false;
    }
    
    /// Get current usage
    pub fn getCurrentUsage(self: *const Self) usize {
        if (self.full) return self.buffer.len;
        if (self.head >= self.tail) return self.head - self.tail;
        return self.buffer.len - self.tail + self.head;
    }
    
    /// Get capacity
    pub fn getCapacity(self: *const Self) usize {
        return self.buffer.len;
    }
};

// Tests
test "memory manager basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var manager = MemoryManager.init(allocator);
    defer manager.deinit();
    
    // Test different allocators
    const perm_alloc = manager.permanentAllocator();
    const temp_alloc = manager.temporaryAllocator();
    const scratch_alloc = manager.scratchAllocator();
    
    // Allocate from each
    const perm_data = try perm_alloc.alloc(u8, 100);
    const temp_data = try temp_alloc.alloc(u8, 200);
    const scratch_data = try scratch_alloc.alloc(u8, 50);
    
    try testing.expect(perm_data.len == 100);
    try testing.expect(temp_data.len == 200);
    try testing.expect(scratch_data.len == 50);
    
    // Reset temporary and scratch
    manager.resetTemporary();
    manager.resetScratch();
    
    // Permanent data should still be valid, temp/scratch are freed
    try testing.expect(perm_data.len == 100);
}

test "arena manager with size limits" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var arena = ArenaManager.init(allocator, 1024);
    defer arena.deinit();
    
    try testing.expect(arena.canAllocate(500));
    try testing.expect(arena.getUsageRatio() == 0.0);
    
    const data = try arena.allocator().alloc(u8, 500);
    try testing.expect(data.len == 500);
    
    arena.reset();
    try testing.expect(arena.getCurrentSize() == 0);
}

test "stack allocator" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var stack = try StackAllocator.init(allocator, 1024);
    defer stack.deinit();
    
    const data1 = try stack.alloc(u32, 10);
    try testing.expect(data1.len == 10);
    
    const data2 = try stack.alloc(f32, 20);
    try testing.expect(data2.len == 20);
    
    try testing.expect(stack.getCurrentUsage() > 0);
    
    stack.reset();
    try testing.expect(stack.getCurrentUsage() == 0);
}

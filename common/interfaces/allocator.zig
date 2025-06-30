const std = @import("std");
const Allocator = std.mem.Allocator;

/// Memory allocator interface for the AI ecosystem
/// Provides abstraction over different allocation strategies
pub const AllocatorInterface = struct {
    /// Allocation strategy types
    pub const Strategy = enum {
        general_purpose,    // Standard GPA
        arena,             // Arena allocator for batch operations
        pool,              // Pool allocator for fixed-size objects
        stack,             // Stack allocator for temporary objects
        ring_buffer,       // Ring buffer for streaming data
    };

    /// Memory statistics
    pub const Stats = struct {
        total_allocated: usize,
        peak_allocated: usize,
        current_allocated: usize,
        allocation_count: usize,
        deallocation_count: usize,
        fragmentation_ratio: f32,
    };

    /// Allocation errors
    pub const AllocatorError = error{
        OutOfMemory,
        InvalidSize,
        InvalidAlignment,
        PoolExhausted,
        ArenaExhausted,
        StackOverflow,
    };

    /// Core allocator operations
    pub const Operations = struct {
        /// Allocate memory
        allocFn: *const fn (ctx: *anyopaque, len: usize, alignment: u8) AllocatorError![]u8,
        
        /// Resize allocation
        resizeFn: *const fn (ctx: *anyopaque, buf: []u8, new_len: usize, alignment: u8) AllocatorError![]u8,
        
        /// Free memory
        freeFn: *const fn (ctx: *anyopaque, buf: []u8, alignment: u8) void,
        
        /// Get statistics
        statsFn: *const fn (ctx: *anyopaque) Stats,
        
        /// Reset allocator (for arena/pool allocators)
        resetFn: *const fn (ctx: *anyopaque) void,
        
        /// Check if allocation is valid
        isValidFn: *const fn (ctx: *anyopaque, buf: []u8) bool,
        
        /// Get allocator strategy
        strategyFn: *const fn (ctx: *anyopaque) Strategy,
    };

    /// Implementation
    impl: Operations,
    ctx: *anyopaque,

    /// Create allocator interface
    pub fn init(ctx: *anyopaque, impl: Operations) AllocatorInterface {
        return AllocatorInterface{
            .impl = impl,
            .ctx = ctx,
        };
    }

    /// Allocate memory
    pub fn alloc(self: *AllocatorInterface, comptime T: type, n: usize) AllocatorError![]T {
        const bytes = try self.impl.allocFn(self.ctx, n * @sizeOf(T), @alignOf(T));
        return @as([*]T, @ptrCast(@alignCast(bytes.ptr)))[0..n];
    }

    /// Allocate single item
    pub fn create(self: *AllocatorInterface, comptime T: type) AllocatorError!*T {
        const slice = try self.alloc(T, 1);
        return &slice[0];
    }

    /// Free memory
    pub fn free(self: *AllocatorInterface, memory: anytype) void {
        const Slice = @TypeOf(memory);
        const slice_info = @typeInfo(Slice).Pointer;
        const alignment = slice_info.alignment;
        const bytes = @as([*]u8, @ptrCast(memory.ptr))[0..memory.len * slice_info.child];
        self.impl.freeFn(self.ctx, bytes, alignment);
    }

    /// Resize allocation
    pub fn resize(self: *AllocatorInterface, old_mem: anytype, new_n: usize) AllocatorError!@TypeOf(old_mem) {
        const Slice = @TypeOf(old_mem);
        const slice_info = @typeInfo(Slice).Pointer;
        const T = slice_info.child;
        const old_bytes = @as([*]u8, @ptrCast(old_mem.ptr))[0..old_mem.len * @sizeOf(T)];
        const new_bytes = try self.impl.resizeFn(self.ctx, old_bytes, new_n * @sizeOf(T), @alignOf(T));
        return @as([*]T, @ptrCast(@alignCast(new_bytes.ptr)))[0..new_n];
    }

    /// Get statistics
    pub fn getStats(self: *AllocatorInterface) Stats {
        return self.impl.statsFn(self.ctx);
    }

    /// Reset allocator
    pub fn reset(self: *AllocatorInterface) void {
        self.impl.resetFn(self.ctx);
    }

    /// Check if allocation is valid
    pub fn isValid(self: *AllocatorInterface, memory: anytype) bool {
        const Slice = @TypeOf(memory);
        const slice_info = @typeInfo(Slice).Pointer;
        const bytes = @as([*]u8, @ptrCast(memory.ptr))[0..memory.len * @sizeOf(slice_info.child)];
        return self.impl.isValidFn(self.ctx, bytes);
    }

    /// Get strategy
    pub fn getStrategy(self: *AllocatorInterface) Strategy {
        return self.impl.strategyFn(self.ctx);
    }

    /// Convert to standard Zig allocator
    pub fn toZigAllocator(self: *AllocatorInterface) Allocator {
        return Allocator{
            .ptr = self.ctx,
            .vtable = &.{
                .alloc = zigAllocFn,
                .resize = zigResizeFn,
                .free = zigFreeFn,
            },
        };
    }

    fn zigAllocFn(ctx: *anyopaque, len: usize, alignment: u8, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self = @as(*AllocatorInterface, @ptrCast(@alignCast(ctx)));
        const bytes = self.impl.allocFn(self.ctx, len, alignment) catch return null;
        return bytes.ptr;
    }

    fn zigResizeFn(ctx: *anyopaque, buf: []u8, alignment: u8, new_len: usize, ret_addr: usize) bool {
        _ = ret_addr;
        const self = @as(*AllocatorInterface, @ptrCast(@alignCast(ctx)));
        const new_bytes = self.impl.resizeFn(self.ctx, buf, new_len, alignment) catch return false;
        return new_bytes.len == new_len;
    }

    fn zigFreeFn(ctx: *anyopaque, buf: []u8, alignment: u8, ret_addr: usize) void {
        _ = ret_addr;
        const self = @as(*AllocatorInterface, @ptrCast(@alignCast(ctx)));
        self.impl.freeFn(self.ctx, buf, alignment);
    }
};

/// Memory pool interface for tensor operations
pub const PoolInterface = struct {
    /// Pool configuration
    pub const Config = struct {
        object_size: usize,
        initial_capacity: usize,
        max_capacity: usize,
        growth_factor: f32,
        alignment: u8,
    };

    /// Pool statistics
    pub const PoolStats = struct {
        total_objects: usize,
        used_objects: usize,
        free_objects: usize,
        peak_usage: usize,
        allocation_failures: usize,
    };

    /// Pool operations
    pub const Operations = struct {
        /// Get object from pool
        getFn: *const fn (ctx: *anyopaque) AllocatorInterface.AllocatorError!*anyopaque,
        
        /// Return object to pool
        putFn: *const fn (ctx: *anyopaque, obj: *anyopaque) void,
        
        /// Get pool statistics
        statsFn: *const fn (ctx: *anyopaque) PoolStats,
        
        /// Reset pool
        resetFn: *const fn (ctx: *anyopaque) void,
        
        /// Grow pool capacity
        growFn: *const fn (ctx: *anyopaque, additional_capacity: usize) AllocatorInterface.AllocatorError!void,
    };

    impl: Operations,
    ctx: *anyopaque,

    pub fn init(ctx: *anyopaque, impl: Operations) PoolInterface {
        return PoolInterface{
            .impl = impl,
            .ctx = ctx,
        };
    }

    pub fn get(self: *PoolInterface, comptime T: type) AllocatorInterface.AllocatorError!*T {
        const ptr = try self.impl.getFn(self.ctx);
        return @as(*T, @ptrCast(@alignCast(ptr)));
    }

    pub fn put(self: *PoolInterface, obj: anytype) void {
        self.impl.putFn(self.ctx, @as(*anyopaque, @ptrCast(obj)));
    }

    pub fn getStats(self: *PoolInterface) PoolStats {
        return self.impl.statsFn(self.ctx);
    }

    pub fn reset(self: *PoolInterface) void {
        self.impl.resetFn(self.ctx);
    }

    pub fn grow(self: *PoolInterface, additional_capacity: usize) AllocatorInterface.AllocatorError!void {
        return self.impl.growFn(self.ctx, additional_capacity);
    }
};

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const MemoryError = error{
    OutOfMemory,
    InvalidSize,
    PoolExhausted,
};

pub const MemoryManager = struct {
    backing_allocator: Allocator,
    permanent_arena: std.heap.ArenaAllocator,
    temporary_arena: std.heap.ArenaAllocator,
    scratch_arena: std.heap.ArenaAllocator,
    
    const Self = @This();
    
    pub fn init(backing_allocator: Allocator) Self {
        return Self{
            .backing_allocator = backing_allocator,
            .permanent_arena = std.heap.ArenaAllocator.init(backing_allocator),
            .temporary_arena = std.heap.ArenaAllocator.init(backing_allocator),
            .scratch_arena = std.heap.ArenaAllocator.init(backing_allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.permanent_arena.deinit();
        self.temporary_arena.deinit();
        self.scratch_arena.deinit();
    }
    
    pub fn permanent_allocator(self: *Self) Allocator {
        return self.permanent_arena.allocator();
    }
    
    pub fn temporary_allocator(self: *Self) Allocator {
        return self.temporary_arena.allocator();
    }
    
    pub fn scratch_allocator(self: *Self) Allocator {
        return self.scratch_arena.allocator();
    }
    
    pub fn reset_temporary(self: *Self) void {
        _ = self.temporary_arena.reset(.retain_capacity);
    }
    
    pub fn reset_scratch(self: *Self) void {
        _ = self.scratch_arena.reset(.retain_capacity);
    }
};

// TODO: Implement tensor pool, memory tracking, etc.

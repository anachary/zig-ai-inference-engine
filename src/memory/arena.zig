const std = @import("std");

/// Arena allocator for temporary allocations during inference
pub const ArenaAllocator = struct {
    buffer: []u8,
    offset: usize,
    base_allocator: std.mem.Allocator,

    pub fn init(base_allocator: std.mem.Allocator, size: usize) !ArenaAllocator {
        const buffer = try base_allocator.alloc(u8, size);
        return ArenaAllocator{
            .buffer = buffer,
            .offset = 0,
            .base_allocator = base_allocator,
        };
    }

    pub fn deinit(self: *ArenaAllocator) void {
        self.base_allocator.free(self.buffer);
    }

    pub fn allocator(self: *ArenaAllocator) std.mem.Allocator {
        return std.mem.Allocator{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }

    pub fn reset(self: *ArenaAllocator) void {
        self.offset = 0;
    }

    pub fn getUsed(self: *ArenaAllocator) usize {
        return self.offset;
    }

    pub fn getRemaining(self: *ArenaAllocator) usize {
        return self.buffer.len - self.offset;
    }

    fn alloc(ctx: *anyopaque, len: usize, log2_ptr_align: u8, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));

        const ptr_align = @as(usize, 1) << @intCast(log2_ptr_align);
        const aligned_offset = std.mem.alignForward(usize, self.offset, ptr_align);

        if (aligned_offset + len > self.buffer.len) {
            return null; // Out of memory
        }

        const result = self.buffer[aligned_offset..].ptr;
        self.offset = aligned_offset + len;

        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = log2_buf_align;
        _ = new_len;
        _ = ret_addr;
        // Arena allocator doesn't support resize
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, ret_addr: usize) void {
        _ = ctx;
        _ = buf;
        _ = log2_buf_align;
        _ = ret_addr;
        // Arena allocator doesn't support individual free
    }
};

test "arena allocator basic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var arena = try ArenaAllocator.init(allocator, 1024);
    defer arena.deinit();

    const arena_allocator = arena.allocator();

    const ptr1 = try arena_allocator.alloc(u8, 100);
    try testing.expect(ptr1.len == 100);
    try testing.expect(arena.getUsed() >= 100);

    const ptr2 = try arena_allocator.alloc(u32, 10);
    try testing.expect(ptr2.len == 10);

    arena.reset();
    try testing.expect(arena.getUsed() == 0);
}

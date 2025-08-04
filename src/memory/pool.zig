const std = @import("std");

/// Memory pool for efficient allocation of fixed-size blocks
pub const MemoryPool = struct {
    buffer: []u8,
    block_size: usize,
    alignment: usize,
    free_list: ?*FreeNode,
    total_blocks: usize,
    used_blocks: usize,
    base_allocator: std.mem.Allocator,

    const FreeNode = struct {
        next: ?*FreeNode,
    };

    pub fn init(base_allocator: std.mem.Allocator, total_size: usize, alignment: usize) !MemoryPool {
        const block_size = @max(@sizeOf(FreeNode), alignment);
        const aligned_block_size = std.mem.alignForward(usize, block_size, alignment);
        const total_blocks = total_size / aligned_block_size;
        const actual_size = total_blocks * aligned_block_size;

        const buffer = try base_allocator.alignedAlloc(u8, alignment, actual_size);

        var pool = MemoryPool{
            .buffer = buffer,
            .block_size = aligned_block_size,
            .alignment = alignment,
            .free_list = null,
            .total_blocks = total_blocks,
            .used_blocks = 0,
            .base_allocator = base_allocator,
        };

        // Initialize free list
        pool.initializeFreeList();

        return pool;
    }

    pub fn deinit(self: *MemoryPool) void {
        self.base_allocator.free(self.buffer);
    }

    pub fn allocator(self: *MemoryPool) std.mem.Allocator {
        return std.mem.Allocator{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }

    pub fn getUsed(self: *MemoryPool) usize {
        return self.used_blocks * self.block_size;
    }

    pub fn getTotal(self: *MemoryPool) usize {
        return self.total_blocks * self.block_size;
    }

    pub fn getUtilization(self: *MemoryPool) f32 {
        return @as(f32, @floatFromInt(self.used_blocks)) / @as(f32, @floatFromInt(self.total_blocks));
    }

    fn initializeFreeList(self: *MemoryPool) void {
        self.free_list = null;

        // Link all blocks in the free list
        var i: usize = 0;
        while (i < self.total_blocks) : (i += 1) {
            const block_ptr = self.getBlockPtr(i);
            const node: *FreeNode = @ptrCast(@alignCast(block_ptr));
            node.next = self.free_list;
            self.free_list = node;
        }
    }

    fn getBlockPtr(self: *MemoryPool, block_index: usize) [*]u8 {
        return self.buffer.ptr + block_index * self.block_size;
    }

    fn getBlockIndex(self: *MemoryPool, ptr: [*]u8) usize {
        const offset = @intFromPtr(ptr) - @intFromPtr(self.buffer.ptr);
        return offset / self.block_size;
    }

    fn allocBlock(self: *MemoryPool) ?[*]u8 {
        if (self.free_list) |node| {
            self.free_list = node.next;
            self.used_blocks += 1;
            return @ptrCast(node);
        }
        return null;
    }

    fn freeBlock(self: *MemoryPool, ptr: [*]u8) void {
        const node: *FreeNode = @ptrCast(@alignCast(ptr));
        node.next = self.free_list;
        self.free_list = node;
        self.used_blocks -= 1;
    }

    fn alloc(ctx: *anyopaque, len: usize, log2_ptr_align: u8, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *MemoryPool = @ptrCast(@alignCast(ctx));

        const ptr_align = @as(usize, 1) << @intCast(log2_ptr_align);

        // Check if request fits in a single block and alignment is compatible
        if (len > self.block_size or ptr_align > self.alignment) {
            return null;
        }

        return self.allocBlock();
    }

    fn resize(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = log2_buf_align;
        _ = new_len;
        _ = ret_addr;
        // Pool allocator doesn't support resize
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, ret_addr: usize) void {
        _ = log2_buf_align;
        _ = ret_addr;
        const self: *MemoryPool = @ptrCast(@alignCast(ctx));

        if (buf.len == 0) return;

        // Verify the pointer is within our buffer
        const buf_start = @intFromPtr(buf.ptr);
        const pool_start = @intFromPtr(self.buffer.ptr);
        const pool_end = pool_start + self.buffer.len;

        if (buf_start < pool_start or buf_start >= pool_end) {
            return; // Not our pointer
        }

        self.freeBlock(buf.ptr);
    }
};

/// Variable-size memory pool using buddy allocation
pub const BuddyPool = struct {
    buffer: []u8,
    min_block_size: usize,
    max_order: u8,
    free_lists: []?*FreeNode,
    base_allocator: std.mem.Allocator,

    const FreeNode = struct {
        next: ?*FreeNode,
        order: u8,
    };

    pub fn init(base_allocator: std.mem.Allocator, total_size: usize, min_block_size: usize) !BuddyPool {
        // Ensure total_size is a power of 2
        const actual_size = std.math.ceilPowerOfTwo(usize, total_size) catch return error.SizeTooLarge;
        const max_order = std.math.log2_int(usize, actual_size / min_block_size);

        const buffer = try base_allocator.alloc(u8, actual_size);
        const free_lists = try base_allocator.alloc(?*FreeNode, max_order + 1);
        @memset(free_lists, null);

        var pool = BuddyPool{
            .buffer = buffer,
            .min_block_size = min_block_size,
            .max_order = @intCast(max_order),
            .free_lists = free_lists,
            .base_allocator = base_allocator,
        };

        // Initialize with one large free block
        const root_node: *FreeNode = @ptrCast(@alignCast(buffer.ptr));
        root_node.next = null;
        root_node.order = pool.max_order;
        pool.free_lists[pool.max_order] = root_node;

        return pool;
    }

    pub fn deinit(self: *BuddyPool) void {
        self.base_allocator.free(self.free_lists);
        self.base_allocator.free(self.buffer);
    }

    pub fn allocator(self: *BuddyPool) std.mem.Allocator {
        return std.mem.Allocator{
            .ptr = self,
            .vtable = &.{
                .alloc = buddyAlloc,
                .resize = buddyResize,
                .free = buddyFree,
            },
        };
    }

    fn getOrderForSize(self: *BuddyPool, size: usize) u8 {
        const blocks_needed = (size + self.min_block_size - 1) / self.min_block_size;
        const order = std.math.log2_int_ceil(usize, blocks_needed);
        return @min(@as(u8, @intCast(order)), self.max_order);
    }

    fn buddyAlloc(ctx: *anyopaque, len: usize, log2_ptr_align: u8, ret_addr: usize) ?[*]u8 {
        _ = log2_ptr_align;
        _ = ret_addr;
        const self: *BuddyPool = @ptrCast(@alignCast(ctx));

        const order = self.getOrderForSize(len);
        return self.allocateBlock(order);
    }

    fn buddyResize(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = log2_buf_align;
        _ = new_len;
        _ = ret_addr;
        return false;
    }

    fn buddyFree(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, ret_addr: usize) void {
        _ = log2_buf_align;
        _ = ret_addr;
        const self: *BuddyPool = @ptrCast(@alignCast(ctx));

        if (buf.len == 0) return;

        const order = self.getOrderForSize(buf.len);
        self.freeBlock(buf.ptr, order);
    }

    fn allocateBlock(self: *BuddyPool, order: u8) ?[*]u8 {
        // Find a free block of the requested order or larger
        var current_order = order;
        while (current_order <= self.max_order) : (current_order += 1) {
            if (self.free_lists[current_order]) |node| {
                // Remove from free list
                self.free_lists[current_order] = node.next;

                // Split larger blocks if necessary
                while (current_order > order) {
                    current_order -= 1;
                    const buddy_ptr = @as([*]u8, @ptrCast(node)) + (@as(usize, 1) << @intCast(current_order)) * self.min_block_size;
                    const buddy_node: *FreeNode = @ptrCast(@alignCast(buddy_ptr));
                    buddy_node.order = current_order;
                    buddy_node.next = self.free_lists[current_order];
                    self.free_lists[current_order] = buddy_node;
                }

                return @ptrCast(node);
            }
        }

        return null; // No free blocks available
    }

    fn freeBlock(self: *BuddyPool, ptr: [*]u8, order: u8) void {
        var current_ptr = ptr;
        var current_order = order;

        // Try to merge with buddy blocks
        while (current_order < self.max_order) {
            const block_size = (@as(usize, 1) << @intCast(current_order)) * self.min_block_size;
            const block_offset = @intFromPtr(current_ptr) - @intFromPtr(self.buffer.ptr);
            const buddy_offset = block_offset ^ block_size;
            const buddy_ptr = self.buffer.ptr + buddy_offset;

            // Find and remove buddy from free list
            var prev: ?*FreeNode = null;
            var current = self.free_lists[current_order];

            while (current) |node| {
                if (@intFromPtr(node) == @intFromPtr(buddy_ptr)) {
                    // Remove buddy from free list
                    if (prev) |p| {
                        p.next = node.next;
                    } else {
                        self.free_lists[current_order] = node.next;
                    }

                    // Merge blocks
                    current_ptr = if (block_offset < buddy_offset) current_ptr else buddy_ptr;
                    current_order += 1;
                    break;
                }
                prev = node;
                current = node.next;
            } else {
                // No buddy found, can't merge further
                break;
            }
        }

        // Add merged block to free list
        const node: *FreeNode = @ptrCast(@alignCast(current_ptr));
        node.order = current_order;
        node.next = self.free_lists[current_order];
        self.free_lists[current_order] = node;
    }
};

test "memory pool basic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var pool = try MemoryPool.init(allocator, 1024, 32);
    defer pool.deinit();

    const pool_allocator = pool.allocator();

    const ptr1 = try pool_allocator.alloc(u8, 16);
    try testing.expect(ptr1.len == 16);
    try testing.expect(pool.used_blocks == 1);

    pool_allocator.free(ptr1);
    try testing.expect(pool.used_blocks == 0);
}

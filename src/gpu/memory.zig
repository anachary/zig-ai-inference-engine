const std = @import("std");
const device = @import("device.zig");
const Allocator = std.mem.Allocator;

/// GPU memory allocation errors
pub const MemoryError = error{
    OutOfMemory,
    InvalidSize,
    InvalidAlignment,
    MemoryNotMapped,
    TransferFailed,
    UnsupportedOperation,
};

/// GPU memory allocation types
pub const MemoryType = enum {
    device_local, // GPU-only memory (fastest)
    host_visible, // CPU-accessible GPU memory
    host_coherent, // CPU-GPU coherent memory
    unified, // Unified memory (if supported)
};

/// GPU memory buffer handle
pub const GPUBuffer = struct {
    ptr: ?*anyopaque,
    size: usize,
    memory_type: MemoryType,
    device_ptr: ?*anyopaque, // Device-specific pointer
    is_mapped: bool,

    const Self = @This();

    /// Map buffer for CPU access (if supported)
    pub fn map(self: *Self) !*anyopaque {
        if (self.is_mapped and self.ptr != null) {
            return self.ptr.?;
        }

        switch (self.memory_type) {
            .host_visible, .host_coherent, .unified => {
                // These types should already be mappable
                if (self.ptr) |ptr| {
                    self.is_mapped = true;
                    return ptr;
                } else {
                    return MemoryError.MemoryNotMapped;
                }
            },
            .device_local => {
                // Device-local memory cannot be mapped
                return MemoryError.UnsupportedOperation;
            },
        }
    }

    /// Unmap buffer from CPU access
    pub fn unmap(self: *Self) void {
        if (self.is_mapped) {
            self.is_mapped = false;
            // Backend-specific unmapping would go here
        }
    }

    /// Get device pointer for GPU operations
    pub fn getDevicePtr(self: *const Self) ?*anyopaque {
        return self.device_ptr;
    }
};

/// GPU memory pool for efficient allocation
pub const GPUMemoryPool = struct {
    allocator: Allocator,
    device_ref: *const device.GPUDevice,
    free_blocks: std.ArrayList(Block),
    allocated_blocks: std.ArrayList(Block),
    total_allocated: usize,
    peak_usage: usize,

    const Block = struct {
        buffer: GPUBuffer,
        offset: usize,
        size: usize,
        is_free: bool,
    };

    const Self = @This();

    /// Initialize GPU memory pool
    pub fn init(allocator: Allocator, gpu_device: *const device.GPUDevice) Self {
        std.log.debug("Initializing memory pool with device type: {s}", .{@tagName(gpu_device.capabilities.device_type)});

        return Self{
            .allocator = allocator,
            .device_ref = gpu_device,
            .free_blocks = std.ArrayList(Block).init(allocator),
            .allocated_blocks = std.ArrayList(Block).init(allocator),
            .total_allocated = 0,
            .peak_usage = 0,
        };
    }

    /// Deinitialize memory pool and free all blocks
    pub fn deinit(self: *Self) void {
        // Free all allocated blocks
        for (self.allocated_blocks.items) |*block| {
            self.freeBlock(block);
        }

        self.free_blocks.deinit();
        self.allocated_blocks.deinit();
    }

    /// Allocate GPU memory buffer
    pub fn allocate(self: *Self, size: usize, memory_type: MemoryType) !GPUBuffer {
        if (size == 0) {
            return MemoryError.InvalidSize;
        }

        std.log.debug("Memory pool allocate called: device_type={s}", .{@tagName(self.device_ref.capabilities.device_type)});

        // Align size to 256 bytes for optimal GPU performance
        const aligned_size = std.mem.alignForward(usize, size, 256);

        // Try to find a suitable free block first
        if (self.findFreeBlock(aligned_size, memory_type)) |block_idx| {
            const block = &self.free_blocks.items[block_idx];
            block.is_free = false;

            // Move from free to allocated
            try self.allocated_blocks.append(block.*);
            _ = self.free_blocks.swapRemove(block_idx);

            return block.buffer;
        }

        // No suitable free block, allocate new one
        const buffer = try self.allocateNewBuffer(aligned_size, memory_type);

        try self.allocated_blocks.append(Block{
            .buffer = buffer,
            .offset = 0,
            .size = aligned_size,
            .is_free = false,
        });

        self.total_allocated += aligned_size;
        self.peak_usage = @max(self.peak_usage, self.total_allocated);

        return buffer;
    }

    /// Free GPU memory buffer
    pub fn free(self: *Self, buffer: GPUBuffer) !void {
        // Find the buffer in allocated blocks
        for (self.allocated_blocks.items, 0..) |*block, i| {
            if (block.buffer.ptr == buffer.ptr) {
                // Move to free blocks
                block.is_free = true;
                try self.free_blocks.append(block.*);
                _ = self.allocated_blocks.swapRemove(i);

                self.total_allocated -= block.size;
                return;
            }
        }

        // Buffer not found, free directly
        self.freeBuffer(buffer);
    }

    /// Get memory usage statistics
    pub fn getStats(self: *const Self) struct {
        total_allocated: usize,
        peak_usage: usize,
        free_blocks: usize,
        allocated_blocks: usize,
    } {
        return .{
            .total_allocated = self.total_allocated,
            .peak_usage = self.peak_usage,
            .free_blocks = self.free_blocks.items.len,
            .allocated_blocks = self.allocated_blocks.items.len,
        };
    }

    /// Private: Find a suitable free block
    fn findFreeBlock(self: *Self, size: usize, memory_type: MemoryType) ?usize {
        for (self.free_blocks.items, 0..) |block, i| {
            if (block.size >= size and block.buffer.memory_type == memory_type) {
                return i;
            }
        }
        return null;
    }

    /// Private: Allocate new GPU buffer
    fn allocateNewBuffer(self: *Self, size: usize, memory_type: MemoryType) !GPUBuffer {
        std.log.debug("Allocating buffer: size={d}, type={s}, device={s}", .{ size, @tagName(memory_type), @tagName(self.device_ref.capabilities.device_type) });

        switch (self.device_ref.capabilities.device_type) {
            .cpu => return self.allocateCpuBuffer(size, memory_type),
            .cuda => return self.allocateCudaBuffer(size, memory_type),
            .vulkan => return self.allocateVulkanBuffer(size, memory_type),
            .opencl => return MemoryError.UnsupportedOperation,
        }
    }

    /// Private: Allocate CPU buffer
    fn allocateCpuBuffer(self: *Self, size: usize, memory_type: MemoryType) !GPUBuffer {
        _ = memory_type; // CPU treats all memory types the same

        const slice = try self.allocator.alignedAlloc(u8, 256, size);

        // Initialize the buffer to zero for safety
        @memset(slice, 0);

        return GPUBuffer{
            .ptr = slice.ptr,
            .size = size,
            .memory_type = .unified,
            .device_ptr = slice.ptr, // Same as host pointer for CPU
            .is_mapped = true,
        };
    }

    /// Private: Allocate CUDA buffer
    fn allocateCudaBuffer(self: *Self, size: usize, memory_type: MemoryType) !GPUBuffer {
        _ = self;
        _ = size;
        _ = memory_type;

        // TODO: Implement CUDA memory allocation
        return MemoryError.UnsupportedOperation;
    }

    /// Private: Allocate Vulkan buffer
    fn allocateVulkanBuffer(self: *Self, size: usize, memory_type: MemoryType) !GPUBuffer {
        _ = self;
        _ = size;
        _ = memory_type;

        // TODO: Implement Vulkan memory allocation
        return MemoryError.UnsupportedOperation;
    }

    /// Private: Free a memory block
    fn freeBlock(self: *Self, block: *Block) void {
        self.freeBuffer(block.buffer);
    }

    /// Private: Free GPU buffer
    fn freeBuffer(self: *Self, buffer: GPUBuffer) void {
        switch (self.device_ref.capabilities.device_type) {
            .cpu => {
                if (buffer.ptr) |ptr| {
                    // Reconstruct the slice for proper deallocation
                    const slice = @as([*]u8, @ptrCast(@alignCast(ptr)))[0..buffer.size];
                    self.allocator.free(slice);
                }
            },
            .cuda => {
                // TODO: Implement CUDA memory deallocation
            },
            .vulkan => {
                // TODO: Implement Vulkan memory deallocation
            },
            .opencl => {
                // TODO: Future OpenCL support
            },
        }
    }
};

/// GPU memory transfer operations
pub const MemoryTransfer = struct {
    /// Copy data from host to GPU
    pub fn hostToDevice(dst: GPUBuffer, src: []const u8) !void {
        if (dst.size < src.len) {
            return MemoryError.InvalidSize;
        }

        const dst_ptr = try dst.map();
        const dst_slice = @as([*]u8, @ptrCast(@alignCast(dst_ptr)))[0..src.len];
        @memcpy(dst_slice, src);
        dst.unmap();
    }

    /// Copy data from GPU to host
    pub fn deviceToHost(dst: []u8, src: GPUBuffer) !void {
        if (src.size < dst.len) {
            return MemoryError.InvalidSize;
        }

        const src_ptr = try src.map();
        const src_slice = @as([*]const u8, @ptrCast(@alignCast(src_ptr)))[0..dst.len];
        @memcpy(dst, src_slice);
        src.unmap();
    }

    /// Copy data between GPU buffers
    pub fn deviceToDevice(dst: *GPUBuffer, src: *GPUBuffer) !void {
        if (dst.size < src.size) {
            return MemoryError.InvalidSize;
        }

        // For now, use host as intermediate (not optimal but works)
        const src_ptr = try src.map();
        const dst_ptr = try dst.map();

        const src_slice = @as([*]const u8, @ptrCast(@alignCast(src_ptr)))[0..src.size];
        const dst_slice = @as([*]u8, @ptrCast(@alignCast(dst_ptr)))[0..src.size];

        @memcpy(dst_slice, src_slice);

        src.unmap();
        dst.unmap();
    }
};

/// Utility function to create optimal memory type for tensor operations
pub fn selectOptimalMemoryType(device_caps: device.DeviceCapabilities, is_input: bool, is_output: bool) MemoryType {
    switch (device_caps.device_type) {
        .cpu => return .unified,
        .cuda, .vulkan => {
            if (is_input or is_output) {
                // Input/output tensors need host access
                return if (device_caps.supports_unified_memory) .unified else .host_visible;
            } else {
                // Intermediate tensors can be device-local for performance
                return .device_local;
            }
        },
        .opencl => return .host_visible,
    }
}

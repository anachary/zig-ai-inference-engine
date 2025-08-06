const std = @import("std");

/// Performance optimization utilities for production deployment
pub const PerformanceOptimizer = struct {
    allocator: std.mem.Allocator,
    memory_pool: ?MemoryPool = null,
    simd_enabled: bool = false,
    batch_size: usize = 1,

    pub fn init(allocator: std.mem.Allocator) PerformanceOptimizer {
        return PerformanceOptimizer{
            .allocator = allocator,
            .simd_enabled = detectSIMDSupport(),
        };
    }

    pub fn deinit(self: *PerformanceOptimizer) void {
        if (self.memory_pool) |*pool| {
            pool.deinit();
        }
    }

    /// Initialize memory pool for hot path allocations
    pub fn initMemoryPool(self: *PerformanceOptimizer, pool_size: usize) !void {
        self.memory_pool = try MemoryPool.init(self.allocator, pool_size);
    }

    /// Get optimized allocator (memory pool if available, otherwise default)
    pub fn getAllocator(self: *PerformanceOptimizer) std.mem.Allocator {
        if (self.memory_pool) |*pool| {
            return pool.allocator();
        }
        return self.allocator;
    }

    /// Detect SIMD capabilities of the current CPU
    fn detectSIMDSupport() bool {
        // For now, assume SIMD is available on x86_64
        // In production, this would check CPU features
        return @import("builtin").cpu.arch == .x86_64;
    }

    /// Get optimal batch size for current hardware
    pub fn getOptimalBatchSize(self: *PerformanceOptimizer) usize {
        // Simple heuristic: larger batches for SIMD-capable hardware
        if (self.simd_enabled) {
            return 8;
        }
        return 4;
    }
};

/// Memory pool for efficient allocation in hot paths
pub const MemoryPool = struct {
    buffer: []u8,
    offset: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(parent_allocator: std.mem.Allocator, size: usize) !MemoryPool {
        const buffer = try parent_allocator.alloc(u8, size);
        return MemoryPool{
            .buffer = buffer,
            .allocator = parent_allocator,
        };
    }

    pub fn deinit(self: *MemoryPool) void {
        self.allocator.free(self.buffer);
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

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *MemoryPool = @ptrCast(@alignCast(ctx));

        const aligned_offset = std.mem.alignForward(usize, self.offset, @as(usize, 1) << @intCast(ptr_align));
        const new_offset = aligned_offset + len;

        if (new_offset > self.buffer.len) {
            return null; // Pool exhausted
        }

        self.offset = new_offset;
        return self.buffer[aligned_offset..new_offset].ptr;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        return false; // Pool allocator doesn't support resize
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = ret_addr;
        // Pool allocator doesn't free individual allocations
    }

    pub fn reset(self: *MemoryPool) void {
        self.offset = 0;
    }
};

/// SIMD-optimized matrix operations
pub const SIMDOps = struct {
    /// Optimized matrix multiplication using SIMD when available
    pub fn matmulOptimized(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
        // For now, use standard implementation
        // In production, this would use AVX2/AVX-512 intrinsics
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..k) |l| {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    /// Optimized vector addition
    pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32) void {
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == result.len);

        // Simple vectorized addition
        for (a, b, result) |a_val, b_val, *r| {
            r.* = a_val + b_val;
        }
    }

    /// Optimized softmax computation
    pub fn softmaxOptimized(input: []const f32, output: []f32) void {
        std.debug.assert(input.len == output.len);

        // Find maximum for numerical stability
        var max_val: f32 = input[0];
        for (input[1..]) |val| {
            max_val = @max(max_val, val);
        }

        // Compute exponentials and sum
        var sum: f32 = 0.0;
        for (input, output) |val, *out| {
            out.* = @exp(val - max_val);
            sum += out.*;
        }

        // Normalize
        for (output) |*val| {
            val.* /= sum;
        }
    }
};

/// Batch processing utilities
pub const BatchProcessor = struct {
    batch_size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, batch_size: usize) BatchProcessor {
        return BatchProcessor{
            .batch_size = batch_size,
            .allocator = allocator,
        };
    }

    /// Process multiple inputs in batches for better throughput
    pub fn processBatch(
        self: *BatchProcessor,
        inputs: []const []const u32,
        processor: *const fn ([]const u32, std.mem.Allocator) anyerror![]u32,
    ) ![][]u32 {
        var results = try self.allocator.alloc([]u32, inputs.len);

        var i: usize = 0;
        while (i < inputs.len) {
            const end = @min(i + self.batch_size, inputs.len);

            // Process batch
            for (i..end) |j| {
                results[j] = try processor(inputs[j], self.allocator);
            }

            i = end;
        }

        return results;
    }
};

/// Performance profiler for identifying bottlenecks
pub const Profiler = struct {
    timings: std.HashMap([]const u8, u64, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Profiler {
        return Profiler{
            .timings = std.HashMap([]const u8, u64, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.timings.deinit();
    }

    pub fn startTimer(self: *Profiler, name: []const u8) !void {
        const start_time = std.time.nanoTimestamp();
        try self.timings.put(name, @intCast(start_time));
    }

    pub fn endTimer(self: *Profiler, name: []const u8) !u64 {
        const end_time = std.time.nanoTimestamp();
        if (self.timings.get(name)) |start_time| {
            const duration = @as(u64, @intCast(end_time)) - start_time;
            try self.timings.put(name, duration);
            return duration;
        }
        return 0;
    }

    pub fn printReport(self: *Profiler) void {
        std.debug.print("\nPerformance Report:\n", .{});
        std.debug.print("==================\n", .{});

        var iterator = self.timings.iterator();
        while (iterator.next()) |entry| {
            const duration_ms = @as(f64, @floatFromInt(entry.value_ptr.*)) / 1_000_000.0;
            std.debug.print("{s}: {d:.2} ms\n", .{ entry.key_ptr.*, duration_ms });
        }
    }
};

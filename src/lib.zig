const std = @import("std");

// Core modules
pub const tensor = @import("core/tensor.zig");
pub const shape = @import("core/shape.zig");
pub const simd = @import("core/simd.zig");
pub const memory = @import("memory/manager.zig");
pub const pool = @import("memory/pool.zig");
pub const engine = @import("engine/inference.zig");
pub const operators = @import("engine/operators.zig");
pub const registry = @import("engine/registry.zig");
pub const scheduler = @import("scheduler/task_queue.zig");
pub const network = @import("network/server.zig");

// Re-export main types
pub const Tensor = tensor.Tensor;
pub const DataType = tensor.DataType;
pub const Engine = engine.InferenceEngine;
pub const MemoryManager = memory.MemoryManager;
pub const TensorPool = pool.TensorPool;
pub const MemoryTracker = pool.MemoryTracker;
pub const Operator = operators.Operator;
pub const OperatorRegistry = registry.OperatorRegistry;
pub const Scheduler = scheduler.TaskScheduler;
pub const Server = network.HTTPServer;

// Version information
pub const version = std.SemanticVersion{
    .major = 0,
    .minor = 1,
    .patch = 0,
};

// Feature flags
pub const features = struct {
    pub const simd_support = true;
    pub const gpu_support = false; // TODO: Implement
    pub const privacy_sandbox = false; // TODO: Implement
    pub const distributed_inference = false; // TODO: Implement
};

// Configuration
pub const Config = struct {
    max_memory_mb: u32 = 1024,
    num_threads: ?u32 = null,
    enable_profiling: bool = false,
    log_level: std.log.Level = .info,
};

// Initialize the library
pub fn init(allocator: std.mem.Allocator, config: Config) !void {
    _ = allocator;
    _ = config;

    std.log.info("Zig AI Interface Engine v{} initialized", .{version});

    // TODO: Initialize global state, thread pools, etc.
}

// Cleanup
pub fn deinit() void {
    std.log.info("Zig AI Interface Engine shutting down", .{});

    // TODO: Cleanup global state
}

// Hardware capabilities structure
pub const HardwareCapabilities = struct {
    simd_level: enum { none, sse, avx2, avx512 },
    num_cores: u32,
    cache_sizes: struct {
        l1: u32,
        l2: u32,
        l3: u32,
    },
};

// Utility functions
pub fn detectHardwareCapabilities() HardwareCapabilities {
    const builtin = @import("builtin");

    var caps = HardwareCapabilities{
        .simd_level = .none,
        .num_cores = @as(u32, @intCast(std.Thread.getCpuCount() catch 1)),
        .cache_sizes = .{ .l1 = 32 * 1024, .l2 = 256 * 1024, .l3 = 8 * 1024 * 1024 },
    };

    if (builtin.cpu.arch == .x86_64) {
        if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) {
            caps.simd_level = .avx512;
        } else if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
            caps.simd_level = .avx2;
        } else if (std.Target.x86.featureSetHas(builtin.cpu.features, .sse)) {
            caps.simd_level = .sse;
        }
    }

    return caps;
}

test "library initialization" {
    const testing = std.testing;

    try init(testing.allocator, .{});
    defer deinit();

    const caps = detectHardwareCapabilities();
    try testing.expect(caps.num_cores > 0);
}

test "version information" {
    const testing = std.testing;

    try testing.expect(version.major == 0);
    try testing.expect(version.minor == 1);
    try testing.expect(version.patch == 0);
}

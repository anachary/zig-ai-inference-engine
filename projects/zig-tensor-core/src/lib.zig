const std = @import("std");

// Core tensor functionality
pub const Tensor = @import("core/tensor.zig").Tensor;
pub const DataType = @import("core/tensor.zig").DataType;
pub const Device = @import("core/tensor.zig").Device;
pub const TensorError = @import("core/tensor.zig").TensorError;

// Shape utilities
pub const shape = @import("core/shape.zig");

// SIMD operations
pub const simd = @import("core/simd.zig");

// Math operations
pub const math = @import("math.zig");

// Memory management
pub const memory = @import("memory/manager.zig");
pub const pool = @import("memory/pool.zig");

// Re-export commonly used types
pub const MemoryManager = memory.MemoryManager;
pub const ArenaManager = memory.ArenaManager;
pub const StackAllocator = memory.StackAllocator;
pub const TensorPool = pool.TensorPool;
pub const MemoryTracker = pool.MemoryTracker;
pub const TrackedAllocator = pool.TrackedAllocator;

// Configuration and utilities
pub const Config = struct {
    /// Memory settings
    max_memory_mb: u32 = 1024,
    tensor_pool_size: usize = 100,
    arena_size_mb: u32 = 256,

    /// SIMD settings
    enable_simd: bool = true,
    simd_alignment: u8 = 32,

    /// Debug settings
    enable_bounds_checking: bool = true,
    enable_memory_tracking: bool = false,

    /// Validate configuration
    pub fn validate(self: *const Config) bool {
        if (self.max_memory_mb == 0) return false;
        if (self.tensor_pool_size == 0) return false;
        if (self.simd_alignment == 0 or (self.simd_alignment & (self.simd_alignment - 1)) != 0) return false;
        if (self.arena_size_mb > self.max_memory_mb) return false;
        return true;
    }

    /// Get recommended configuration for device type
    pub fn forDevice(device_type: DeviceType, available_memory_mb: u32) Config {
        return switch (device_type) {
            .iot => Config{
                .max_memory_mb = @min(512, available_memory_mb / 2),
                .tensor_pool_size = 20,
                .arena_size_mb = @min(128, available_memory_mb / 4),
                .enable_simd = true,
                .simd_alignment = 16,
                .enable_bounds_checking = false,
                .enable_memory_tracking = false,
            },
            .desktop => Config{
                .max_memory_mb = @min(2048, available_memory_mb / 2),
                .tensor_pool_size = 100,
                .arena_size_mb = @min(512, available_memory_mb / 4),
                .enable_simd = true,
                .simd_alignment = 32,
                .enable_bounds_checking = true,
                .enable_memory_tracking = true,
            },
            .server => Config{
                .max_memory_mb = @min(8192, available_memory_mb / 2),
                .tensor_pool_size = 500,
                .arena_size_mb = @min(2048, available_memory_mb / 4),
                .enable_simd = true,
                .simd_alignment = 32,
                .enable_bounds_checking = false,
                .enable_memory_tracking = true,
            },
        };
    }
};

/// Device types for configuration
pub const DeviceType = enum {
    iot,
    desktop,
    server,
};

/// Tensor core context for managing resources
pub const TensorCore = struct {
    allocator: std.mem.Allocator,
    config: Config,
    memory_manager: MemoryManager,
    tensor_pool: TensorPool,
    memory_tracker: ?MemoryTracker,

    const Self = @This();

    /// Initialize tensor core with configuration
    pub fn init(allocator: std.mem.Allocator, config: Config) !Self {
        if (!config.validate()) {
            return error.InvalidConfiguration;
        }

        var memory_manager = MemoryManager.init(allocator);
        var tensor_pool = TensorPool.init(allocator, config.tensor_pool_size);

        var memory_tracker: ?MemoryTracker = null;
        if (config.enable_memory_tracking) {
            memory_tracker = MemoryTracker.init(allocator);
        }

        return Self{
            .allocator = allocator,
            .config = config,
            .memory_manager = memory_manager,
            .tensor_pool = tensor_pool,
            .memory_tracker = memory_tracker,
        };
    }

    /// Deinitialize tensor core
    pub fn deinit(self: *Self) void {
        self.tensor_pool.deinit();
        self.memory_manager.deinit();
        if (self.memory_tracker) |*tracker| {
            tracker.deinit();
        }
    }

    /// Create a new tensor
    pub fn createTensor(self: *Self, tensor_shape: []const usize, dtype: DataType) !Tensor {
        return self.tensor_pool.getTensor(tensor_shape, dtype);
    }

    /// Return a tensor to the pool
    pub fn returnTensor(self: *Self, tensor: Tensor) !void {
        return self.tensor_pool.returnTensor(tensor);
    }

    /// Get temporary allocator
    pub fn tempAllocator(self: *Self) std.mem.Allocator {
        return self.memory_manager.temporaryAllocator();
    }

    /// Get scratch allocator
    pub fn scratchAllocator(self: *Self) std.mem.Allocator {
        return self.memory_manager.scratchAllocator();
    }

    /// Reset temporary memory
    pub fn resetTemp(self: *Self) void {
        self.memory_manager.resetTemporary();
    }

    /// Reset scratch memory
    pub fn resetScratch(self: *Self) void {
        self.memory_manager.resetScratch();
    }

    /// Get memory statistics
    pub fn getMemoryStats(self: *Self) ?pool.MemoryStats {
        if (self.memory_tracker) |*tracker| {
            return tracker.getStats();
        }
        return null;
    }

    /// Get tensor pool statistics
    pub fn getPoolStats(self: *const Self) pool.PoolStats {
        return self.tensor_pool.getStats();
    }
};

/// Convenience functions for common operations
pub const ops = struct {
    /// Create a tensor filled with zeros
    pub fn zeros(allocator: std.mem.Allocator, tensor_shape: []const usize, dtype: DataType) !Tensor {
        var tensor = try Tensor.init(allocator, tensor_shape, dtype);
        tensor.zero();
        return tensor;
    }

    /// Create a tensor filled with ones
    pub fn ones(allocator: std.mem.Allocator, tensor_shape: []const usize, dtype: DataType) !Tensor {
        var tensor = try Tensor.init(allocator, tensor_shape, dtype);
        switch (dtype) {
            .f32 => try tensor.fill(@as(f32, 1.0)),
            .i32 => try tensor.fill(@as(i32, 1)),
            .i8 => try tensor.fill(@as(i8, 1)),
            .u8 => try tensor.fill(@as(u8, 1)),
            else => return TensorError.UnsupportedDataType,
        }
        return tensor;
    }

    /// Create a tensor filled with a specific value
    pub fn full(allocator: std.mem.Allocator, tensor_shape: []const usize, dtype: DataType, value: anytype) !Tensor {
        var tensor = try Tensor.init(allocator, tensor_shape, dtype);
        try tensor.fill(value);
        return tensor;
    }

    /// Create an identity matrix
    pub fn eye(allocator: std.mem.Allocator, n: usize, dtype: DataType) !Tensor {
        const matrix_shape = [_]usize{ n, n };
        var tensor = try zeros(allocator, &matrix_shape, dtype);

        switch (dtype) {
            .f32 => {
                for (0..n) |i| {
                    try tensor.setF32(&[_]usize{ i, i }, 1.0);
                }
            },
            .i32 => {
                for (0..n) |i| {
                    try tensor.setI32(&[_]usize{ i, i }, 1);
                }
            },
            else => return TensorError.UnsupportedDataType,
        }

        return tensor;
    }

    /// Create a tensor from a slice of data
    pub fn fromSlice(allocator: std.mem.Allocator, data: anytype, tensor_shape: []const usize) !Tensor {
        return Tensor.fromSlice(allocator, data, tensor_shape);
    }
};

/// Version information
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;
    pub const string = "0.1.0";
};

// Tests
test "tensor core initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = Config.forDevice(.desktop, 4096);
    var core = try TensorCore.init(allocator, config);
    defer core.deinit();

    // Test tensor creation
    var tensor = try core.createTensor(&[_]usize{ 2, 3 }, .f32);
    try tensor.fill(@as(f32, 5.0));

    // Return to pool
    try core.returnTensor(tensor);

    // Get pool stats
    const stats = core.getPoolStats();
    try testing.expect(stats.num_pools >= 1);
}

test "convenience operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test zeros
    var zeros_tensor = try ops.zeros(allocator, &[_]usize{ 2, 2 }, .f32);
    defer zeros_tensor.deinit();
    try testing.expect(try zeros_tensor.getF32(&[_]usize{ 0, 0 }) == 0.0);

    // Test ones
    var ones_tensor = try ops.ones(allocator, &[_]usize{ 2, 2 }, .f32);
    defer ones_tensor.deinit();
    try testing.expect(try ones_tensor.getF32(&[_]usize{ 0, 0 }) == 1.0);

    // Test identity matrix
    var eye_tensor = try ops.eye(allocator, 3, .f32);
    defer eye_tensor.deinit();
    try testing.expect(try eye_tensor.getF32(&[_]usize{ 0, 0 }) == 1.0);
    try testing.expect(try eye_tensor.getF32(&[_]usize{ 0, 1 }) == 0.0);
    try testing.expect(try eye_tensor.getF32(&[_]usize{ 1, 1 }) == 1.0);
}

test "configuration validation" {
    const testing = std.testing;

    const valid_config = Config{
        .max_memory_mb = 1024,
        .tensor_pool_size = 100,
        .simd_alignment = 16,
    };
    try testing.expect(valid_config.validate());

    const invalid_config = Config{
        .max_memory_mb = 0, // Invalid
        .tensor_pool_size = 100,
        .simd_alignment = 16,
    };
    try testing.expect(!invalid_config.validate());
}

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const DataType = enum {
    f32,
    f16,
    i32,
    i16,
    i8,
    u8,

    pub fn size(self: DataType) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .i16 => 2,
            .i8, .u8 => 1,
        };
    }

    pub fn alignment(self: DataType) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .i16 => 2,
            .i8, .u8 => 1,
        };
    }
};

pub const Device = enum {
    cpu,
    gpu,
    npu,
};

pub const TensorError = error{
    InvalidShape,
    ShapeMismatch,
    IndexOutOfBounds,
    UnsupportedDataType,
    OutOfMemory,
};

pub const Tensor = struct {
    data: []u8,
    shape: []const usize,
    strides: []const usize,
    dtype: DataType,
    device: Device,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, shape: []const usize, dtype: DataType) !Self {
        // Handle 0D scalar case
        if (shape.len == 0) {
            // 0D scalar: single element, no strides needed
            const data = try allocator.alloc(u8, dtype.size());
            const shape_copy = try allocator.alloc(usize, 0); // Empty shape array
            const strides = try allocator.alloc(usize, 0); // Empty strides array

            return Self{
                .data = data,
                .shape = shape_copy,
                .strides = strides,
                .dtype = dtype,
                .device = .cpu,
                .allocator = allocator,
            };
        }

        // Calculate total elements for N-D tensors
        var total_elements: usize = 1;
        for (shape) |dim| {
            if (dim == 0) return TensorError.InvalidShape;
            total_elements *= dim;
        }

        // Calculate strides (row-major order)
        const strides = try allocator.alloc(usize, shape.len);
        var stride: usize = 1;
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= shape[i];
        }

        // Allocate data buffer
        const data_size = total_elements * dtype.size();
        const data = try allocator.alloc(u8, data_size);

        // Copy shape
        const shape_copy = try allocator.dupe(usize, shape);

        return Self{
            .data = data,
            .shape = shape_copy,
            .strides = strides,
            .dtype = dtype,
            .device = .cpu,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
    }

    /// Create a 0D scalar tensor with the given value
    pub fn scalar(allocator: Allocator, value: anytype, dtype: DataType) !Self {
        var tensor = try Self.init(allocator, &[_]usize{}, dtype);

        switch (dtype) {
            .f32 => {
                const f32_value: f32 = switch (@TypeOf(value)) {
                    f32 => value,
                    f64 => @floatCast(value),
                    comptime_float => @floatCast(value),
                    i32, i16, i8, u8 => @floatFromInt(value),
                    comptime_int => @floatFromInt(value),
                    else => @compileError("Unsupported type for f32 scalar"),
                };
                try tensor.set_scalar_f32(f32_value);
            },
            .i32 => {
                const i32_value: i32 = switch (@TypeOf(value)) {
                    i32 => value,
                    i16, i8 => @intCast(value),
                    u8 => @intCast(value),
                    comptime_int => @intCast(value),
                    f32, f64, comptime_float => @intFromFloat(value),
                    else => @compileError("Unsupported type for i32 scalar"),
                };
                const data_ptr = @as([*]i32, @ptrCast(@alignCast(tensor.data.ptr)));
                data_ptr[0] = i32_value;
            },
            else => return TensorError.UnsupportedDataType,
        }

        return tensor;
    }

    pub fn numel(self: *const Self) usize {
        // 0D scalar has 1 element
        if (self.shape.len == 0) return 1;

        var total: usize = 1;
        for (self.shape) |dim| {
            total *= dim;
        }
        return total;
    }

    pub fn ndim(self: *const Self) usize {
        return self.shape.len;
    }

    pub fn size_bytes(self: *const Self) usize {
        return self.numel() * self.dtype.size();
    }

    pub fn get_f32(self: *const Self, indices: []const usize) !f32 {
        if (self.dtype != .f32) return TensorError.UnsupportedDataType;
        const offset = try self.compute_offset(indices);
        const data_ptr = @as([*]const f32, @ptrCast(@alignCast(self.data.ptr)));
        return data_ptr[offset / 4];
    }

    pub fn set_f32(self: *Self, indices: []const usize, value: f32) !void {
        if (self.dtype != .f32) return TensorError.UnsupportedDataType;
        const offset = try self.compute_offset(indices);
        const data_ptr = @as([*]f32, @ptrCast(@alignCast(self.data.ptr)));
        data_ptr[offset / 4] = value;
    }

    pub fn get_f32_flat(self: *const Self, index: usize) !f32 {
        if (self.dtype != .f32) return TensorError.UnsupportedDataType;
        if (index >= self.numel()) return TensorError.IndexOutOfBounds;
        const data_ptr = @as([*]const f32, @ptrCast(@alignCast(self.data.ptr)));
        return data_ptr[index];
    }

    pub fn set_f32_flat(self: *Self, index: usize, value: f32) !void {
        if (self.dtype != .f32) return TensorError.UnsupportedDataType;
        if (index >= self.numel()) return TensorError.IndexOutOfBounds;
        const data_ptr = @as([*]f32, @ptrCast(@alignCast(self.data.ptr)));
        data_ptr[index] = value;
    }

    /// Get scalar value from 0D tensor
    pub fn get_scalar_f32(self: *const Self) !f32 {
        if (self.dtype != .f32) return TensorError.UnsupportedDataType;
        if (self.shape.len != 0) return TensorError.InvalidShape;
        const data_ptr = @as([*]const f32, @ptrCast(@alignCast(self.data.ptr)));
        return data_ptr[0];
    }

    /// Set scalar value for 0D tensor
    pub fn set_scalar_f32(self: *Self, value: f32) !void {
        if (self.dtype != .f32) return TensorError.UnsupportedDataType;
        if (self.shape.len != 0) return TensorError.InvalidShape;
        const data_ptr = @as([*]f32, @ptrCast(@alignCast(self.data.ptr)));
        data_ptr[0] = value;
    }

    /// Check if tensor is a scalar (0D)
    pub fn is_scalar(self: *const Self) bool {
        return self.shape.len == 0;
    }

    fn compute_offset(self: *const Self, indices: []const usize) !usize {
        // 0D scalar case: only allow empty indices
        if (self.shape.len == 0) {
            if (indices.len != 0) return TensorError.IndexOutOfBounds;
            return 0; // Scalar always at offset 0
        }

        if (indices.len != self.shape.len) return TensorError.IndexOutOfBounds;

        var offset: usize = 0;
        for (indices, self.shape, self.strides) |idx, dim, stride| {
            if (idx >= dim) return TensorError.IndexOutOfBounds;
            offset += idx * stride * self.dtype.size();
        }

        return offset;
    }

    pub fn reshape(self: *Self, allocator: Allocator, new_shape: []const usize) !Self {
        // Validate that total elements remain the same
        var new_total: usize = 1;
        for (new_shape) |dim| {
            new_total *= dim;
        }

        if (new_total != self.numel()) return TensorError.ShapeMismatch;

        // Create new tensor with same data but different shape
        var reshaped = try Self.init(allocator, new_shape, self.dtype);
        @memcpy(reshaped.data, self.data);

        return reshaped;
    }

    pub fn slice(self: *const Self, allocator: Allocator, start: []const usize, end: []const usize) !Self {
        if (start.len != self.shape.len or end.len != self.shape.len) {
            return TensorError.IndexOutOfBounds;
        }

        // Calculate new shape
        var new_shape = try allocator.alloc(usize, self.shape.len);
        for (start, end, new_shape, self.shape) |s, e, *new_dim, orig_dim| {
            if (s >= orig_dim or e > orig_dim or s >= e) {
                allocator.free(new_shape);
                return TensorError.IndexOutOfBounds;
            }
            new_dim.* = e - s;
        }

        // Create new tensor (this is a simplified implementation)
        // In a full implementation, this would create a view without copying data
        var sliced = try Self.init(allocator, new_shape, self.dtype);

        // TODO: Implement actual slicing logic
        // For now, just return empty tensor with correct shape

        return sliced;
    }

    pub fn format(self: *const Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.print("Tensor(shape=[", .{});
        for (self.shape, 0..) |dim, i| {
            if (i > 0) try writer.print(", ", .{});
            try writer.print("{d}", .{dim});
        }
        try writer.print("], dtype={s}, device={s})", .{ @tagName(self.dtype), @tagName(self.device) });
    }
};

// Utility functions
pub fn zeros(allocator: Allocator, shape: []const usize, dtype: DataType) !Tensor {
    var tensor = try Tensor.init(allocator, shape, dtype);
    @memset(tensor.data, 0);
    return tensor;
}

pub fn ones(allocator: Allocator, shape: []const usize, dtype: DataType) !Tensor {
    var tensor = try Tensor.init(allocator, shape, dtype);

    switch (dtype) {
        .f32 => {
            const data_ptr = @as([*]f32, @ptrCast(@alignCast(tensor.data.ptr)));
            const len = tensor.numel();
            for (0..len) |i| {
                data_ptr[i] = 1.0;
            }
        },
        else => return TensorError.UnsupportedDataType,
    }

    return tensor;
}

pub fn arange(allocator: Allocator, start: f32, end: f32, step: f32, dtype: DataType) !Tensor {
    if (dtype != .f32) return TensorError.UnsupportedDataType;

    const count = @as(usize, @intFromFloat(@ceil((end - start) / step)));
    const shape = [_]usize{count};

    var tensor = try Tensor.init(allocator, &shape, dtype);
    const data_ptr = @as([*]f32, @ptrCast(@alignCast(tensor.data.ptr)));

    for (0..count) |i| {
        data_ptr[i] = start + @as(f32, @floatFromInt(i)) * step;
    }

    return tensor;
}

test "tensor creation and basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test tensor creation
    const shape = [_]usize{ 2, 3 };
    var tensor = try Tensor.init(allocator, &shape, .f32);
    defer tensor.deinit();

    try testing.expect(tensor.numel() == 6);
    try testing.expect(tensor.ndim() == 2);
    try testing.expect(tensor.size_bytes() == 24);

    // Test element access
    try tensor.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try tensor.set_f32(&[_]usize{ 1, 2 }, 2.5);

    try testing.expect(try tensor.get_f32(&[_]usize{ 0, 0 }) == 1.0);
    try testing.expect(try tensor.get_f32(&[_]usize{ 1, 2 }) == 2.5);
}

test "tensor utility functions" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test zeros
    const shape = [_]usize{ 2, 2 };
    var zeros_tensor = try zeros(allocator, &shape, .f32);
    defer zeros_tensor.deinit();

    try testing.expect(try zeros_tensor.get_f32(&[_]usize{ 0, 0 }) == 0.0);
    try testing.expect(try zeros_tensor.get_f32(&[_]usize{ 1, 1 }) == 0.0);

    // Test ones
    var ones_tensor = try ones(allocator, &shape, .f32);
    defer ones_tensor.deinit();

    try testing.expect(try ones_tensor.get_f32(&[_]usize{ 0, 0 }) == 1.0);
    try testing.expect(try ones_tensor.get_f32(&[_]usize{ 1, 1 }) == 1.0);
}

test "0D scalar tensor support" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test 0D scalar creation
    var scalar_tensor = try Tensor.init(allocator, &[_]usize{}, .f32);
    defer scalar_tensor.deinit();

    // Test scalar properties
    try testing.expect(scalar_tensor.numel() == 1);
    try testing.expect(scalar_tensor.ndim() == 0);
    try testing.expect(scalar_tensor.size_bytes() == 4);
    try testing.expect(scalar_tensor.is_scalar());

    // Test scalar value access
    try scalar_tensor.set_scalar_f32(42.5);
    try testing.expect(try scalar_tensor.get_scalar_f32() == 42.5);

    // Test scalar access with empty indices (should work)
    try scalar_tensor.set_f32(&[_]usize{}, 3.14);
    try testing.expect(try scalar_tensor.get_f32(&[_]usize{}) == 3.14);

    // Test flat access for scalar
    try scalar_tensor.set_f32_flat(0, 1.5);
    try testing.expect(try scalar_tensor.get_f32_flat(0) == 1.5);

    // Test error cases
    try testing.expectError(TensorError.IndexOutOfBounds, scalar_tensor.get_f32(&[_]usize{0}));
    try testing.expectError(TensorError.IndexOutOfBounds, scalar_tensor.set_f32(&[_]usize{0}, 1.0));
    try testing.expectError(TensorError.IndexOutOfBounds, scalar_tensor.get_f32_flat(1));
}

test "scalar tensor creation convenience function" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test f32 scalar creation
    var f32_scalar = try Tensor.scalar(allocator, 3.14, .f32);
    defer f32_scalar.deinit();

    try testing.expect(f32_scalar.is_scalar());
    try testing.expect(try f32_scalar.get_scalar_f32() == 3.14);

    // Test i32 scalar creation
    var i32_scalar = try Tensor.scalar(allocator, 42, .i32);
    defer i32_scalar.deinit();

    try testing.expect(i32_scalar.is_scalar());
    const data_ptr = @as([*]const i32, @ptrCast(@alignCast(i32_scalar.data.ptr)));
    try testing.expect(data_ptr[0] == 42);

    // Test type conversion
    var converted_scalar = try Tensor.scalar(allocator, 5, .f32); // int to f32
    defer converted_scalar.deinit();

    try testing.expect(try converted_scalar.get_scalar_f32() == 5.0);
}

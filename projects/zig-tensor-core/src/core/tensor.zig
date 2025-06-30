const std = @import("std");
const Allocator = std.mem.Allocator;

/// Data types supported by tensors
pub const DataType = enum {
    f32,
    f16,
    i32,
    i16,
    i8,
    u8,

    /// Get size in bytes for this data type
    pub fn size(self: DataType) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .i16 => 2,
            .i8, .u8 => 1,
        };
    }

    /// Get alignment requirement for this data type
    pub fn alignment(self: DataType) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .i16 => 2,
            .i8, .u8 => 1,
        };
    }
};

/// Device types where tensors can reside
pub const Device = enum {
    cpu,
    gpu,
    npu,
};

/// Tensor operation errors
pub const TensorError = error{
    InvalidShape,
    ShapeMismatch,
    IndexOutOfBounds,
    UnsupportedDataType,
    OutOfMemory,
    DeviceMismatch,
};

/// Multi-dimensional tensor with efficient memory layout
pub const Tensor = struct {
    data: []u8,
    shape: []const usize,
    strides: []const usize,
    dtype: DataType,
    device: Device,
    allocator: Allocator,

    const Self = @This();

    /// Initialize a new tensor with given shape and data type
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

    /// Initialize tensor from existing data slice
    pub fn fromSlice(allocator: Allocator, data_slice: anytype, shape: []const usize) !Self {
        const T = @TypeOf(data_slice[0]);
        const dtype = switch (T) {
            f32 => DataType.f32,
            f16 => DataType.f16,
            i32 => DataType.i32,
            i16 => DataType.i16,
            i8 => DataType.i8,
            u8 => DataType.u8,
            else => return TensorError.UnsupportedDataType,
        };

        var tensor = try Self.init(allocator, shape, dtype);

        // Copy data
        const expected_elements = tensor.numel();
        if (data_slice.len != expected_elements) {
            tensor.deinit();
            return TensorError.ShapeMismatch;
        }

        const typed_data = @as([*]T, @ptrCast(@alignCast(tensor.data.ptr)));
        @memcpy(typed_data[0..expected_elements], data_slice);

        return tensor;
    }

    /// Create a copy of this tensor
    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var tensor = try Self.init(allocator, self.shape, self.dtype);
        @memcpy(tensor.data, self.data);
        return tensor;
    }

    /// Free tensor memory
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
    }

    /// Get number of elements in tensor
    pub fn numel(self: *const Self) usize {
        // 0D scalar has 1 element
        if (self.shape.len == 0) return 1;

        var total: usize = 1;
        for (self.shape) |dim| {
            total *= dim;
        }
        return total;
    }

    /// Get number of dimensions
    pub fn ndim(self: *const Self) usize {
        return self.shape.len;
    }

    /// Get size in bytes
    pub fn sizeBytes(self: *const Self) usize {
        return self.numel() * self.dtype.size();
    }

    /// Compute byte offset for given indices
    fn computeOffset(self: *const Self, indices: []const usize) !usize {
        if (indices.len != self.shape.len) {
            return TensorError.IndexOutOfBounds;
        }

        // 0D scalar case
        if (self.shape.len == 0) {
            if (indices.len != 0) return TensorError.IndexOutOfBounds;
            return 0;
        }

        var offset: usize = 0;
        for (indices, self.shape, self.strides) |idx, dim, stride| {
            if (idx >= dim) return TensorError.IndexOutOfBounds;
            offset += idx * stride;
        }

        return offset * self.dtype.size();
    }

    /// Get f32 value at given indices
    pub fn getF32(self: *const Self, indices: []const usize) !f32 {
        if (self.dtype != .f32) return TensorError.UnsupportedDataType;
        const offset = try self.computeOffset(indices);
        const data_ptr = @as([*]const f32, @ptrCast(@alignCast(self.data.ptr)));
        return data_ptr[offset / 4];
    }

    /// Set f32 value at given indices
    pub fn setF32(self: *Self, indices: []const usize, value: f32) !void {
        if (self.dtype != .f32) return TensorError.UnsupportedDataType;
        const offset = try self.computeOffset(indices);
        const data_ptr = @as([*]f32, @ptrCast(@alignCast(self.data.ptr)));
        data_ptr[offset / 4] = value;
    }

    /// Get i32 value at given indices
    pub fn getI32(self: *const Self, indices: []const usize) !i32 {
        if (self.dtype != .i32) return TensorError.UnsupportedDataType;
        const offset = try self.computeOffset(indices);
        const data_ptr = @as([*]const i32, @ptrCast(@alignCast(self.data.ptr)));
        return data_ptr[offset / 4];
    }

    /// Set i32 value at given indices
    pub fn setI32(self: *Self, indices: []const usize, value: i32) !void {
        if (self.dtype != .i32) return TensorError.UnsupportedDataType;
        const offset = try self.computeOffset(indices);
        const data_ptr = @as([*]i32, @ptrCast(@alignCast(self.data.ptr)));
        data_ptr[offset / 4] = value;
    }

    /// Get raw data as typed slice
    pub fn dataAs(self: *const Self, comptime T: type) []T {
        const expected_dtype = switch (T) {
            f32 => DataType.f32,
            f16 => DataType.f16,
            i32 => DataType.i32,
            i16 => DataType.i16,
            i8 => DataType.i8,
            u8 => DataType.u8,
            else => @compileError("Unsupported data type"),
        };

        if (self.dtype != expected_dtype) {
            @panic("Data type mismatch");
        }

        const typed_ptr = @as([*]T, @ptrCast(@alignCast(self.data.ptr)));
        return typed_ptr[0..self.numel()];
    }

    /// Get mutable raw data as typed slice
    pub fn dataMutAs(self: *Self, comptime T: type) []T {
        const expected_dtype = switch (T) {
            f32 => DataType.f32,
            f16 => DataType.f16,
            i32 => DataType.i32,
            i16 => DataType.i16,
            i8 => DataType.i8,
            u8 => DataType.u8,
            else => @compileError("Unsupported data type"),
        };

        if (self.dtype != expected_dtype) {
            @panic("Data type mismatch");
        }

        const typed_ptr = @as([*]T, @ptrCast(@alignCast(self.data.ptr)));
        return typed_ptr[0..self.numel()];
    }

    /// Reshape tensor to new shape (must have same number of elements)
    pub fn reshape(self: *Self, new_shape: []const usize) !void {
        // Calculate new total elements
        var new_total: usize = 1;
        for (new_shape) |dim| {
            if (dim == 0) return TensorError.InvalidShape;
            new_total *= dim;
        }

        // Check if total elements match
        if (new_total != self.numel()) {
            return TensorError.ShapeMismatch;
        }

        // Free old shape and strides
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);

        // Allocate new shape and strides
        self.shape = try self.allocator.dupe(usize, new_shape);

        // Calculate new strides
        const strides = try self.allocator.alloc(usize, new_shape.len);
        var stride: usize = 1;
        var i = new_shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= new_shape[i];
        }
        self.strides = strides;
    }

    /// Fill tensor with a single value
    pub fn fill(self: *Self, value: anytype) !void {
        const T = @TypeOf(value);
        const expected_dtype = switch (T) {
            f32 => DataType.f32,
            f16 => DataType.f16,
            i32 => DataType.i32,
            i16 => DataType.i16,
            i8 => DataType.i8,
            u8 => DataType.u8,
            else => return TensorError.UnsupportedDataType,
        };

        if (self.dtype != expected_dtype) {
            return TensorError.UnsupportedDataType;
        }

        const data_slice = self.dataMutAs(T);
        for (data_slice) |*elem| {
            elem.* = value;
        }
    }

    /// Zero out all tensor data
    pub fn zero(self: *Self) void {
        @memset(self.data, 0);
    }

    /// Check if tensor shapes are compatible for element-wise operations
    pub fn isCompatibleWith(self: *const Self, other: *const Self) bool {
        if (self.shape.len != other.shape.len) return false;
        for (self.shape, other.shape) |dim1, dim2| {
            if (dim1 != dim2) return false;
        }
        return true;
    }
};

// Tests
test "tensor creation and basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test tensor creation
    const shape = [_]usize{ 2, 3 };
    var tensor = try Tensor.init(allocator, &shape, .f32);
    defer tensor.deinit();

    try testing.expect(tensor.numel() == 6);
    try testing.expect(tensor.ndim() == 2);
    try testing.expect(tensor.sizeBytes() == 24);

    // Test element access
    try tensor.setF32(&[_]usize{ 0, 0 }, 1.0);
    try tensor.setF32(&[_]usize{ 1, 2 }, 2.5);

    try testing.expect(try tensor.getF32(&[_]usize{ 0, 0 }) == 1.0);
    try testing.expect(try tensor.getF32(&[_]usize{ 1, 2 }) == 2.5);
}

test "0D scalar tensor" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test 0D scalar tensor
    const shape = [_]usize{};
    var scalar = try Tensor.init(allocator, &shape, .f32);
    defer scalar.deinit();

    try testing.expect(scalar.numel() == 1);
    try testing.expect(scalar.ndim() == 0);
    try testing.expect(scalar.sizeBytes() == 4);

    // Test scalar access
    try scalar.setF32(&[_]usize{}, 42.0);
    try testing.expect(try scalar.getF32(&[_]usize{}) == 42.0);
}

test "tensor from slice" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const shape = [_]usize{ 2, 3 };

    var tensor = try Tensor.fromSlice(allocator, &data, &shape);
    defer tensor.deinit();

    try testing.expect(tensor.numel() == 6);
    try testing.expect(try tensor.getF32(&[_]usize{ 0, 0 }) == 1.0);
    try testing.expect(try tensor.getF32(&[_]usize{ 1, 2 }) == 6.0);
}

test "tensor reshape" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 }, .f32);
    defer tensor.deinit();

    // Fill with test data
    try tensor.fill(@as(f32, 5.0));

    // Reshape to 1D
    try tensor.reshape(&[_]usize{6});
    try testing.expect(tensor.shape.len == 1);
    try testing.expect(tensor.shape[0] == 6);
    try testing.expect(tensor.numel() == 6);

    // Reshape to 3x2
    try tensor.reshape(&[_]usize{ 3, 2 });
    try testing.expect(tensor.shape.len == 2);
    try testing.expect(tensor.shape[0] == 3);
    try testing.expect(tensor.shape[1] == 2);
}

test "tensor fill and zero" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var tensor = try Tensor.init(allocator, &[_]usize{ 2, 2 }, .f32);
    defer tensor.deinit();

    // Test fill
    try tensor.fill(@as(f32, 3.14));
    const data = tensor.dataAs(f32);
    for (data) |val| {
        try testing.expect(val == 3.14);
    }

    // Test zero
    tensor.zero();
    for (data) |val| {
        try testing.expect(val == 0.0);
    }
}

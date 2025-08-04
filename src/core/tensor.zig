const std = @import("std");

/// Data types supported by tensors
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
};

/// Shape of a tensor - comptime known for performance
pub fn Shape(comptime rank: usize) type {
    return struct {
        const Self = @This();

        dims: [rank]usize,

        pub fn init(dims: [rank]usize) Self {
            return Self{ .dims = dims };
        }

        pub fn numel(self: Self) usize {
            var total: usize = 1;
            for (self.dims) |dim| {
                total *= dim;
            }
            return total;
        }

        pub fn getRank() usize {
            return rank;
        }
    };
}

/// Universal tensor type with explicit memory management
pub fn Tensor(comptime T: type, comptime shape: anytype) type {
    const ShapeType = @TypeOf(shape);

    return struct {
        const Self = @This();

        data: []T,
        shape: ShapeType,
        allocator: std.mem.Allocator,
        owns_data: bool,

        pub fn init(allocator: std.mem.Allocator, tensor_shape: ShapeType) !Self {
            const total_elements = tensor_shape.numel();
            const data = try allocator.alloc(T, total_elements);

            return Self{
                .data = data,
                .shape = tensor_shape,
                .allocator = allocator,
                .owns_data = true,
            };
        }

        pub fn initFromSlice(data: []T, tensor_shape: ShapeType) Self {
            std.debug.assert(data.len == tensor_shape.numel());

            return Self{
                .data = data,
                .shape = tensor_shape,
                .allocator = undefined, // Not used for non-owning tensors
                .owns_data = false,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.owns_data) {
                self.allocator.free(self.data);
            }
        }

        pub fn get(self: Self, indices: anytype) T {
            const index = self.computeIndex(indices);
            return self.data[index];
        }

        pub fn set(self: *Self, indices: anytype, value: T) void {
            const index = self.computeIndex(indices);
            self.data[index] = value;
        }

        fn computeIndex(self: Self, indices: anytype) usize {
            const indices_info = @typeInfo(@TypeOf(indices));
            comptime std.debug.assert(indices_info == .Struct);

            var index: usize = 0;
            var stride: usize = 1;

            // Compute index in row-major order
            comptime var i = self.shape.dims.len;
            inline while (i > 0) {
                i -= 1;
                const dim_index = @field(indices, std.fmt.comptimePrint("{d}", .{i}));
                index += dim_index * stride;
                stride *= self.shape.dims[i];
            }

            return index;
        }

        pub fn fill(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        pub fn copy(self: Self, dest: *Self) void {
            std.debug.assert(self.data.len == dest.data.len);
            @memcpy(dest.data, self.data);
        }
    };
}

/// Dynamic tensor for runtime-determined shapes
pub const DynamicTensor = struct {
    data: []u8,
    shape: []usize,
    dtype: DataType,
    allocator: std.mem.Allocator,
    owns_data: bool,

    pub fn init(allocator: std.mem.Allocator, dtype: DataType, shape: []const usize) !DynamicTensor {
        var total_elements: usize = 1;
        for (shape) |dim| {
            total_elements *= dim;
        }

        const data_size = total_elements * dtype.size();
        const data = try allocator.alloc(u8, data_size);
        const shape_copy = try allocator.dupe(usize, shape);

        return DynamicTensor{
            .data = data,
            .shape = shape_copy,
            .dtype = dtype,
            .allocator = allocator,
            .owns_data = true,
        };
    }

    pub fn initFromSlice(data: []u8, dtype: DataType, shape: []const usize, allocator: std.mem.Allocator) !DynamicTensor {
        var total_elements: usize = 1;
        for (shape) |dim| {
            total_elements *= dim;
        }

        std.debug.assert(data.len == total_elements * dtype.size());

        const shape_copy = try allocator.dupe(usize, shape);

        return DynamicTensor{
            .data = data,
            .shape = shape_copy,
            .dtype = dtype,
            .allocator = allocator,
            .owns_data = false,
        };
    }

    pub fn deinit(self: *DynamicTensor) void {
        if (self.owns_data) {
            self.allocator.free(self.data);
        }
        self.allocator.free(self.shape);
    }

    pub fn numel(self: DynamicTensor) usize {
        var total: usize = 1;
        for (self.shape) |dim| {
            total *= dim;
        }
        return total;
    }

    pub fn getTypedData(self: DynamicTensor, comptime T: type) []T {
        const expected_size = @sizeOf(T);
        std.debug.assert(expected_size == self.dtype.size());

        const typed_len = self.data.len / expected_size;
        return @as([*]T, @ptrCast(@alignCast(self.data.ptr)))[0..typed_len];
    }
};

test "tensor creation and access" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test static tensor
    const shape = Shape(2).init(.{ 3, 4 });
    var tensor = try Tensor(f32, shape).init(allocator, shape);
    defer tensor.deinit();

    tensor.set(.{ 1, 2 }, 42.0);
    try testing.expect(tensor.get(.{ 1, 2 }) == 42.0);

    // Test dynamic tensor
    var dyn_tensor = try DynamicTensor.init(allocator, .f32, &[_]usize{ 2, 3 });
    defer dyn_tensor.deinit();

    try testing.expect(dyn_tensor.numel() == 6);
    try testing.expect(dyn_tensor.shape.len == 2);
}

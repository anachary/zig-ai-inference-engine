const std = @import("std");
const Allocator = std.mem.Allocator;

/// Simple, clean tensor interface for the Zig AI Platform
pub const Tensor = struct {
    data: []u8,
    shape: []const usize,
    dtype: DataType,
    strides: []const usize,
    allocator: Allocator,

    /// Supported data types
    pub const DataType = enum {
        f32,
        f16,
        i32,
        i16,
        i8,
        u8,
        bool,
        f64,
        i64,
        u32,
        u16,
        u64,

        pub fn size(self: DataType) usize {
            return switch (self) {
                .f32, .i32, .u32 => 4,
                .f16, .i16, .u16 => 2,
                .i8, .u8, .bool => 1,
                .f64, .i64, .u64 => 8,
            };
        }
    };

    /// Create a new tensor with given shape and data type
    pub fn init(allocator: Allocator, shape: []const usize, dtype: DataType) !Tensor {
        const element_size = dtype.size();
        const total_elements = calculateTotalElements(shape);
        const data = try allocator.alloc(u8, total_elements * element_size);
        
        const owned_shape = try allocator.dupe(usize, shape);
        const strides = try calculateStrides(allocator, shape);
        
        return Tensor{
            .data = data,
            .shape = owned_shape,
            .dtype = dtype,
            .strides = strides,
            .allocator = allocator,
        };
    }

    /// Free tensor memory
    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
    }

    /// Get tensor data as typed slice (read-only)
    pub fn getData(self: *const Tensor, comptime T: type) []const T {
        return std.mem.bytesAsSlice(T, self.data);
    }

    /// Get tensor data as typed slice (mutable)
    pub fn getMutableData(self: *Tensor, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.data);
    }

    /// Get total number of elements
    pub fn getElementCount(self: *const Tensor) usize {
        return calculateTotalElements(self.shape);
    }

    /// Set tensor data from slice
    pub fn setData(self: *Tensor, comptime T: type, data: []const T) !void {
        const tensor_data = self.getMutableData(T);
        if (tensor_data.len != data.len) {
            return error.ShapeMismatch;
        }
        @memcpy(tensor_data, data);
    }

    /// Copy data from another tensor
    pub fn copyFrom(self: *Tensor, other: *const Tensor) !void {
        if (self.data.len != other.data.len) {
            return error.ShapeMismatch;
        }
        if (self.dtype != other.dtype) {
            return error.DataTypeMismatch;
        }
        @memcpy(self.data, other.data);
    }

    /// Check if two tensors have the same shape
    pub fn sameShape(self: *const Tensor, other: *const Tensor) bool {
        return shapesEqual(self.shape, other.shape);
    }

    /// Fill tensor with a single value
    pub fn fill(self: *Tensor, comptime T: type, value: T) void {
        const data = self.getMutableData(T);
        for (data) |*element| {
            element.* = value;
        }
    }

    /// Zero out the tensor
    pub fn zero(self: *Tensor) void {
        @memset(self.data, 0);
    }

    // Helper functions
    fn calculateTotalElements(shape: []const usize) usize {
        var total: usize = 1;
        for (shape) |dim| {
            total *= dim;
        }
        return total;
    }

    fn calculateStrides(allocator: Allocator, shape: []const usize) ![]usize {
        const strides = try allocator.alloc(usize, shape.len);
        if (shape.len == 0) return strides;
        
        strides[shape.len - 1] = 1;
        var i = shape.len - 1;
        while (i > 0) {
            i -= 1;
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
};

/// Attributes for operators (simplified)
pub const Attributes = struct {
    map: std.StringHashMap(AttributeValue),
    allocator: Allocator,

    pub const AttributeValue = union(enum) {
        int: i64,
        float: f64,
        string: []const u8,
        ints: []const i64,
        floats: []const f64,
        strings: []const []const u8,
    };

    pub fn init(allocator: Allocator) Attributes {
        return Attributes{
            .map = std.StringHashMap(AttributeValue).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Attributes) void {
        self.map.deinit();
    }

    pub fn set(self: *Attributes, key: []const u8, value: AttributeValue) !void {
        try self.map.put(key, value);
    }

    pub fn get(self: *const Attributes, key: []const u8) ?AttributeValue {
        return self.map.get(key);
    }

    pub fn getInt(self: *const Attributes, key: []const u8, default: i64) i64 {
        if (self.get(key)) |value| {
            return switch (value) {
                .int => |i| i,
                else => default,
            };
        }
        return default;
    }

    pub fn getFloat(self: *const Attributes, key: []const u8, default: f64) f64 {
        if (self.get(key)) |value| {
            return switch (value) {
                .float => |f| f,
                else => default,
            };
        }
        return default;
    }

    pub fn getString(self: *const Attributes, key: []const u8, default: []const u8) []const u8 {
        if (self.get(key)) |value| {
            return switch (value) {
                .string => |s| s,
                else => default,
            };
        }
        return default;
    }
};

/// Execution context (simplified)
pub const ExecutionContext = struct {
    allocator: Allocator,
    device: Device = .cpu,
    profiling_enabled: bool = false,

    pub const Device = enum {
        cpu,
        gpu,
        auto,
    };

    pub fn init(allocator: Allocator) ExecutionContext {
        return ExecutionContext{
            .allocator = allocator,
        };
    }

    pub fn createTensor(self: *ExecutionContext, shape: []const usize, dtype: Tensor.DataType) !Tensor {
        return Tensor.init(self.allocator, shape, dtype);
    }
};

/// Utility functions
pub fn createTensor(allocator: Allocator, shape: []const usize, dtype: Tensor.DataType) !Tensor {
    return Tensor.init(allocator, shape, dtype);
}

pub fn createAttributes(allocator: Allocator) Attributes {
    return Attributes.init(allocator);
}

pub fn createExecutionContext(allocator: Allocator) ExecutionContext {
    return ExecutionContext.init(allocator);
}

pub fn shapesEqual(shape1: []const usize, shape2: []const usize) bool {
    if (shape1.len != shape2.len) return false;
    
    for (shape1, shape2) |dim1, dim2| {
        if (dim1 != dim2) return false;
    }
    return true;
}

pub fn calculateTotalElements(shape: []const usize) usize {
    var total: usize = 1;
    for (shape) |dim| {
        total *= dim;
    }
    return total;
}

/// Check if two shapes are broadcast compatible
pub fn checkBroadcastCompatibility(shape1: []const usize, shape2: []const usize) bool {
    const max_dims = @max(shape1.len, shape2.len);
    
    var i: usize = 0;
    while (i < max_dims) : (i += 1) {
        const dim1 = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
        const dim2 = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;
        
        if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
            return false;
        }
    }
    return true;
}

/// Calculate broadcast output shape
pub fn calculateBroadcastShape(shape1: []const usize, shape2: []const usize, allocator: Allocator) ![]usize {
    const max_dims = @max(shape1.len, shape2.len);
    const output_shape = try allocator.alloc(usize, max_dims);
    
    var i: usize = 0;
    while (i < max_dims) : (i += 1) {
        const dim1 = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
        const dim2 = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;
        
        output_shape[max_dims - 1 - i] = @max(dim1, dim2);
    }
    
    return output_shape;
}

// Tests
test "tensor creation and basic operations" {
    const allocator = std.testing.allocator;
    
    const shape = [_]usize{ 2, 3 };
    var tensor = try createTensor(allocator, &shape, .f32);
    defer tensor.deinit();
    
    // Test basic properties
    try std.testing.expect(tensor.shape.len == 2);
    try std.testing.expect(tensor.shape[0] == 2);
    try std.testing.expect(tensor.shape[1] == 3);
    try std.testing.expect(tensor.dtype == .f32);
    try std.testing.expect(tensor.getElementCount() == 6);
    
    // Test data access
    const data = tensor.getMutableData(f32);
    try std.testing.expect(data.len == 6);
    
    // Test fill
    tensor.fill(f32, 3.14);
    const filled_data = tensor.getData(f32);
    for (filled_data) |value| {
        try std.testing.expect(value == 3.14);
    }
    
    // Test setData
    const test_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try tensor.setData(f32, &test_data);
    
    const retrieved_data = tensor.getData(f32);
    try std.testing.expectEqualSlices(f32, &test_data, retrieved_data);
}

test "attributes" {
    const allocator = std.testing.allocator;
    
    var attrs = createAttributes(allocator);
    defer attrs.deinit();
    
    try attrs.set("int_attr", Attributes.AttributeValue{ .int = 42 });
    try attrs.set("float_attr", Attributes.AttributeValue{ .float = 3.14 });
    try attrs.set("string_attr", Attributes.AttributeValue{ .string = "test" });
    
    try std.testing.expect(attrs.getInt("int_attr", 0) == 42);
    try std.testing.expect(attrs.getFloat("float_attr", 0.0) == 3.14);
    try std.testing.expectEqualStrings(attrs.getString("string_attr", ""), "test");
    
    // Test default values
    try std.testing.expect(attrs.getInt("nonexistent", 100) == 100);
}

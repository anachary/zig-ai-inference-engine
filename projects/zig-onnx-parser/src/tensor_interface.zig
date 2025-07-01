const std = @import("std");

/// Data types supported by tensors
pub const DataType = enum {
    f32,
    f64,
    i32,
    i64,
    u32,
    u64,
    i8,
    u8,
    i16,
    u16,
    bool,
    string,
    
    pub fn toString(self: DataType) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .f64 => "f64",
            .i32 => "i32",
            .i64 => "i64",
            .u32 => "u32",
            .u64 => "u64",
            .i8 => "i8",
            .u8 => "u8",
            .i16 => "i16",
            .u16 => "u16",
            .bool => "bool",
            .string => "string",
        };
    }
    
    pub fn sizeInBytes(self: DataType) usize {
        return switch (self) {
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
            .i8, .u8, .bool => 1,
            .i16, .u16 => 2,
            .string => 0, // Variable size
        };
    }
};

/// Tensor shape representation
pub const TensorShape = struct {
    dims: []const i64,
    
    pub fn init(allocator: std.mem.Allocator, dimensions: []const i64) !TensorShape {
        const dims = try allocator.dupe(i64, dimensions);
        return TensorShape{ .dims = dims };
    }
    
    pub fn deinit(self: *TensorShape, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
    }
    
    pub fn totalElements(self: TensorShape) i64 {
        var total: i64 = 1;
        for (self.dims) |dim| {
            if (dim > 0) total *= dim;
        }
        return total;
    }
};

/// Basic tensor interface for ONNX compatibility
pub const TensorInterface = struct {
    data_type: DataType,
    shape: TensorShape,
    data: ?[]const u8,
    
    pub fn init(allocator: std.mem.Allocator, data_type: DataType, shape_dims: []const i64) !TensorInterface {
        const shape = try TensorShape.init(allocator, shape_dims);
        return TensorInterface{
            .data_type = data_type,
            .shape = shape,
            .data = null,
        };
    }
    
    pub fn deinit(self: *TensorInterface, allocator: std.mem.Allocator) void {
        self.shape.deinit(allocator);
        if (self.data) |data| {
            allocator.free(data);
        }
    }
};

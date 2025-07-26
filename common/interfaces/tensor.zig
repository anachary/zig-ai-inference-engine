const std = @import("std");
const Allocator = std.mem.Allocator;

/// Common tensor interface that all projects must implement
/// This ensures compatibility across the entire ecosystem
pub const TensorInterface = struct {
    /// Data type enumeration
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

    /// Device enumeration
    pub const Device = enum {
        cpu,
        gpu,
        npu,
    };

    /// Tensor errors
    pub const TensorError = error{
        InvalidShape,
        ShapeMismatch,
        IndexOutOfBounds,
        UnsupportedDataType,
        OutOfMemory,
        DeviceMismatch,
    };

    /// Core tensor operations that must be implemented
    pub const Operations = struct {
        /// Initialize a tensor with given shape and data type
        initFn: *const fn (allocator: Allocator, shape: []const usize, dtype: DataType) TensorError!*anyopaque,

        /// Deinitialize a tensor
        deinitFn: *const fn (tensor: *anyopaque) void,

        /// Get tensor shape
        shapeFn: *const fn (tensor: *const anyopaque) []const usize,

        /// Get tensor data type
        dtypeFn: *const fn (tensor: *const anyopaque) DataType,

        /// Get tensor device
        deviceFn: *const fn (tensor: *const anyopaque) Device,

        /// Get number of elements
        numelFn: *const fn (tensor: *const anyopaque) usize,

        /// Get number of dimensions
        ndimFn: *const fn (tensor: *const anyopaque) usize,

        /// Get size in bytes
        sizeByteFn: *const fn (tensor: *const anyopaque) usize,

        /// Get raw data pointer
        dataFn: *const fn (tensor: *const anyopaque) []u8,

        /// Set f32 value at index
        setF32Fn: *const fn (tensor: *anyopaque, indices: []const usize, value: f32) TensorError!void,

        /// Get f32 value at index
        getF32Fn: *const fn (tensor: *const anyopaque, indices: []const usize) TensorError!f32,

        /// Copy tensor
        copyFn: *const fn (allocator: Allocator, tensor: *const anyopaque) TensorError!*anyopaque,

        /// Reshape tensor
        reshapeFn: *const fn (tensor: *anyopaque, new_shape: []const usize) TensorError!void,
    };

    /// Tensor implementation
    impl: Operations,
    ptr: *anyopaque,

    /// Initialize tensor
    pub fn init(allocator: Allocator, tensor_shape: []const usize, data_type: DataType, impl: Operations) TensorError!TensorInterface {
        const ptr = try impl.initFn(allocator, tensor_shape, data_type);
        return TensorInterface{
            .impl = impl,
            .ptr = ptr,
        };
    }

    /// Deinitialize tensor
    pub fn deinit(self: *TensorInterface) void {
        self.impl.deinitFn(self.ptr);
    }

    /// Get shape
    pub fn shape(self: *const TensorInterface) []const usize {
        return self.impl.shapeFn(self.ptr);
    }

    /// Get data type
    pub fn dtype(self: *const TensorInterface) DataType {
        return self.impl.dtypeFn(self.ptr);
    }

    /// Get device
    pub fn device(self: *const TensorInterface) Device {
        return self.impl.deviceFn(self.ptr);
    }

    /// Get number of elements
    pub fn numel(self: *const TensorInterface) usize {
        return self.impl.numelFn(self.ptr);
    }

    /// Get number of dimensions
    pub fn ndim(self: *const TensorInterface) usize {
        return self.impl.ndimFn(self.ptr);
    }

    /// Get size in bytes
    pub fn sizeBytes(self: *const TensorInterface) usize {
        return self.impl.sizeByteFn(self.ptr);
    }

    /// Get raw data
    pub fn data(self: *const TensorInterface) []u8 {
        return self.impl.dataFn(self.ptr);
    }

    /// Set f32 value
    pub fn setF32(self: *TensorInterface, indices: []const usize, value: f32) TensorError!void {
        return self.impl.setF32Fn(self.ptr, indices, value);
    }

    /// Get f32 value
    pub fn getF32(self: *const TensorInterface, indices: []const usize) TensorError!f32 {
        return self.impl.getF32Fn(self.ptr, indices);
    }

    /// Copy tensor
    pub fn copy(self: *const TensorInterface, allocator: Allocator) TensorError!TensorInterface {
        const new_ptr = try self.impl.copyFn(allocator, self.ptr);
        return TensorInterface{
            .impl = self.impl,
            .ptr = new_ptr,
        };
    }

    /// Reshape tensor
    pub fn reshape(self: *TensorInterface, new_shape: []const usize) TensorError!void {
        return self.impl.reshapeFn(self.ptr, new_shape);
    }
};

/// Math operations interface
pub const MathInterface = struct {
    /// Add two tensors
    addFn: *const fn (allocator: Allocator, a: *const TensorInterface, b: *const TensorInterface) TensorInterface.TensorError!TensorInterface,

    /// Subtract two tensors
    subFn: *const fn (allocator: Allocator, a: *const TensorInterface, b: *const TensorInterface) TensorInterface.TensorError!TensorInterface,

    /// Multiply two tensors
    mulFn: *const fn (allocator: Allocator, a: *const TensorInterface, b: *const TensorInterface) TensorInterface.TensorError!TensorInterface,

    /// Matrix multiplication
    matmulFn: *const fn (allocator: Allocator, a: *const TensorInterface, b: *const TensorInterface) TensorInterface.TensorError!TensorInterface,

    /// Apply activation function
    activationFn: *const fn (allocator: Allocator, input: *const TensorInterface, activation_type: ActivationType) TensorInterface.TensorError!TensorInterface,

    pub const ActivationType = enum {
        relu,
        sigmoid,
        tanh,
        softmax,
        gelu,
    };
};

/// SIMD operations interface
pub const SIMDInterface = struct {
    /// Check if SIMD is available
    isAvailableFn: *const fn () bool,

    /// Get SIMD vector width for data type
    vectorWidthFn: *const fn (dtype: TensorInterface.DataType) usize,

    /// Vectorized add operation
    vectorAddFn: *const fn (a: []const u8, b: []const u8, result: []u8, dtype: TensorInterface.DataType) void,

    /// Vectorized multiply operation
    vectorMulFn: *const fn (a: []const u8, b: []const u8, result: []u8, dtype: TensorInterface.DataType) void,
};

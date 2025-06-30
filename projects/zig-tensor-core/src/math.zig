const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("core/tensor.zig");
const simd = @import("core/simd.zig");
const shape = @import("core/shape.zig");

const Tensor = tensor.Tensor;
const TensorError = tensor.TensorError;

/// Math operation errors
pub const MathError = error{
    ShapeMismatch,
    UnsupportedDataType,
    InvalidOperation,
    OutOfMemory,
} || TensorError;

/// Element-wise addition of two tensors
pub fn add(allocator: Allocator, a: Tensor, b: Tensor) MathError!Tensor {
    if (!a.isCompatibleWith(&b)) {
        return MathError.ShapeMismatch;
    }
    
    if (a.dtype != b.dtype) {
        return MathError.UnsupportedDataType;
    }
    
    var result = try Tensor.init(allocator, a.shape, a.dtype);
    
    switch (a.dtype) {
        .f32 => {
            const a_data = a.dataAs(f32);
            const b_data = b.dataAs(f32);
            const result_data = result.dataMutAs(f32);
            
            if (simd.isAvailable()) {
                try simd.vectorAddF32(a_data, b_data, result_data);
            } else {
                for (a_data, b_data, result_data) |a_val, b_val, *r| {
                    r.* = a_val + b_val;
                }
            }
        },
        .i32 => {
            const a_data = a.dataAs(i32);
            const b_data = b.dataAs(i32);
            const result_data = result.dataMutAs(i32);
            
            for (a_data, b_data, result_data) |a_val, b_val, *r| {
                r.* = a_val + b_val;
            }
        },
        else => return MathError.UnsupportedDataType,
    }
    
    return result;
}

/// Element-wise subtraction of two tensors
pub fn sub(allocator: Allocator, a: Tensor, b: Tensor) MathError!Tensor {
    if (!a.isCompatibleWith(&b)) {
        return MathError.ShapeMismatch;
    }
    
    if (a.dtype != b.dtype) {
        return MathError.UnsupportedDataType;
    }
    
    var result = try Tensor.init(allocator, a.shape, a.dtype);
    
    switch (a.dtype) {
        .f32 => {
            const a_data = a.dataAs(f32);
            const b_data = b.dataAs(f32);
            const result_data = result.dataMutAs(f32);
            
            if (simd.isAvailable()) {
                try simd.vectorSubF32(a_data, b_data, result_data);
            } else {
                for (a_data, b_data, result_data) |a_val, b_val, *r| {
                    r.* = a_val - b_val;
                }
            }
        },
        .i32 => {
            const a_data = a.dataAs(i32);
            const b_data = b.dataAs(i32);
            const result_data = result.dataMutAs(i32);
            
            for (a_data, b_data, result_data) |a_val, b_val, *r| {
                r.* = a_val - b_val;
            }
        },
        else => return MathError.UnsupportedDataType,
    }
    
    return result;
}

/// Element-wise multiplication of two tensors
pub fn mul(allocator: Allocator, a: Tensor, b: Tensor) MathError!Tensor {
    if (!a.isCompatibleWith(&b)) {
        return MathError.ShapeMismatch;
    }
    
    if (a.dtype != b.dtype) {
        return MathError.UnsupportedDataType;
    }
    
    var result = try Tensor.init(allocator, a.shape, a.dtype);
    
    switch (a.dtype) {
        .f32 => {
            const a_data = a.dataAs(f32);
            const b_data = b.dataAs(f32);
            const result_data = result.dataMutAs(f32);
            
            if (simd.isAvailable()) {
                try simd.vectorMulF32(a_data, b_data, result_data);
            } else {
                for (a_data, b_data, result_data) |a_val, b_val, *r| {
                    r.* = a_val * b_val;
                }
            }
        },
        .i32 => {
            const a_data = a.dataAs(i32);
            const b_data = b.dataAs(i32);
            const result_data = result.dataMutAs(i32);
            
            for (a_data, b_data, result_data) |a_val, b_val, *r| {
                r.* = a_val * b_val;
            }
        },
        else => return MathError.UnsupportedDataType,
    }
    
    return result;
}

/// Element-wise division of two tensors
pub fn div(allocator: Allocator, a: Tensor, b: Tensor) MathError!Tensor {
    if (!a.isCompatibleWith(&b)) {
        return MathError.ShapeMismatch;
    }
    
    if (a.dtype != b.dtype) {
        return MathError.UnsupportedDataType;
    }
    
    var result = try Tensor.init(allocator, a.shape, a.dtype);
    
    switch (a.dtype) {
        .f32 => {
            const a_data = a.dataAs(f32);
            const b_data = b.dataAs(f32);
            const result_data = result.dataMutAs(f32);
            
            if (simd.isAvailable()) {
                try simd.vectorDivF32(a_data, b_data, result_data);
            } else {
                for (a_data, b_data, result_data) |a_val, b_val, *r| {
                    r.* = a_val / b_val;
                }
            }
        },
        else => return MathError.UnsupportedDataType,
    }
    
    return result;
}

/// Matrix multiplication of two 2D tensors
pub fn matmul(allocator: Allocator, a: Tensor, b: Tensor) MathError!Tensor {
    if (a.ndim() != 2 or b.ndim() != 2) {
        return MathError.InvalidOperation;
    }
    
    if (a.shape[1] != b.shape[0]) {
        return MathError.ShapeMismatch;
    }
    
    if (a.dtype != b.dtype) {
        return MathError.UnsupportedDataType;
    }
    
    const result_shape = [_]usize{ a.shape[0], b.shape[1] };
    var result = try Tensor.init(allocator, &result_shape, a.dtype);
    
    switch (a.dtype) {
        .f32 => {
            const a_data = a.dataAs(f32);
            const b_data = b.dataAs(f32);
            const result_data = result.dataMutAs(f32);
            
            // Initialize result to zero
            @memset(result_data, 0);
            
            // Perform matrix multiplication
            for (0..a.shape[0]) |i| {
                for (0..b.shape[1]) |j| {
                    var sum: f32 = 0.0;
                    for (0..a.shape[1]) |k| {
                        const a_idx = i * a.shape[1] + k;
                        const b_idx = k * b.shape[1] + j;
                        sum += a_data[a_idx] * b_data[b_idx];
                    }
                    const result_idx = i * b.shape[1] + j;
                    result_data[result_idx] = sum;
                }
            }
        },
        else => return MathError.UnsupportedDataType,
    }
    
    return result;
}

/// Transpose a 2D tensor
pub fn transpose(allocator: Allocator, input: Tensor) MathError!Tensor {
    if (input.ndim() != 2) {
        return MathError.InvalidOperation;
    }
    
    const result_shape = [_]usize{ input.shape[1], input.shape[0] };
    var result = try Tensor.init(allocator, &result_shape, input.dtype);
    
    switch (input.dtype) {
        .f32 => {
            const input_data = input.dataAs(f32);
            const result_data = result.dataMutAs(f32);
            
            for (0..input.shape[0]) |i| {
                for (0..input.shape[1]) |j| {
                    const input_idx = i * input.shape[1] + j;
                    const result_idx = j * input.shape[0] + i;
                    result_data[result_idx] = input_data[input_idx];
                }
            }
        },
        .i32 => {
            const input_data = input.dataAs(i32);
            const result_data = result.dataMutAs(i32);
            
            for (0..input.shape[0]) |i| {
                for (0..input.shape[1]) |j| {
                    const input_idx = i * input.shape[1] + j;
                    const result_idx = j * input.shape[0] + i;
                    result_data[result_idx] = input_data[input_idx];
                }
            }
        },
        else => return MathError.UnsupportedDataType,
    }
    
    return result;
}

/// Sum all elements in a tensor
pub fn sum(allocator: Allocator, input: Tensor) MathError!Tensor {
    const result_shape = [_]usize{}; // 0D scalar
    var result = try Tensor.init(allocator, &result_shape, input.dtype);
    
    switch (input.dtype) {
        .f32 => {
            const input_data = input.dataAs(f32);
            var total: f32 = 0.0;
            
            if (simd.isAvailable()) {
                total = simd.vectorSumF32(input_data);
            } else {
                for (input_data) |val| {
                    total += val;
                }
            }
            
            try result.setF32(&[_]usize{}, total);
        },
        .i32 => {
            const input_data = input.dataAs(i32);
            var total: i32 = 0;
            
            for (input_data) |val| {
                total += val;
            }
            
            try result.setI32(&[_]usize{}, total);
        },
        else => return MathError.UnsupportedDataType,
    }
    
    return result;
}

/// Compute mean of all elements in a tensor
pub fn mean(allocator: Allocator, input: Tensor) MathError!Tensor {
    const sum_result = try sum(allocator, input);
    defer sum_result.deinit();
    
    const result_shape = [_]usize{}; // 0D scalar
    var result = try Tensor.init(allocator, &result_shape, input.dtype);
    
    switch (input.dtype) {
        .f32 => {
            const sum_val = try sum_result.getF32(&[_]usize{});
            const mean_val = sum_val / @as(f32, @floatFromInt(input.numel()));
            try result.setF32(&[_]usize{}, mean_val);
        },
        .i32 => {
            const sum_val = try sum_result.getI32(&[_]usize{});
            const mean_val = @divTrunc(sum_val, @as(i32, @intCast(input.numel())));
            try result.setI32(&[_]usize{}, mean_val);
        },
        else => return MathError.UnsupportedDataType,
    }
    
    return result;
}

/// Scalar multiplication (broadcast scalar to all elements)
pub fn scalarMul(allocator: Allocator, input: Tensor, scalar: anytype) MathError!Tensor {
    var result = try Tensor.init(allocator, input.shape, input.dtype);
    
    const T = @TypeOf(scalar);
    const expected_dtype = switch (T) {
        f32 => tensor.DataType.f32,
        i32 => tensor.DataType.i32,
        else => return MathError.UnsupportedDataType,
    };
    
    if (input.dtype != expected_dtype) {
        return MathError.UnsupportedDataType;
    }
    
    switch (T) {
        f32 => {
            const input_data = input.dataAs(f32);
            const result_data = result.dataMutAs(f32);
            
            if (simd.isAvailable()) {
                try simd.vectorScalarMulF32(scalar, input_data, result_data);
            } else {
                for (input_data, result_data) |val, *r| {
                    r.* = scalar * val;
                }
            }
        },
        i32 => {
            const input_data = input.dataAs(i32);
            const result_data = result.dataMutAs(i32);
            
            for (input_data, result_data) |val, *r| {
                r.* = scalar * val;
            }
        },
        else => return MathError.UnsupportedDataType,
    }
    
    return result;
}

// Tests
test "tensor math operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    // Create test tensors
    const shape = [_]usize{ 2, 2 };
    var a = try Tensor.init(allocator, &shape, .f32);
    defer a.deinit();
    var b = try Tensor.init(allocator, &shape, .f32);
    defer b.deinit();
    
    // Fill with test data
    try a.fill(@as(f32, 2.0));
    try b.fill(@as(f32, 3.0));
    
    // Test addition
    var add_result = try add(allocator, a, b);
    defer add_result.deinit();
    try testing.expect(try add_result.getF32(&[_]usize{ 0, 0 }) == 5.0);
    
    // Test multiplication
    var mul_result = try mul(allocator, a, b);
    defer mul_result.deinit();
    try testing.expect(try mul_result.getF32(&[_]usize{ 0, 0 }) == 6.0);
    
    // Test sum
    var sum_result = try sum(allocator, a);
    defer sum_result.deinit();
    try testing.expect(try sum_result.getF32(&[_]usize{}) == 8.0); // 2.0 * 4 elements
}

test "matrix multiplication" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    // Create 2x3 and 3x2 matrices
    var a = try Tensor.init(allocator, &[_]usize{ 2, 3 }, .f32);
    defer a.deinit();
    var b = try Tensor.init(allocator, &[_]usize{ 3, 2 }, .f32);
    defer b.deinit();
    
    // Fill with test data
    try a.fill(@as(f32, 1.0));
    try b.fill(@as(f32, 2.0));
    
    // Test matrix multiplication
    var result = try matmul(allocator, a, b);
    defer result.deinit();
    
    try testing.expect(result.shape[0] == 2);
    try testing.expect(result.shape[1] == 2);
    try testing.expect(try result.getF32(&[_]usize{ 0, 0 }) == 6.0); // 1*2 + 1*2 + 1*2
}

test "transpose operation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var input = try Tensor.init(allocator, &[_]usize{ 2, 3 }, .f32);
    defer input.deinit();
    
    // Set some test values
    try input.setF32(&[_]usize{ 0, 0 }, 1.0);
    try input.setF32(&[_]usize{ 0, 1 }, 2.0);
    try input.setF32(&[_]usize{ 1, 0 }, 3.0);
    
    var result = try transpose(allocator, input);
    defer result.deinit();
    
    try testing.expect(result.shape[0] == 3);
    try testing.expect(result.shape[1] == 2);
    try testing.expect(try result.getF32(&[_]usize{ 0, 0 }) == 1.0);
    try testing.expect(try result.getF32(&[_]usize{ 1, 0 }) == 2.0);
    try testing.expect(try result.getF32(&[_]usize{ 0, 1 }) == 3.0);
}

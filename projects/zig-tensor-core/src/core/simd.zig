const std = @import("std");
const builtin = @import("builtin");

/// SIMD operation errors
pub const SIMDError = error{
    InvalidLength,
    UnsupportedOperation,
    AlignmentError,
};

/// Check if SIMD is available on current platform
pub fn isAvailable() bool {
    return switch (builtin.cpu.arch) {
        .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .sse) or
                   std.Target.x86.featureSetHas(builtin.cpu.features, .avx2),
        .aarch64 => std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon),
        else => false,
    };
}

/// Get optimal vector width for data type
pub fn vectorWidth(comptime T: type) usize {
    return switch (T) {
        f32 => if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) 8 else 4,
        f64 => if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) 4 else 2,
        i32 => if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) 8 else 4,
        else => 1,
    };
}

/// Vector addition for f32 arrays with SIMD optimization
pub fn vectorAddF32(a: []const f32, b: []const f32, result: []f32) SIMDError!void {
    if (a.len != b.len or a.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64) {
        if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
            return vectorAddF32AVX2(a, b, result);
        } else if (std.Target.x86.featureSetHas(builtin.cpu.features, .sse)) {
            return vectorAddF32SSE(a, b, result);
        }
    } else if (builtin.cpu.arch == .aarch64) {
        if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon)) {
            return vectorAddF32NEON(a, b, result);
        }
    }
    
    // Fallback to scalar implementation
    return vectorAddF32Scalar(a, b, result);
}

/// Vector subtraction for f32 arrays
pub fn vectorSubF32(a: []const f32, b: []const f32, result: []f32) SIMDError!void {
    if (a.len != b.len or a.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vectorSubF32AVX2(a, b, result);
    }
    
    // Fallback to scalar
    for (a, b, result) |a_val, b_val, *r| {
        r.* = a_val - b_val;
    }
}

/// Vector multiplication for f32 arrays
pub fn vectorMulF32(a: []const f32, b: []const f32, result: []f32) SIMDError!void {
    if (a.len != b.len or a.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vectorMulF32AVX2(a, b, result);
    }
    
    // Fallback to scalar
    for (a, b, result) |a_val, b_val, *r| {
        r.* = a_val * b_val;
    }
}

/// Vector division for f32 arrays
pub fn vectorDivF32(a: []const f32, b: []const f32, result: []f32) SIMDError!void {
    if (a.len != b.len or a.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    // Fallback to scalar (division is typically not vectorized efficiently)
    for (a, b, result) |a_val, b_val, *r| {
        r.* = a_val / b_val;
    }
}

/// Scalar implementation of vector addition
fn vectorAddF32Scalar(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |a_val, b_val, *r| {
        r.* = a_val + b_val;
    }
}

/// AVX2 implementation of vector addition
fn vectorAddF32AVX2(a: []const f32, b: []const f32, result: []f32) void {
    const vec_size = 8; // AVX2 processes 8 f32s at once
    var i: usize = 0;
    
    // Process 8 elements at a time
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: @Vector(8, f32) = a[i..i+vec_size][0..8].*;
        const vb: @Vector(8, f32) = b[i..i+vec_size][0..8].*;
        const vr = va + vb;
        result[i..i+vec_size][0..8].* = vr;
    }
    
    // Handle remaining elements
    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// SSE implementation of vector addition
fn vectorAddF32SSE(a: []const f32, b: []const f32, result: []f32) void {
    const vec_size = 4; // SSE processes 4 f32s at once
    var i: usize = 0;
    
    // Process 4 elements at a time
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: @Vector(4, f32) = a[i..i+vec_size][0..4].*;
        const vb: @Vector(4, f32) = b[i..i+vec_size][0..4].*;
        const vr = va + vb;
        result[i..i+vec_size][0..4].* = vr;
    }
    
    // Handle remaining elements
    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// NEON implementation of vector addition (ARM)
fn vectorAddF32NEON(a: []const f32, b: []const f32, result: []f32) void {
    const vec_size = 4; // NEON processes 4 f32s at once
    var i: usize = 0;
    
    // Process 4 elements at a time
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: @Vector(4, f32) = a[i..i+vec_size][0..4].*;
        const vb: @Vector(4, f32) = b[i..i+vec_size][0..4].*;
        const vr = va + vb;
        result[i..i+vec_size][0..4].* = vr;
    }
    
    // Handle remaining elements
    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// AVX2 implementation of vector subtraction
fn vectorSubF32AVX2(a: []const f32, b: []const f32, result: []f32) void {
    const vec_size = 8;
    var i: usize = 0;
    
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: @Vector(8, f32) = a[i..i+vec_size][0..8].*;
        const vb: @Vector(8, f32) = b[i..i+vec_size][0..8].*;
        const vr = va - vb;
        result[i..i+vec_size][0..8].* = vr;
    }
    
    while (i < a.len) : (i += 1) {
        result[i] = a[i] - b[i];
    }
}

/// AVX2 implementation of vector multiplication
fn vectorMulF32AVX2(a: []const f32, b: []const f32, result: []f32) void {
    const vec_size = 8;
    var i: usize = 0;
    
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: @Vector(8, f32) = a[i..i+vec_size][0..8].*;
        const vb: @Vector(8, f32) = b[i..i+vec_size][0..8].*;
        const vr = va * vb;
        result[i..i+vec_size][0..8].* = vr;
    }
    
    while (i < a.len) : (i += 1) {
        result[i] = a[i] * b[i];
    }
}

/// Dot product of two vectors
pub fn vectorDotF32(a: []const f32, b: []const f32) SIMDError!f32 {
    if (a.len != b.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vectorDotF32AVX2(a, b);
    }
    
    // Fallback to scalar
    var sum: f32 = 0.0;
    for (a, b) |a_val, b_val| {
        sum += a_val * b_val;
    }
    return sum;
}

/// AVX2 implementation of dot product
fn vectorDotF32AVX2(a: []const f32, b: []const f32) f32 {
    const vec_size = 8;
    var i: usize = 0;
    var sum_vec: @Vector(8, f32) = @splat(0.0);
    
    // Process 8 elements at a time
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: @Vector(8, f32) = a[i..i+vec_size][0..8].*;
        const vb: @Vector(8, f32) = b[i..i+vec_size][0..8].*;
        sum_vec += va * vb;
    }
    
    // Horizontal sum of vector
    var sum: f32 = 0.0;
    for (0..8) |j| {
        sum += sum_vec[j];
    }
    
    // Handle remaining elements
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

/// Scalar multiplication (broadcast)
pub fn vectorScalarMulF32(scalar: f32, vector: []const f32, result: []f32) SIMDError!void {
    if (vector.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vectorScalarMulF32AVX2(scalar, vector, result);
    }
    
    // Fallback to scalar
    for (vector, result) |v, *r| {
        r.* = scalar * v;
    }
}

/// AVX2 implementation of scalar multiplication
fn vectorScalarMulF32AVX2(scalar: f32, vector: []const f32, result: []f32) void {
    const vec_size = 8;
    var i: usize = 0;
    const scalar_vec: @Vector(8, f32) = @splat(scalar);
    
    while (i + vec_size <= vector.len) : (i += vec_size) {
        const v: @Vector(8, f32) = vector[i..i+vec_size][0..8].*;
        const vr = scalar_vec * v;
        result[i..i+vec_size][0..8].* = vr;
    }
    
    while (i < vector.len) : (i += 1) {
        result[i] = scalar * vector[i];
    }
}

/// Sum all elements in a vector
pub fn vectorSumF32(vector: []const f32) f32 {
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vectorSumF32AVX2(vector);
    }
    
    // Fallback to scalar
    var sum: f32 = 0.0;
    for (vector) |v| {
        sum += v;
    }
    return sum;
}

/// AVX2 implementation of vector sum
fn vectorSumF32AVX2(vector: []const f32) f32 {
    const vec_size = 8;
    var i: usize = 0;
    var sum_vec: @Vector(8, f32) = @splat(0.0);
    
    while (i + vec_size <= vector.len) : (i += vec_size) {
        const v: @Vector(8, f32) = vector[i..i+vec_size][0..8].*;
        sum_vec += v;
    }
    
    // Horizontal sum
    var sum: f32 = 0.0;
    for (0..8) |j| {
        sum += sum_vec[j];
    }
    
    // Handle remaining elements
    while (i < vector.len) : (i += 1) {
        sum += vector[i];
    }
    
    return sum;
}

// Tests
test "SIMD vector operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const size = 100;
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, size);
    defer allocator.free(result);

    // Initialize test data
    for (0..size) |i| {
        a[i] = @as(f32, @floatFromInt(i));
        b[i] = @as(f32, @floatFromInt(i + 1));
    }

    // Test vector addition
    try vectorAddF32(a, b, result);
    for (0..size) |i| {
        const expected = @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(i + 1));
        try testing.expect(result[i] == expected);
    }

    // Test vector multiplication
    try vectorMulF32(a, b, result);
    for (0..size) |i| {
        const expected = @as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(i + 1));
        try testing.expect(result[i] == expected);
    }

    // Test dot product
    const dot = try vectorDotF32(a, b);
    var expected_dot: f32 = 0.0;
    for (0..size) |i| {
        expected_dot += @as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(i + 1));
    }
    try testing.expect(@abs(dot - expected_dot) < 0.001);
}

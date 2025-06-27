const std = @import("std");
const builtin = @import("builtin");

pub const SIMDError = error{
    InvalidLength,
    UnsupportedOperation,
};

/// Vector addition for f32 arrays with SIMD optimization
pub fn vector_add_f32(a: []const f32, b: []const f32, result: []f32) SIMDError!void {
    if (a.len != b.len or a.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64) {
        if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
            return vector_add_f32_avx2(a, b, result);
        } else if (std.Target.x86.featureSetHas(builtin.cpu.features, .sse)) {
            return vector_add_f32_sse(a, b, result);
        }
    } else if (builtin.cpu.arch == .aarch64) {
        if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon)) {
            return vector_add_f32_neon(a, b, result);
        }
    }
    
    // Fallback to scalar implementation
    return vector_add_f32_scalar(a, b, result);
}

/// Scalar implementation of vector addition
fn vector_add_f32_scalar(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |a_val, b_val, *r| {
        r.* = a_val + b_val;
    }
}

/// AVX2 implementation of vector addition
fn vector_add_f32_avx2(a: []const f32, b: []const f32, result: []f32) void {
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
fn vector_add_f32_sse(a: []const f32, b: []const f32, result: []f32) void {
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
fn vector_add_f32_neon(a: []const f32, b: []const f32, result: []f32) void {
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

/// Vector subtraction for f32 arrays
pub fn vector_sub_f32(a: []const f32, b: []const f32, result: []f32) SIMDError!void {
    if (a.len != b.len or a.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vector_sub_f32_avx2(a, b, result);
    }
    
    // Fallback to scalar
    for (a, b, result) |a_val, b_val, *r| {
        r.* = a_val - b_val;
    }
}

fn vector_sub_f32_avx2(a: []const f32, b: []const f32, result: []f32) void {
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

/// Vector multiplication for f32 arrays
pub fn vector_mul_f32(a: []const f32, b: []const f32, result: []f32) SIMDError!void {
    if (a.len != b.len or a.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vector_mul_f32_avx2(a, b, result);
    }
    
    // Fallback to scalar
    for (a, b, result) |a_val, b_val, *r| {
        r.* = a_val * b_val;
    }
}

fn vector_mul_f32_avx2(a: []const f32, b: []const f32, result: []f32) void {
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

/// Scalar multiplication (broadcast)
pub fn vector_scale_f32(a: []const f32, scalar: f32, result: []f32) SIMDError!void {
    if (a.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vector_scale_f32_avx2(a, scalar, result);
    }
    
    // Fallback to scalar
    for (a, result) |a_val, *r| {
        r.* = a_val * scalar;
    }
}

fn vector_scale_f32_avx2(a: []const f32, scalar: f32, result: []f32) void {
    const vec_size = 8;
    const scalar_vec: @Vector(8, f32) = @splat(scalar);
    var i: usize = 0;
    
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: @Vector(8, f32) = a[i..i+vec_size][0..8].*;
        const vr = va * scalar_vec;
        result[i..i+vec_size][0..8].* = vr;
    }
    
    while (i < a.len) : (i += 1) {
        result[i] = a[i] * scalar;
    }
}

/// Dot product of two vectors
pub fn vector_dot_f32(a: []const f32, b: []const f32) SIMDError!f32 {
    if (a.len != b.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vector_dot_f32_avx2(a, b);
    }
    
    // Fallback to scalar
    var sum: f32 = 0.0;
    for (a, b) |a_val, b_val| {
        sum += a_val * b_val;
    }
    return sum;
}

fn vector_dot_f32_avx2(a: []const f32, b: []const f32) f32 {
    const vec_size = 8;
    var sum_vec: @Vector(8, f32) = @splat(0.0);
    var i: usize = 0;
    
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

/// Apply ReLU activation function
pub fn vector_relu_f32(a: []const f32, result: []f32) SIMDError!void {
    if (a.len != result.len) {
        return SIMDError.InvalidLength;
    }
    
    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return vector_relu_f32_avx2(a, result);
    }
    
    // Fallback to scalar
    for (a, result) |a_val, *r| {
        r.* = @max(0.0, a_val);
    }
}

fn vector_relu_f32_avx2(a: []const f32, result: []f32) void {
    const vec_size = 8;
    const zero_vec: @Vector(8, f32) = @splat(0.0);
    var i: usize = 0;
    
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: @Vector(8, f32) = a[i..i+vec_size][0..8].*;
        const vr = @max(zero_vec, va);
        result[i..i+vec_size][0..8].* = vr;
    }
    
    while (i < a.len) : (i += 1) {
        result[i] = @max(0.0, a[i]);
    }
}

test "SIMD vector operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const len = 16;
    const a = try allocator.alloc(f32, len);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, len);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, len);
    defer allocator.free(result);
    
    // Initialize test data
    for (0..len) |i| {
        a[i] = @as(f32, @floatFromInt(i));
        b[i] = @as(f32, @floatFromInt(i + 1));
    }
    
    // Test vector addition
    try vector_add_f32(a, b, result);
    for (0..len) |i| {
        const expected = @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(i + 1));
        try testing.expectApproxEqAbs(result[i], expected, 1e-6);
    }
    
    // Test vector dot product
    const dot = try vector_dot_f32(a, b);
    var expected_dot: f32 = 0.0;
    for (0..len) |i| {
        expected_dot += @as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(i + 1));
    }
    try testing.expectApproxEqAbs(dot, expected_dot, 1e-6);
    
    // Test ReLU
    a[0] = -1.0;
    a[1] = 2.0;
    try vector_relu_f32(a[0..2], result[0..2]);
    try testing.expectApproxEqAbs(result[0], 0.0, 1e-6);
    try testing.expectApproxEqAbs(result[1], 2.0, 1e-6);
}

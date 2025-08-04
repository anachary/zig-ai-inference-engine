const std = @import("std");
const builtin = @import("builtin");

/// Check if SIMD operations are available on this platform
pub fn isAvailable() bool {
    return switch (builtin.cpu.arch) {
        .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .avx2),
        .aarch64 => std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon),
        else => false,
    };
}

/// SIMD-optimized matrix multiplication
pub fn matmul(a: anytype, b: anytype, c: anytype) void {
    if (comptime !isAvailable()) {
        @compileError("SIMD not available on this platform");
    }

    switch (builtin.cpu.arch) {
        .x86_64 => matmulAVX2(a, b, c),
        .aarch64 => matmulNEON(a, b, c),
        else => unreachable,
    }
}

/// AVX2-optimized matrix multiplication for x86_64
fn matmulAVX2(a: anytype, b: anytype, c: anytype) void {
    if (!comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        @compileError("AVX2 not available");
    }

    // Initialize result to zero
    c.fill(0.0);

    const Vector = @Vector(8, f32);
    
    // Process in blocks of 8 for AVX2
    var i: usize = 0;
    while (i < a.rows) : (i += 1) {
        var j: usize = 0;
        while (j + 8 <= b.cols) : (j += 8) {
            var sum_vec: Vector = @splat(0.0);
            
            for (0..a.cols) |k| {
                const a_val: Vector = @splat(a.get(i, k));
                const b_row = b.getRow(k);
                const b_vec: Vector = b_row[j..j+8][0..8].*;
                sum_vec += a_val * b_vec;
            }
            
            // Store result
            const result_row = c.getRow(i);
            result_row[j..j+8][0..8].* = sum_vec;
        }
        
        // Handle remaining columns
        while (j < b.cols) : (j += 1) {
            var sum: f32 = 0.0;
            for (0..a.cols) |k| {
                sum += a.get(i, k) * b.get(k, j);
            }
            c.set(i, j, sum);
        }
    }
}

/// NEON-optimized matrix multiplication for ARM64
fn matmulNEON(a: anytype, b: anytype, c: anytype) void {
    if (!comptime std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon)) {
        @compileError("NEON not available");
    }

    // Initialize result to zero
    c.fill(0.0);

    const Vector = @Vector(4, f32);
    
    // Process in blocks of 4 for NEON
    var i: usize = 0;
    while (i < a.rows) : (i += 1) {
        var j: usize = 0;
        while (j + 4 <= b.cols) : (j += 4) {
            var sum_vec: Vector = @splat(0.0);
            
            for (0..a.cols) |k| {
                const a_val: Vector = @splat(a.get(i, k));
                const b_row = b.getRow(k);
                const b_vec: Vector = b_row[j..j+4][0..4].*;
                sum_vec += a_val * b_vec;
            }
            
            // Store result
            const result_row = c.getRow(i);
            result_row[j..j+4][0..4].* = sum_vec;
        }
        
        // Handle remaining columns
        while (j < b.cols) : (j += 1) {
            var sum: f32 = 0.0;
            for (0..a.cols) |k| {
                sum += a.get(i, k) * b.get(k, j);
            }
            c.set(i, j, sum);
        }
    }
}

/// SIMD-optimized vector operations
pub const vector = struct {
    /// Dot product of two vectors
    pub fn dot(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        
        if (!isAvailable()) {
            return dotScalar(a, b);
        }
        
        return switch (builtin.cpu.arch) {
            .x86_64 => dotAVX2(a, b),
            .aarch64 => dotNEON(a, b),
            else => dotScalar(a, b),
        };
    }
    
    fn dotScalar(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0.0;
        for (a, b) |a_val, b_val| {
            sum += a_val * b_val;
        }
        return sum;
    }
    
    fn dotAVX2(a: []const f32, b: []const f32) f32 {
        const Vector = @Vector(8, f32);
        var sum_vec: Vector = @splat(0.0);
        
        var i: usize = 0;
        while (i + 8 <= a.len) : (i += 8) {
            const a_vec: Vector = a[i..i+8][0..8].*;
            const b_vec: Vector = b[i..i+8][0..8].*;
            sum_vec += a_vec * b_vec;
        }
        
        // Horizontal sum
        var sum: f32 = @reduce(.Add, sum_vec);
        
        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }
    
    fn dotNEON(a: []const f32, b: []const f32) f32 {
        const Vector = @Vector(4, f32);
        var sum_vec: Vector = @splat(0.0);
        
        var i: usize = 0;
        while (i + 4 <= a.len) : (i += 4) {
            const a_vec: Vector = a[i..i+4][0..4].*;
            const b_vec: Vector = b[i..i+4][0..4].*;
            sum_vec += a_vec * b_vec;
        }
        
        // Horizontal sum
        var sum: f32 = @reduce(.Add, sum_vec);
        
        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }
    
    /// Element-wise addition: c = a + b
    pub fn add(a: []const f32, b: []const f32, c: []f32) void {
        std.debug.assert(a.len == b.len and b.len == c.len);
        
        if (!isAvailable()) {
            for (a, b, c) |a_val, b_val, *c_val| {
                c_val.* = a_val + b_val;
            }
            return;
        }
        
        switch (builtin.cpu.arch) {
            .x86_64 => addAVX2(a, b, c),
            .aarch64 => addNEON(a, b, c),
            else => {
                for (a, b, c) |a_val, b_val, *c_val| {
                    c_val.* = a_val + b_val;
                }
            },
        }
    }
    
    fn addAVX2(a: []const f32, b: []const f32, c: []f32) void {
        const Vector = @Vector(8, f32);
        
        var i: usize = 0;
        while (i + 8 <= a.len) : (i += 8) {
            const a_vec: Vector = a[i..i+8][0..8].*;
            const b_vec: Vector = b[i..i+8][0..8].*;
            c[i..i+8][0..8].* = a_vec + b_vec;
        }
        
        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            c[i] = a[i] + b[i];
        }
    }
    
    fn addNEON(a: []const f32, b: []const f32, c: []f32) void {
        const Vector = @Vector(4, f32);
        
        var i: usize = 0;
        while (i + 4 <= a.len) : (i += 4) {
            const a_vec: Vector = a[i..i+4][0..4].*;
            const b_vec: Vector = b[i..i+4][0..4].*;
            c[i..i+4][0..4].* = a_vec + b_vec;
        }
        
        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            c[i] = a[i] + b[i];
        }
    }
};

test "simd availability" {
    // This test just checks that the function compiles and runs
    _ = isAvailable();
}

test "vector dot product" {
    const testing = std.testing;
    
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    
    const result = vector.dot(&a, &b);
    const expected: f32 = 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0; // = 70.0
    
    try testing.expect(result == expected);
}

test "vector addition" {
    const testing = std.testing;
    
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var c = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    
    vector.add(&a, &b, &c);
    
    try testing.expect(c[0] == 6.0);
    try testing.expect(c[1] == 8.0);
    try testing.expect(c[2] == 10.0);
    try testing.expect(c[3] == 12.0);
}

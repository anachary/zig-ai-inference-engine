const std = @import("std");
const simd = @import("simd.zig");

/// Matrix data structure for neural network operations
pub const Matrix = struct {
    data: []f32,
    rows: usize,
    cols: usize,
    stride: usize, // For sub-matrix views
    allocator: ?std.mem.Allocator, // null if this is a view

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const data = try allocator.alloc(f32, rows * cols);
        return Matrix{
            .data = data,
            .rows = rows,
            .cols = cols,
            .stride = cols,
            .allocator = allocator,
        };
    }

    pub fn initZeros(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        var matrix = try init(allocator, rows, cols);
        @memset(matrix.data, 0.0);
        return matrix;
    }

    pub fn initOnes(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        var matrix = try init(allocator, rows, cols);
        for (matrix.data) |*val| {
            val.* = 1.0;
        }
        return matrix;
    }

    pub fn initRandom(allocator: std.mem.Allocator, rows: usize, cols: usize, rng: std.rand.Random) !Matrix {
        var matrix = try init(allocator, rows, cols);
        for (matrix.data) |*val| {
            val.* = rng.float(f32) * 2.0 - 1.0; // Random between -1 and 1
        }
        return matrix;
    }

    pub fn deinit(self: *Matrix) void {
        if (self.allocator) |allocator| {
            allocator.free(self.data);
        }
    }

    pub fn get(self: Matrix, row: usize, col: usize) f32 {
        std.debug.assert(row < self.rows and col < self.cols);
        return self.data[row * self.stride + col];
    }

    pub fn set(self: *Matrix, row: usize, col: usize, value: f32) void {
        std.debug.assert(row < self.rows and col < self.cols);
        self.data[row * self.stride + col] = value;
    }

    pub fn getRow(self: Matrix, row: usize) []f32 {
        std.debug.assert(row < self.rows);
        const start = row * self.stride;
        return self.data[start .. start + self.cols];
    }

    pub fn fill(self: *Matrix, value: f32) void {
        for (self.data) |*val| {
            val.* = value;
        }
    }

    pub fn copy(self: Matrix, allocator: std.mem.Allocator) !Matrix {
        var result = try init(allocator, self.rows, self.cols);
        @memcpy(result.data, self.data);
        return result;
    }

    /// Create a view into a submatrix (no allocation)
    pub fn view(self: Matrix, start_row: usize, start_col: usize, rows: usize, cols: usize) MatrixView {
        std.debug.assert(start_row + rows <= self.rows);
        std.debug.assert(start_col + cols <= self.cols);
        
        const start_idx = start_row * self.stride + start_col;
        return MatrixView{
            .data = self.data[start_idx..],
            .rows = rows,
            .cols = cols,
            .stride = self.stride,
            .allocator = null,
        };
    }
};

/// Matrix view for zero-copy submatrix operations
pub const MatrixView = Matrix;

/// Matrix multiplication: C = A * B
pub fn matmul(a: Matrix, b: Matrix, c: *Matrix) !void {
    if (a.cols != b.rows or c.rows != a.rows or c.cols != b.cols) {
        return error.IncompatibleDimensions;
    }

    // Use SIMD-optimized implementation when available
    if (simd.isAvailable()) {
        return simd.matmul(a, b, c);
    }

    // Fallback to standard implementation
    return matmulStandard(a, b, c);
}

/// Standard matrix multiplication implementation
fn matmulStandard(a: Matrix, b: Matrix, c: *Matrix) void {
    // Initialize result to zero
    c.fill(0.0);

    // Compute C = A * B
    for (0..a.rows) |i| {
        for (0..b.cols) |j| {
            var sum: f32 = 0.0;
            for (0..a.cols) |k| {
                sum += a.get(i, k) * b.get(k, j);
            }
            c.set(i, j, sum);
        }
    }
}

/// Matrix-vector multiplication: y = A * x
pub fn matvec(a: Matrix, x: []const f32, y: []f32) !void {
    if (a.cols != x.len or a.rows != y.len) {
        return error.IncompatibleDimensions;
    }

    for (0..a.rows) |i| {
        var sum: f32 = 0.0;
        for (0..a.cols) |j| {
            sum += a.get(i, j) * x[j];
        }
        y[i] = sum;
    }
}

/// Element-wise addition: C = A + B
pub fn add(a: Matrix, b: Matrix, c: *Matrix) !void {
    if (a.rows != b.rows or a.cols != b.cols or 
        c.rows != a.rows or c.cols != a.cols) {
        return error.IncompatibleDimensions;
    }

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            c.set(i, j, a.get(i, j) + b.get(i, j));
        }
    }
}

/// Element-wise addition with broadcasting: C = A + b (scalar)
pub fn addScalar(a: Matrix, scalar: f32, c: *Matrix) !void {
    if (c.rows != a.rows or c.cols != a.cols) {
        return error.IncompatibleDimensions;
    }

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            c.set(i, j, a.get(i, j) + scalar);
        }
    }
}

/// Element-wise multiplication: C = A ⊙ B
pub fn hadamard(a: Matrix, b: Matrix, c: *Matrix) !void {
    if (a.rows != b.rows or a.cols != b.cols or 
        c.rows != a.rows or c.cols != a.cols) {
        return error.IncompatibleDimensions;
    }

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            c.set(i, j, a.get(i, j) * b.get(i, j));
        }
    }
}

/// Transpose matrix: B = A^T
pub fn transpose(a: Matrix, b: *Matrix) !void {
    if (b.rows != a.cols or b.cols != a.rows) {
        return error.IncompatibleDimensions;
    }

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            b.set(j, i, a.get(i, j));
        }
    }
}

/// Scale matrix by scalar: B = α * A
pub fn scale(a: Matrix, scalar: f32, b: *Matrix) !void {
    if (b.rows != a.rows or b.cols != a.cols) {
        return error.IncompatibleDimensions;
    }

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            b.set(i, j, a.get(i, j) * scalar);
        }
    }
}

test "matrix creation and basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var a = try Matrix.initZeros(allocator, 2, 3);
    defer a.deinit();

    a.set(0, 0, 1.0);
    a.set(1, 2, 2.0);

    try testing.expect(a.get(0, 0) == 1.0);
    try testing.expect(a.get(1, 2) == 2.0);
    try testing.expect(a.get(0, 1) == 0.0);
}

test "matrix multiplication" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var a = try Matrix.init(allocator, 2, 3);
    defer a.deinit();
    var b = try Matrix.init(allocator, 3, 2);
    defer b.deinit();
    var c = try Matrix.initZeros(allocator, 2, 2);
    defer c.deinit();

    // Set up test matrices
    a.set(0, 0, 1.0); a.set(0, 1, 2.0); a.set(0, 2, 3.0);
    a.set(1, 0, 4.0); a.set(1, 1, 5.0); a.set(1, 2, 6.0);

    b.set(0, 0, 7.0); b.set(0, 1, 8.0);
    b.set(1, 0, 9.0); b.set(1, 1, 10.0);
    b.set(2, 0, 11.0); b.set(2, 1, 12.0);

    try matmul(a, b, &c);

    // Expected result: [[58, 64], [139, 154]]
    try testing.expect(c.get(0, 0) == 58.0);
    try testing.expect(c.get(0, 1) == 64.0);
    try testing.expect(c.get(1, 0) == 139.0);
    try testing.expect(c.get(1, 1) == 154.0);
}

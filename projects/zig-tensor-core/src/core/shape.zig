const std = @import("std");
const Allocator = std.mem.Allocator;

/// Shape operation errors
pub const ShapeError = error{
    InvalidShape,
    IncompatibleShapes,
    OutOfMemory,
};

/// Compute row-major strides for a given shape
pub fn computeStrides(shape: []const usize, allocator: Allocator) ![]usize {
    // 0D scalar case: empty strides array
    if (shape.len == 0) {
        return try allocator.alloc(usize, 0);
    }

    const strides = try allocator.alloc(usize, shape.len);
    var stride: usize = 1;
    var i = shape.len;

    while (i > 0) {
        i -= 1;
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

/// Compute column-major strides for a given shape
pub fn computeStridesColumnMajor(shape: []const usize, allocator: Allocator) ![]usize {
    // 0D scalar case: empty strides array
    if (shape.len == 0) {
        return try allocator.alloc(usize, 0);
    }

    const strides = try allocator.alloc(usize, shape.len);
    var stride: usize = 1;

    for (shape, 0..) |dim, i| {
        strides[i] = stride;
        stride *= dim;
    }

    return strides;
}

/// Validate that a shape is valid (no zero dimensions, but 0D scalars are allowed)
pub fn validateShape(shape: []const usize) bool {
    // 0D scalar is valid (empty shape)
    if (shape.len == 0) return true;

    for (shape) |dim| {
        if (dim == 0) return false;
    }

    return true;
}

/// Calculate total number of elements in a shape
pub fn shapeNumel(shape: []const usize) usize {
    // 0D scalar has 1 element
    if (shape.len == 0) return 1;

    var total: usize = 1;
    for (shape) |dim| {
        total *= dim;
    }
    return total;
}

/// Check if two shapes are equal
pub fn shapesEqual(shape1: []const usize, shape2: []const usize) bool {
    if (shape1.len != shape2.len) return false;

    for (shape1, shape2) |dim1, dim2| {
        if (dim1 != dim2) return false;
    }

    return true;
}

/// Broadcast two shapes according to NumPy broadcasting rules
pub fn broadcastShapes(shape1: []const usize, shape2: []const usize, allocator: Allocator) ![]usize {
    const max_ndim = @max(shape1.len, shape2.len);
    const result_shape = try allocator.alloc(usize, max_ndim);

    var i: usize = 0;
    while (i < max_ndim) : (i += 1) {
        const idx = max_ndim - 1 - i;

        const dim1 = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
        const dim2 = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;

        if (dim1 == dim2) {
            result_shape[idx] = dim1;
        } else if (dim1 == 1) {
            result_shape[idx] = dim2;
        } else if (dim2 == 1) {
            result_shape[idx] = dim1;
        } else {
            allocator.free(result_shape);
            return ShapeError.IncompatibleShapes;
        }
    }

    return result_shape;
}

/// Check if two shapes are broadcastable
pub fn canBroadcast(shape1: []const usize, shape2: []const usize) bool {
    const max_ndim = @max(shape1.len, shape2.len);
    
    var i: usize = 0;
    while (i < max_ndim) : (i += 1) {
        const dim1 = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
        const dim2 = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;

        if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
            return false;
        }
    }

    return true;
}

/// Compute the flat index from multi-dimensional indices
pub fn computeFlatIndex(indices: []const usize, strides: []const usize) usize {
    var flat_index: usize = 0;
    for (indices, strides) |idx, stride| {
        flat_index += idx * stride;
    }
    return flat_index;
}

/// Convert flat index to multi-dimensional indices
pub fn computeMultiIndex(flat_index: usize, shape: []const usize, allocator: Allocator) ![]usize {
    const indices = try allocator.alloc(usize, shape.len);
    var remaining = flat_index;

    var i = shape.len;
    while (i > 0) {
        i -= 1;
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }

    return indices;
}

/// Transpose shape (reverse dimensions)
pub fn transposeShape(shape: []const usize, allocator: Allocator) ![]usize {
    const transposed = try allocator.alloc(usize, shape.len);
    
    for (shape, 0..) |dim, i| {
        transposed[shape.len - 1 - i] = dim;
    }

    return transposed;
}

/// Squeeze shape (remove dimensions of size 1)
pub fn squeezeShape(shape: []const usize, allocator: Allocator) ![]usize {
    var new_dims = std.ArrayList(usize).init(allocator);
    defer new_dims.deinit();

    for (shape) |dim| {
        if (dim != 1) {
            try new_dims.append(dim);
        }
    }

    return new_dims.toOwnedSlice();
}

/// Unsqueeze shape (add dimension of size 1 at specified axis)
pub fn unsqueezeShape(shape: []const usize, axis: usize, allocator: Allocator) ![]usize {
    if (axis > shape.len) return ShapeError.InvalidShape;

    const new_shape = try allocator.alloc(usize, shape.len + 1);
    
    // Copy dimensions before axis
    for (0..axis) |i| {
        new_shape[i] = shape[i];
    }
    
    // Insert new dimension
    new_shape[axis] = 1;
    
    // Copy dimensions after axis
    for (axis..shape.len) |i| {
        new_shape[i + 1] = shape[i];
    }

    return new_shape;
}

/// Reshape utilities
pub const reshape = struct {
    /// Check if reshape is valid (same total elements)
    pub fn isValidReshape(old_shape: []const usize, new_shape: []const usize) bool {
        return shapeNumel(old_shape) == shapeNumel(new_shape);
    }

    /// Infer dimension size for reshape (-1 means infer)
    pub fn inferDimension(old_shape: []const usize, new_shape: []const i64) ![]usize {
        const old_numel = shapeNumel(old_shape);
        var infer_idx: ?usize = null;
        var new_numel: usize = 1;

        // Find -1 dimension and calculate known dimensions
        for (new_shape, 0..) |dim, i| {
            if (dim == -1) {
                if (infer_idx != null) return ShapeError.InvalidShape; // Multiple -1 not allowed
                infer_idx = i;
            } else if (dim <= 0) {
                return ShapeError.InvalidShape;
            } else {
                new_numel *= @as(usize, @intCast(dim));
            }
        }

        // Create result shape
        var result = std.ArrayList(usize).init(std.heap.page_allocator);
        defer result.deinit();

        for (new_shape, 0..) |dim, i| {
            if (dim == -1) {
                if (new_numel == 0) return ShapeError.InvalidShape;
                const inferred_dim = old_numel / new_numel;
                if (inferred_dim * new_numel != old_numel) return ShapeError.InvalidShape;
                try result.append(inferred_dim);
            } else {
                try result.append(@as(usize, @intCast(dim)));
            }
        }

        return result.toOwnedSlice();
    }
};

// Tests
test "shape utilities" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test stride computation
    const shape = [_]usize{ 2, 3, 4 };
    const strides = try computeStrides(&shape, allocator);
    defer allocator.free(strides);

    try testing.expect(strides[0] == 12); // 3 * 4
    try testing.expect(strides[1] == 4); // 4
    try testing.expect(strides[2] == 1); // 1

    // Test shape validation
    try testing.expect(validateShape(&shape));
    try testing.expect(!validateShape(&[_]usize{ 2, 0, 4 }));
    try testing.expect(validateShape(&[_]usize{})); // 0D scalar is valid

    // Test broadcasting
    const shape1 = [_]usize{ 3, 1, 4 };
    const shape2 = [_]usize{ 1, 2, 4 };
    const broadcast_result = try broadcastShapes(&shape1, &shape2, allocator);
    defer allocator.free(broadcast_result);

    try testing.expect(broadcast_result.len == 3);
    try testing.expect(broadcast_result[0] == 3);
    try testing.expect(broadcast_result[1] == 2);
    try testing.expect(broadcast_result[2] == 4);
}

test "shape transformations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const shape = [_]usize{ 2, 1, 3, 1 };

    // Test transpose
    const transposed = try transposeShape(&shape, allocator);
    defer allocator.free(transposed);
    try testing.expect(transposed[0] == 1);
    try testing.expect(transposed[1] == 3);
    try testing.expect(transposed[2] == 1);
    try testing.expect(transposed[3] == 2);

    // Test squeeze
    const squeezed = try squeezeShape(&shape, allocator);
    defer allocator.free(squeezed);
    try testing.expect(squeezed.len == 2);
    try testing.expect(squeezed[0] == 2);
    try testing.expect(squeezed[1] == 3);

    // Test unsqueeze
    const unsqueezed = try unsqueezeShape(&[_]usize{ 2, 3 }, 1, allocator);
    defer allocator.free(unsqueezed);
    try testing.expect(unsqueezed.len == 3);
    try testing.expect(unsqueezed[0] == 2);
    try testing.expect(unsqueezed[1] == 1);
    try testing.expect(unsqueezed[2] == 3);
}

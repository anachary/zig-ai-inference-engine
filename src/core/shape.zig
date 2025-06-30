const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ShapeError = error{
    InvalidShape,
    IncompatibleShapes,
    OutOfMemory,
};

/// Compute row-major strides for a given shape
pub fn compute_strides(shape: []const usize, allocator: Allocator) ![]usize {
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
pub fn compute_strides_column_major(shape: []const usize, allocator: Allocator) ![]usize {
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
pub fn validate_shape(shape: []const usize) bool {
    // 0D scalar is valid (empty shape)
    if (shape.len == 0) return true;

    for (shape) |dim| {
        if (dim == 0) return false;
    }

    return true;
}

/// Calculate total number of elements in a shape
pub fn shape_numel(shape: []const usize) usize {
    // 0D scalar has 1 element
    if (shape.len == 0) return 1;

    var total: usize = 1;
    for (shape) |dim| {
        total *= dim;
    }
    return total;
}

/// Check if two shapes are equal
pub fn shapes_equal(shape1: []const usize, shape2: []const usize) bool {
    if (shape1.len != shape2.len) return false;

    for (shape1, shape2) |dim1, dim2| {
        if (dim1 != dim2) return false;
    }

    return true;
}

/// Broadcast two shapes according to NumPy broadcasting rules
pub fn broadcast_shapes(shape1: []const usize, shape2: []const usize, allocator: Allocator) ![]usize {
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

/// Check if shape1 can be broadcast to shape2
pub fn can_broadcast_to(shape1: []const usize, shape2: []const usize) bool {
    if (shape1.len > shape2.len) return false;

    var i: usize = 0;
    while (i < shape1.len) : (i += 1) {
        const dim1 = shape1[shape1.len - 1 - i];
        const dim2 = shape2[shape2.len - 1 - i];

        if (dim1 != dim2 and dim1 != 1) {
            return false;
        }
    }

    return true;
}

/// Reshape a tensor while preserving total elements
pub fn can_reshape(old_shape: []const usize, new_shape: []const usize) bool {
    return shape_numel(old_shape) == shape_numel(new_shape);
}

/// Calculate the linear index from multi-dimensional indices
pub fn ravel_index(indices: []const usize, shape: []const usize, strides: []const usize) !usize {
    if (indices.len != shape.len or indices.len != strides.len) {
        return ShapeError.InvalidShape;
    }

    var linear_index: usize = 0;
    for (indices, shape, strides) |idx, dim, stride| {
        if (idx >= dim) return ShapeError.InvalidShape;
        linear_index += idx * stride;
    }

    return linear_index;
}

/// Calculate multi-dimensional indices from linear index
pub fn unravel_index(linear_index: usize, shape: []const usize, allocator: Allocator) ![]usize {
    const indices = try allocator.alloc(usize, shape.len);
    var remaining = linear_index;

    var i = shape.len;
    while (i > 0) {
        i -= 1;
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }

    return indices;
}

/// Transpose shape (reverse dimensions)
pub fn transpose_shape(shape: []const usize, allocator: Allocator) ![]usize {
    const transposed = try allocator.alloc(usize, shape.len);

    for (shape, 0..) |dim, i| {
        transposed[shape.len - 1 - i] = dim;
    }

    return transposed;
}

/// Squeeze shape (remove dimensions of size 1)
pub fn squeeze_shape(shape: []const usize, allocator: Allocator) ![]usize {
    var new_len: usize = 0;
    for (shape) |dim| {
        if (dim != 1) new_len += 1;
    }

    if (new_len == 0) new_len = 1; // Keep at least one dimension

    const squeezed = try allocator.alloc(usize, new_len);
    var j: usize = 0;

    for (shape) |dim| {
        if (dim != 1) {
            squeezed[j] = dim;
            j += 1;
        }
    }

    // If all dimensions were 1, keep one
    if (j == 0) {
        squeezed[0] = 1;
    }

    return squeezed;
}

test "shape utilities" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test stride computation
    const shape = [_]usize{ 2, 3, 4 };
    const strides = try compute_strides(&shape, allocator);
    defer allocator.free(strides);

    try testing.expect(strides[0] == 12); // 3 * 4
    try testing.expect(strides[1] == 4); // 4
    try testing.expect(strides[2] == 1); // 1

    // Test shape validation
    try testing.expect(validate_shape(&shape));
    try testing.expect(!validate_shape(&[_]usize{ 2, 0, 4 }));
    try testing.expect(validate_shape(&[_]usize{})); // 0D scalar is now valid!

    // Test broadcasting
    const shape1 = [_]usize{ 3, 1 };
    const shape2 = [_]usize{ 1, 4 };
    const broadcast_result = try broadcast_shapes(&shape1, &shape2, allocator);
    defer allocator.free(broadcast_result);

    try testing.expect(broadcast_result.len == 2);
    try testing.expect(broadcast_result[0] == 3);
    try testing.expect(broadcast_result[1] == 4);

    // Test reshape validation
    const old_shape = [_]usize{ 2, 6 };
    const new_shape = [_]usize{ 3, 4 };
    try testing.expect(can_reshape(&old_shape, &new_shape));

    const invalid_new_shape = [_]usize{ 3, 5 };
    try testing.expect(!can_reshape(&old_shape, &invalid_new_shape));
}

test "ravel and unravel index" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const shape = [_]usize{ 2, 3 };
    const strides = try compute_strides(&shape, allocator);
    defer allocator.free(strides);

    // Test ravel_index
    const indices = [_]usize{ 1, 2 };
    const linear_idx = try ravel_index(&indices, &shape, strides);
    try testing.expect(linear_idx == 5); // 1 * 3 + 2

    // Test unravel_index
    const unraveled = try unravel_index(linear_idx, &shape, allocator);
    defer allocator.free(unraveled);

    try testing.expect(unraveled[0] == 1);
    try testing.expect(unraveled[1] == 2);
}

test "0D scalar shape utilities" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test 0D scalar shape validation
    const scalar_shape = [_]usize{};
    try testing.expect(validate_shape(&scalar_shape));

    // Test 0D scalar numel
    try testing.expect(shape_numel(&scalar_shape) == 1);

    // Test 0D scalar stride computation
    const strides = try compute_strides(&scalar_shape, allocator);
    defer allocator.free(strides);
    try testing.expect(strides.len == 0);

    // Test 0D scalar reshape validation
    const scalar_shape2 = [_]usize{};
    try testing.expect(can_reshape(&scalar_shape, &scalar_shape2));

    const vector_shape = [_]usize{1};
    try testing.expect(can_reshape(&scalar_shape, &vector_shape));
    try testing.expect(can_reshape(&vector_shape, &scalar_shape));
}

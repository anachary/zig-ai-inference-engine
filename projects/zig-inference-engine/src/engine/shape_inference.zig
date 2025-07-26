const std = @import("std");
const Allocator = std.mem.Allocator;

/// Shape inference utilities for operators
pub const ShapeInference = struct {
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
        };
    }

    /// Infer output shape for element-wise binary operations (Add, Mul, etc.)
    pub fn inferBinaryOpShape(self: *Self, shape1: []const usize, shape2: []const usize) ![]usize {
        return try broadcastShapes(shape1, shape2, self.allocator);
    }

    /// Infer output shape for matrix multiplication
    pub fn inferMatMulShape(self: *Self, shape1: []const usize, shape2: []const usize) ![]usize {
        if (shape1.len < 2 or shape2.len < 2) {
            return error.InvalidShape;
        }

        const m = shape1[shape1.len - 2];
        const k1 = shape1[shape1.len - 1];
        const k2 = shape2[shape2.len - 2];
        const n = shape2[shape2.len - 1];

        if (k1 != k2) {
            return error.IncompatibleShapes;
        }

        // Handle batch dimensions
        const batch_dims = @max(shape1.len, shape2.len) - 2;
        var result_shape = try self.allocator.alloc(usize, batch_dims + 2);

        // Broadcast batch dimensions
        if (batch_dims > 0) {
            const batch1 = shape1[0..shape1.len - 2];
            const batch2 = shape2[0..shape2.len - 2];
            const broadcast_batch = try broadcastShapes(batch1, batch2, self.allocator);
            defer self.allocator.free(broadcast_batch);

            @memcpy(result_shape[0..batch_dims], broadcast_batch);
        }

        // Set matrix dimensions
        result_shape[batch_dims] = m;
        result_shape[batch_dims + 1] = n;

        return result_shape;
    }

    /// Infer output shape for convolution
    pub fn inferConvShape(
        self: *Self,
        input_shape: []const usize,
        kernel_shape: []const usize,
        strides: []const usize,
        pads: []const usize,
        dilations: []const usize,
    ) ![]usize {
        if (input_shape.len != 4 or kernel_shape.len != 4) {
            return error.InvalidShape; // Expect NCHW format
        }

        const batch_size = input_shape[0];
        const out_channels = kernel_shape[0];
        const input_h = input_shape[2];
        const input_w = input_shape[3];
        const kernel_h = kernel_shape[2];
        const kernel_w = kernel_shape[3];

        const stride_h = if (strides.len >= 1) strides[0] else 1;
        const stride_w = if (strides.len >= 2) strides[1] else stride_h;

        const pad_h = if (pads.len >= 1) pads[0] else 0;
        const pad_w = if (pads.len >= 2) pads[1] else pad_h;

        const dilation_h = if (dilations.len >= 1) dilations[0] else 1;
        const dilation_w = if (dilations.len >= 2) dilations[1] else dilation_h;

        const effective_kernel_h = (kernel_h - 1) * dilation_h + 1;
        const effective_kernel_w = (kernel_w - 1) * dilation_w + 1;

        const output_h = (input_h + 2 * pad_h - effective_kernel_h) / stride_h + 1;
        const output_w = (input_w + 2 * pad_w - effective_kernel_w) / stride_w + 1;

        var result_shape = try self.allocator.alloc(usize, 4);
        result_shape[0] = batch_size;
        result_shape[1] = out_channels;
        result_shape[2] = output_h;
        result_shape[3] = output_w;

        return result_shape;
    }

    /// Infer output shape for pooling operations
    pub fn inferPoolShape(
        self: *Self,
        input_shape: []const usize,
        kernel_size: []const usize,
        strides: []const usize,
        pads: []const usize,
    ) ![]usize {
        if (input_shape.len != 4) {
            return error.InvalidShape; // Expect NCHW format
        }

        const batch_size = input_shape[0];
        const channels = input_shape[1];
        const input_h = input_shape[2];
        const input_w = input_shape[3];

        const kernel_h = if (kernel_size.len >= 1) kernel_size[0] else 1;
        const kernel_w = if (kernel_size.len >= 2) kernel_size[1] else kernel_h;

        const stride_h = if (strides.len >= 1) strides[0] else 1;
        const stride_w = if (strides.len >= 2) strides[1] else stride_h;

        const pad_h = if (pads.len >= 1) pads[0] else 0;
        const pad_w = if (pads.len >= 2) pads[1] else pad_h;

        const output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
        const output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

        var result_shape = try self.allocator.alloc(usize, 4);
        result_shape[0] = batch_size;
        result_shape[1] = channels;
        result_shape[2] = output_h;
        result_shape[3] = output_w;

        return result_shape;
    }

    /// Infer output shape for transpose operation
    pub fn inferTransposeShape(self: *Self, input_shape: []const usize, perm: []const usize) ![]usize {
        if (perm.len != input_shape.len) {
            return error.InvalidPermutation;
        }

        var result_shape = try self.allocator.alloc(usize, input_shape.len);
        for (perm, 0..) |p, i| {
            if (p >= input_shape.len) {
                self.allocator.free(result_shape);
                return error.InvalidPermutation;
            }
            result_shape[i] = input_shape[p];
        }

        return result_shape;
    }

    /// Infer output shape for reshape operation
    pub fn inferReshapeShape(self: *Self, input_shape: []const usize, target_shape: []const i64) ![]usize {
        const input_numel = computeNumel(input_shape);
        
        // Count -1 dimensions and compute known size
        var inferred_dim: ?usize = null;
        var known_size: usize = 1;
        
        for (target_shape, 0..) |dim, i| {
            if (dim == -1) {
                if (inferred_dim != null) {
                    return error.MultipleInferredDims;
                }
                inferred_dim = i;
            } else if (dim <= 0) {
                return error.InvalidShape;
            } else {
                known_size *= @intCast(dim);
            }
        }

        var result_shape = try self.allocator.alloc(usize, target_shape.len);
        
        for (target_shape, 0..) |dim, i| {
            if (dim == -1) {
                result_shape[i] = input_numel / known_size;
            } else {
                result_shape[i] = @intCast(dim);
            }
        }

        // Verify total elements match
        if (computeNumel(result_shape) != input_numel) {
            self.allocator.free(result_shape);
            return error.ShapeMismatch;
        }

        return result_shape;
    }

    /// Infer output shape for squeeze operation
    pub fn inferSqueezeShape(self: *Self, input_shape: []const usize, axes: ?[]const i32) ![]usize {
        var result_dims = std.ArrayList(usize).init(self.allocator);
        defer result_dims.deinit();

        if (axes) |squeeze_axes| {
            // Squeeze specific axes
            for (input_shape, 0..) |dim, i| {
                var should_squeeze = false;
                for (squeeze_axes) |axis| {
                    if (axis >= 0 and @as(usize, @intCast(axis)) == i and dim == 1) {
                        should_squeeze = true;
                        break;
                    }
                }
                if (!should_squeeze) {
                    try result_dims.append(dim);
                }
            }
        } else {
            // Squeeze all dimensions of size 1
            for (input_shape) |dim| {
                if (dim != 1) {
                    try result_dims.append(dim);
                }
            }
        }

        return result_dims.toOwnedSlice();
    }

    /// Infer output shape for unsqueeze operation
    pub fn inferUnsqueezeShape(self: *Self, input_shape: []const usize, axes: []const i32) ![]usize {
        const output_ndim = input_shape.len + axes.len;
        var result_shape = try self.allocator.alloc(usize, output_ndim);
        
        // Initialize with input dimensions
        var input_idx: usize = 0;
        for (0..output_ndim) |i| {
            var is_new_axis = false;
            for (axes) |axis| {
                if (axis >= 0 and @as(usize, @intCast(axis)) == i) {
                    is_new_axis = true;
                    break;
                }
            }
            
            if (is_new_axis) {
                result_shape[i] = 1;
            } else {
                result_shape[i] = input_shape[input_idx];
                input_idx += 1;
            }
        }

        return result_shape;
    }
};

/// Broadcast two shapes according to NumPy broadcasting rules
fn broadcastShapes(shape1: []const usize, shape2: []const usize, allocator: Allocator) ![]usize {
    const max_ndim = @max(shape1.len, shape2.len);
    var result_shape = try allocator.alloc(usize, max_ndim);

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
            return error.IncompatibleShapes;
        }
    }

    return result_shape;
}

/// Compute total number of elements in a shape
fn computeNumel(shape: []const usize) usize {
    var total: usize = 1;
    for (shape) |dim| {
        total *= dim;
    }
    return total;
}

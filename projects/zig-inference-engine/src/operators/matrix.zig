const std = @import("std");
const Allocator = std.mem.Allocator;

// Import common interfaces and registry
const common_interfaces = @import("common-interfaces");
const TensorInterface = common_interfaces.TensorInterface;
const OperatorInfo = @import("registry.zig").OperatorInfo;
const OperatorFn = @import("registry.zig").OperatorFn;
const ValidatorFn = @import("registry.zig").ValidatorFn;

/// Matrix multiplication operator: C = A @ B
pub const MatMul = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "MatMul",
            .description = "Matrix multiplication of two tensors",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
            .supports_broadcasting = true,
            .compute_fn = compute,
            .validate_fn = validate,
        };
    }

    fn compute(
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror!void {
        _ = attributes;
        _ = allocator;

        if (inputs.len != 2 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const a = &inputs[0];
        const b = &inputs[1];
        const c = &outputs[0];

        const a_shape = a.shape();
        const b_shape = b.shape();
        _ = c.shape(); // Suppress unused warning

        // Validate matrix multiplication dimensions
        if (a_shape.len < 2 or b_shape.len < 2) {
            return error.InvalidDimensions;
        }

        const m = a_shape[a_shape.len - 2];
        const k = a_shape[a_shape.len - 1];
        const k2 = b_shape[b_shape.len - 2];
        const n = b_shape[b_shape.len - 1];

        if (k != k2) {
            return error.IncompatibleDimensions;
        }

        // Handle batch dimensions
        const batch_size = calculateBatchSize(a_shape[0 .. a_shape.len - 2]);

        switch (a.dtype()) {
            .f32 => {
                try matmulF32(a, b, c, batch_size, m, k, n);
            },
            .f16 => {
                try matmulF16(a, b, c, batch_size, m, k, n);
            },
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        if (input_shapes.len != 2) {
            return error.InvalidInputCount;
        }

        const a_shape = input_shapes[0];
        const b_shape = input_shapes[1];

        if (a_shape.len < 2 or b_shape.len < 2) {
            return error.InvalidDimensions;
        }

        const k1 = a_shape[a_shape.len - 1];
        const k2 = b_shape[b_shape.len - 2];

        if (k1 != k2) {
            return error.IncompatibleDimensions;
        }

        // Calculate output shape
        const m = a_shape[a_shape.len - 2];
        const n = b_shape[b_shape.len - 1];

        // Handle batch dimensions
        const max_batch_dims = @max(a_shape.len - 2, b_shape.len - 2);
        var output_shape = try allocator.alloc(usize, max_batch_dims + 2);

        // Copy batch dimensions (simplified - assumes compatible batch dims)
        for (0..max_batch_dims) |i| {
            if (i < a_shape.len - 2) {
                output_shape[i] = a_shape[i];
            } else {
                output_shape[i] = 1;
            }
        }

        output_shape[max_batch_dims] = m;
        output_shape[max_batch_dims + 1] = n;

        var result = try allocator.alloc([]usize, 1);
        result[0] = output_shape;
        return result;
    }
};

/// General matrix multiplication (GEMM): C = alpha * A @ B + beta * C
pub const Gemm = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Gemm",
            .description = "General matrix multiplication with scaling",
            .min_inputs = 2,
            .max_inputs = 3,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
            .supports_broadcasting = false,
            .compute_fn = compute,
            .validate_fn = validate,
        };
    }

    fn compute(
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror!void {
        _ = allocator;

        if (inputs.len < 2 or inputs.len > 3 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const a = &inputs[0];
        const b = &inputs[1];
        const c_bias = if (inputs.len == 3) &inputs[2] else null;
        const c = &outputs[0];

        // Parse attributes
        const alpha = parseFloatAttribute(attributes, "alpha") orelse 1.0;
        const beta = parseFloatAttribute(attributes, "beta") orelse 1.0;
        const trans_a = parseBoolAttribute(attributes, "transA") orelse false;
        const trans_b = parseBoolAttribute(attributes, "transB") orelse false;

        const a_shape = a.shape();
        const b_shape = b.shape();

        if (a_shape.len != 2 or b_shape.len != 2) {
            return error.InvalidDimensions;
        }

        const m = if (trans_a) a_shape[1] else a_shape[0];
        const k = if (trans_a) a_shape[0] else a_shape[1];
        const k2 = if (trans_b) b_shape[1] else b_shape[0];
        const n = if (trans_b) b_shape[0] else b_shape[1];

        if (k != k2) {
            return error.IncompatibleDimensions;
        }

        switch (a.dtype()) {
            .f32 => {
                try gemmF32(a, b, c_bias, c, alpha, beta, trans_a, trans_b, m, k, n);
            },
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        if (input_shapes.len < 2 or input_shapes.len > 3) {
            return error.InvalidInputCount;
        }

        const a_shape = input_shapes[0];
        const b_shape = input_shapes[1];

        if (a_shape.len != 2 or b_shape.len != 2) {
            return error.InvalidDimensions;
        }

        // For simplicity, assume no transpose
        const m = a_shape[0];
        const n = b_shape[1];

        var output_shape = try allocator.alloc(usize, 2);
        output_shape[0] = m;
        output_shape[1] = n;

        var result = try allocator.alloc([]usize, 1);
        result[0] = output_shape;
        return result;
    }
};

/// Matrix transpose operator: B = A^T
pub const Transpose = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Transpose",
            .description = "Matrix transpose operation",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = false,
            .supports_broadcasting = false,
            .compute_fn = compute,
            .validate_fn = validate,
        };
    }

    fn compute(
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror!void {
        _ = allocator;

        if (inputs.len != 1 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const a = &inputs[0];
        const b = &outputs[0];

        // Parse permutation from attributes
        const perm = parsePermAttribute(attributes, "perm") orelse getDefaultPermutation(a.shape().len);

        switch (a.dtype()) {
            .f32 => {
                try transposeF32(a, b, perm);
            },
            .f16 => {
                try transposeF16(a, b, perm);
            },
            .i32 => {
                try transposeI32(a, b, perm);
            },
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        if (input_shapes.len != 1) {
            return error.InvalidInputCount;
        }

        const input_shape = input_shapes[0];
        const perm = parsePermAttribute(attributes, "perm") orelse getDefaultPermutation(input_shape.len);

        var output_shape = try allocator.alloc(usize, input_shape.len);
        for (perm, 0..) |p, i| {
            output_shape[i] = input_shape[p];
        }

        var result = try allocator.alloc([]usize, 1);
        result[0] = output_shape;
        return result;
    }
};

// Helper functions for matrix operations

fn matmulF32(
    a: *const TensorInterface,
    b: *const TensorInterface,
    c: *const TensorInterface,
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
) !void {
    const a_data = std.mem.bytesAsSlice(f32, a.data());
    const b_data = std.mem.bytesAsSlice(f32, b.data());
    const c_data = std.mem.bytesAsSlice(f32, c.data());

    for (0..batch_size) |batch| {
        const a_offset = batch * m * k;
        const b_offset = batch * k * n;
        const c_offset = batch * m * n;

        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..k) |l| {
                    const a_idx = a_offset + i * k + l;
                    const b_idx = b_offset + l * n + j;
                    sum += a_data[a_idx] * b_data[b_idx];
                }
                const c_idx = c_offset + i * n + j;
                c_data[c_idx] = sum;
            }
        }
    }
}

fn matmulF16(
    a: *const TensorInterface,
    b: *const TensorInterface,
    c: *const TensorInterface,
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
) !void {
    const a_data = std.mem.bytesAsSlice(f16, a.data());
    const b_data = std.mem.bytesAsSlice(f16, b.data());
    const c_data = std.mem.bytesAsSlice(f16, c.data());

    for (0..batch_size) |batch| {
        const a_offset = batch * m * k;
        const b_offset = batch * k * n;
        const c_offset = batch * m * n;

        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f16 = 0.0;
                for (0..k) |l| {
                    const a_idx = a_offset + i * k + l;
                    const b_idx = b_offset + l * n + j;
                    sum += a_data[a_idx] * b_data[b_idx];
                }
                const c_idx = c_offset + i * n + j;
                c_data[c_idx] = sum;
            }
        }
    }
}

fn gemmF32(
    a: *const TensorInterface,
    b: *const TensorInterface,
    c_bias: ?*const TensorInterface,
    c: *const TensorInterface,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    m: usize,
    k: usize,
    n: usize,
) !void {
    _ = trans_a;
    _ = trans_b;

    const a_data = std.mem.bytesAsSlice(f32, a.data());
    const b_data = std.mem.bytesAsSlice(f32, b.data());
    const c_data = std.mem.bytesAsSlice(f32, c.data());

    const bias_data = if (c_bias) |bias| std.mem.bytesAsSlice(f32, bias.data()) else null;

    // Simplified GEMM implementation (assumes no transpose for now)
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |l| {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }

            var result = alpha * sum;
            if (bias_data) |bias| {
                result += beta * bias[i * n + j];
            }

            c_data[i * n + j] = result;
        }
    }
}

fn transposeF32(a: *const TensorInterface, b: *const TensorInterface, perm: []const usize) !void {
    const a_data = std.mem.bytesAsSlice(f32, a.data());
    const b_data = std.mem.bytesAsSlice(f32, b.data());
    const a_shape = a.shape();

    // Simplified transpose for 2D case
    if (a_shape.len == 2 and perm.len == 2 and perm[0] == 1 and perm[1] == 0) {
        const rows = a_shape[0];
        const cols = a_shape[1];

        for (0..rows) |i| {
            for (0..cols) |j| {
                b_data[j * rows + i] = a_data[i * cols + j];
            }
        }
    } else {
        // General N-dimensional transpose (simplified implementation)
        const numel = a.numel();
        for (0..numel) |i| {
            const src_indices = linearToMultiIndex(i, a_shape);
            var dst_indices = try std.heap.page_allocator.alloc(usize, src_indices.len);
            defer std.heap.page_allocator.free(dst_indices);

            for (perm, 0..) |p, j| {
                dst_indices[j] = src_indices[p];
            }

            const dst_idx = multiIndexToLinear(dst_indices, b.shape());
            b_data[dst_idx] = a_data[i];
        }
    }
}

fn transposeF16(a: *const TensorInterface, b: *const TensorInterface, perm: []const usize) !void {
    _ = a;
    _ = b;
    _ = perm;
    // Similar implementation for f16
}

fn transposeI32(a: *const TensorInterface, b: *const TensorInterface, perm: []const usize) !void {
    _ = a;
    _ = b;
    _ = perm;
    // Similar implementation for i32
}

// Utility functions
fn calculateBatchSize(batch_dims: []const usize) usize {
    var size: usize = 1;
    for (batch_dims) |dim| {
        size *= dim;
    }
    return size;
}

fn parseFloatAttribute(attributes: std.StringHashMap([]const u8), key: []const u8) ?f32 {
    if (attributes.get(key)) |value| {
        return std.fmt.parseFloat(f32, value) catch null;
    }
    return null;
}

fn parseBoolAttribute(attributes: std.StringHashMap([]const u8), key: []const u8) ?bool {
    if (attributes.get(key)) |value| {
        return std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "1");
    }
    return null;
}

fn parsePermAttribute(attributes: std.StringHashMap([]const u8), key: []const u8) ?[]const usize {
    _ = attributes;
    _ = key;
    // TODO: Parse permutation array from string
    return null;
}

fn getDefaultPermutation(ndim: usize) []const usize {
    // Default permutation reverses all dimensions
    var perm = std.heap.page_allocator.alloc(usize, ndim) catch return &[_]usize{};
    for (0..ndim) |i| {
        perm[i] = ndim - 1 - i;
    }
    return perm;
}

fn linearToMultiIndex(linear_idx: usize, shape: []const usize) []usize {
    var indices = std.heap.page_allocator.alloc(usize, shape.len) catch return &[_]usize{};
    var remaining = linear_idx;

    for (0..shape.len) |i| {
        const dim_idx = shape.len - 1 - i;
        indices[dim_idx] = remaining % shape[dim_idx];
        remaining /= shape[dim_idx];
    }

    return indices;
}

fn multiIndexToLinear(indices: []const usize, shape: []const usize) usize {
    var linear_idx: usize = 0;
    var stride: usize = 1;

    for (0..shape.len) |i| {
        const dim_idx = shape.len - 1 - i;
        linear_idx += indices[dim_idx] * stride;
        stride *= shape[dim_idx];
    }

    return linear_idx;
}

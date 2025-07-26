const std = @import("std");
const Allocator = std.mem.Allocator;

// Import common interfaces and registry
const common_interfaces = @import("common-interfaces");
const TensorInterface = common_interfaces.TensorInterface;
const OperatorInfo = @import("registry.zig").OperatorInfo;
const OperatorFn = @import("registry.zig").OperatorFn;
const ValidatorFn = @import("registry.zig").ValidatorFn;

/// Addition operator: C = A + B
pub const Add = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Add",
            .description = "Element-wise addition of two tensors",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
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

        if (inputs.len != 2 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const a = &inputs[0];
        const b = &inputs[1];
        var c = &outputs[0];

        // Check shapes are broadcastable
        if (!canBroadcast(a.shape(), b.shape())) {
            return error.IncompatibleShapes;
        }

        // Perform element-wise addition using tensor interface
        const a_shape = a.shape();
        const output_shape = try broadcastShapes(allocator, a_shape, b.shape());
        defer allocator.free(output_shape);

        // For now, implement simple element-wise addition for same-shaped tensors
        if (!shapesCompatible(a_shape, b.shape())) {
            return error.IncompatibleShapes;
        }

        switch (a.dtype()) {
            .f32 => {
                try addF32Tensors(a, b, c);
            },
            .f16 => {
                try addF16Tensors(a, b, c);
            },
            .i32 => {
                try addI32Tensors(a, b, c);
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

        const shape_a = input_shapes[0];
        const shape_b = input_shapes[1];

        if (!shapesCompatible(shape_a, shape_b)) {
            return error.IncompatibleShapes;
        }

        // Output shape is the broadcasted shape
        const output_shape = try broadcastShapes(allocator, shape_a, shape_b);
        var result = try allocator.alloc([]usize, 1);
        result[0] = output_shape;
        return result;
    }
};

/// Subtraction operator: C = A - B
pub const Sub = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Sub",
            .description = "Element-wise subtraction of two tensors",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
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
        var c = &outputs[0];

        // Check shapes are compatible
        if (!shapesCompatible(a.shape(), b.shape())) {
            return error.IncompatibleShapes;
        }

        switch (a.dtype()) {
            .f32 => {
                try subF32Tensors(a, b, c);
            },
            .f16 => {
                try subF16Tensors(a, b, c);
            },
            .i32 => {
                try subI32Tensors(a, b, c);
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

        const output_shape = try broadcastShapes(allocator, input_shapes[0], input_shapes[1]);
        var result = try allocator.alloc([]usize, 1);
        result[0] = output_shape;
        return result;
    }
};

/// Multiplication operator: C = A * B
pub const Mul = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Mul",
            .description = "Element-wise multiplication of two tensors",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
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

        const a = &inputs[0];
        const b = &inputs[1];
        const c = &outputs[0];

        const numel = a.numel();
        const a_data = a.data();
        const b_data = b.data();
        const c_data = c.data();

        switch (a.dtype()) {
            .f32 => {
                const a_f32 = std.mem.bytesAsSlice(f32, a_data);
                const b_f32 = std.mem.bytesAsSlice(f32, b_data);
                const c_f32 = std.mem.bytesAsSlice(f32, c_data);

                for (0..numel) |i| {
                    c_f32[i] = a_f32[i] * b_f32[i];
                }
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

        const output_shape = try broadcastShapes(allocator, input_shapes[0], input_shapes[1]);
        var result = try allocator.alloc([]usize, 1);
        result[0] = output_shape;
        return result;
    }
};

/// Division operator: C = A / B
pub const Div = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Div",
            .description = "Element-wise division of two tensors",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
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

        const a = &inputs[0];
        const b = &inputs[1];
        const c = &outputs[0];

        const numel = a.numel();
        const a_data = a.data();
        const b_data = b.data();
        const c_data = c.data();

        switch (a.dtype()) {
            .f32 => {
                const a_f32 = std.mem.bytesAsSlice(f32, a_data);
                const b_f32 = std.mem.bytesAsSlice(f32, b_data);
                const c_f32 = std.mem.bytesAsSlice(f32, c_data);

                for (0..numel) |i| {
                    if (b_f32[i] == 0.0) {
                        return error.DivisionByZero;
                    }
                    c_f32[i] = a_f32[i] / b_f32[i];
                }
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

        const output_shape = try broadcastShapes(allocator, input_shapes[0], input_shapes[1]);
        var result = try allocator.alloc([]usize, 1);
        result[0] = output_shape;
        return result;
    }
};

/// Power operator: C = A ^ B
pub const Pow = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Pow",
            .description = "Element-wise power operation",
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
            .supports_inplace = true,
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

        const a = &inputs[0];
        const b = &inputs[1];
        const c = &outputs[0];

        const numel = a.numel();
        const a_data = a.data();
        const b_data = b.data();
        const c_data = c.data();

        switch (a.dtype()) {
            .f32 => {
                const a_f32 = std.mem.bytesAsSlice(f32, a_data);
                const b_f32 = std.mem.bytesAsSlice(f32, b_data);
                const c_f32 = std.mem.bytesAsSlice(f32, c_data);

                for (0..numel) |i| {
                    c_f32[i] = std.math.pow(f32, a_f32[i], b_f32[i]);
                }
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

        const output_shape = try broadcastShapes(allocator, input_shapes[0], input_shapes[1]);
        var result = try allocator.alloc([]usize, 1);
        result[0] = output_shape;
        return result;
    }
};

/// Square root operator: C = sqrt(A)
pub const Sqrt = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Sqrt",
            .description = "Element-wise square root",
            .min_inputs = 1,
            .max_inputs = 1,
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
        _ = attributes;
        _ = allocator;

        const a = &inputs[0];
        const c = &outputs[0];

        const numel = a.numel();
        const a_data = a.data();
        const c_data = c.data();

        switch (a.dtype()) {
            .f32 => {
                const a_f32 = std.mem.bytesAsSlice(f32, a_data);
                const c_f32 = std.mem.bytesAsSlice(f32, c_data);

                for (0..numel) |i| {
                    if (a_f32[i] < 0.0) {
                        return error.InvalidInput;
                    }
                    c_f32[i] = std.math.sqrt(a_f32[i]);
                }
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

        if (input_shapes.len != 1) {
            return error.InvalidInputCount;
        }

        var result = try allocator.alloc([]usize, 1);
        result[0] = try allocator.dupe(usize, input_shapes[0]);
        return result;
    }
};

/// Exponential operator: C = exp(A)
pub const Exp = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Exp",
            .description = "Element-wise exponential function",
            .min_inputs = 1,
            .max_inputs = 1,
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
        _ = attributes;
        _ = allocator;

        const a = &inputs[0];
        const c = &outputs[0];

        const numel = a.numel();
        const a_data = a.data();
        const c_data = c.data();

        switch (a.dtype()) {
            .f32 => {
                const a_f32 = std.mem.bytesAsSlice(f32, a_data);
                const c_f32 = std.mem.bytesAsSlice(f32, c_data);

                for (0..numel) |i| {
                    c_f32[i] = std.math.exp(a_f32[i]);
                }
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

        var result = try allocator.alloc([]usize, 1);
        result[0] = try allocator.dupe(usize, input_shapes[0]);
        return result;
    }
};

/// Natural logarithm operator: C = log(A)
pub const Log = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Log",
            .description = "Element-wise natural logarithm",
            .min_inputs = 1,
            .max_inputs = 1,
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
        _ = attributes;
        _ = allocator;

        const a = &inputs[0];
        const c = &outputs[0];

        const numel = a.numel();
        const a_data = a.data();
        const c_data = c.data();

        switch (a.dtype()) {
            .f32 => {
                const a_f32 = std.mem.bytesAsSlice(f32, a_data);
                const c_f32 = std.mem.bytesAsSlice(f32, c_data);

                for (0..numel) |i| {
                    if (a_f32[i] <= 0.0) {
                        return error.InvalidInput;
                    }
                    c_f32[i] = std.math.log(f32, std.math.e, a_f32[i]);
                }
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

        var result = try allocator.alloc([]usize, 1);
        result[0] = try allocator.dupe(usize, input_shapes[0]);
        return result;
    }
};

// Helper functions
fn shapesCompatible(shape_a: []const usize, shape_b: []const usize) bool {
    if (shape_a.len != shape_b.len) return false;

    for (shape_a, shape_b) |a, b| {
        if (a != b) return false;
    }

    return true;
}

fn broadcastShapes(allocator: Allocator, shape_a: []const usize, shape_b: []const usize) ![]usize {
    // For simplicity, assume shapes are identical for now
    // In a full implementation, this would handle broadcasting rules
    if (!shapesCompatible(shape_a, shape_b)) {
        return error.IncompatibleShapes;
    }

    return allocator.dupe(usize, shape_a);
}

// Helper functions for type-specific tensor operations
fn addF32Tensors(a: *const TensorInterface, b: *const TensorInterface, c: *TensorInterface) !void {
    const a_shape = a.shape();

    // Calculate total elements
    var total_elements: usize = 1;
    for (a_shape) |dim| {
        total_elements *= dim;
    }

    // Perform element-wise addition using tensor interface
    for (0..total_elements) |i| {
        const indices = try flatIndexToIndices(a_shape, i);
        defer std.heap.page_allocator.free(indices);

        const a_val = try a.getF32(indices);
        const b_val = try b.getF32(indices);
        try c.setF32(indices, a_val + b_val);
    }
}

fn addF16Tensors(a: *const TensorInterface, b: *const TensorInterface, c: *TensorInterface) !void {
    // Similar to addF32Tensors but for f16
    // For now, convert through f32 since TensorInterface doesn't have getF16/setF16
    const a_shape = a.shape();

    var total_elements: usize = 1;
    for (a_shape) |dim| {
        total_elements *= dim;
    }

    for (0..total_elements) |i| {
        const indices = try flatIndexToIndices(a_shape, i);
        defer std.heap.page_allocator.free(indices);

        const a_val = try a.getF32(indices);
        const b_val = try b.getF32(indices);
        try c.setF32(indices, a_val + b_val);
    }
}

fn addI32Tensors(a: *const TensorInterface, b: *const TensorInterface, c: *TensorInterface) !void {
    // Similar to addF32Tensors but for i32
    // For now, convert through f32 since TensorInterface doesn't have getI32/setI32
    const a_shape = a.shape();

    var total_elements: usize = 1;
    for (a_shape) |dim| {
        total_elements *= dim;
    }

    for (0..total_elements) |i| {
        const indices = try flatIndexToIndices(a_shape, i);
        defer std.heap.page_allocator.free(indices);

        const a_val = try a.getF32(indices);
        const b_val = try b.getF32(indices);
        try c.setF32(indices, a_val + b_val);
    }
}

// Subtraction helper functions
fn subF32Tensors(a: *const TensorInterface, b: *const TensorInterface, c: *TensorInterface) !void {
    const a_shape = a.shape();

    var total_elements: usize = 1;
    for (a_shape) |dim| {
        total_elements *= dim;
    }

    for (0..total_elements) |i| {
        const indices = try flatIndexToIndices(a_shape, i);
        defer std.heap.page_allocator.free(indices);

        const a_val = try a.getF32(indices);
        const b_val = try b.getF32(indices);
        try c.setF32(indices, a_val - b_val);
    }
}

fn subF16Tensors(a: *const TensorInterface, b: *const TensorInterface, c: *TensorInterface) !void {
    const a_shape = a.shape();

    var total_elements: usize = 1;
    for (a_shape) |dim| {
        total_elements *= dim;
    }

    for (0..total_elements) |i| {
        const indices = try flatIndexToIndices(a_shape, i);
        defer std.heap.page_allocator.free(indices);

        const a_val = try a.getF32(indices);
        const b_val = try b.getF32(indices);
        try c.setF32(indices, a_val - b_val);
    }
}

fn subI32Tensors(a: *const TensorInterface, b: *const TensorInterface, c: *TensorInterface) !void {
    const a_shape = a.shape();

    var total_elements: usize = 1;
    for (a_shape) |dim| {
        total_elements *= dim;
    }

    for (0..total_elements) |i| {
        const indices = try flatIndexToIndices(a_shape, i);
        defer std.heap.page_allocator.free(indices);

        const a_val = try a.getF32(indices);
        const b_val = try b.getF32(indices);
        try c.setF32(indices, a_val - b_val);
    }
}

fn flatIndexToIndices(shape: []const usize, flat_index: usize) ![]usize {
    var indices = try std.heap.page_allocator.alloc(usize, shape.len);
    var remaining = flat_index;

    var i = shape.len;
    while (i > 0) {
        i -= 1;
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }

    return indices;
}

/// Check if two shapes are broadcastable according to NumPy rules
fn canBroadcast(shape1: []const usize, shape2: []const usize) bool {
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

/// Compute broadcast shape for two input shapes
fn broadcastShape(shape1: []const usize, shape2: []const usize, allocator: Allocator) ![]usize {
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

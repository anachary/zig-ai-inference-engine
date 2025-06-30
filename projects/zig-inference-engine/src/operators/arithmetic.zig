const std = @import("std");
const Allocator = std.mem.Allocator;

// Import common interfaces and registry
const TensorInterface = @import("../../../common/interfaces/tensor.zig").TensorInterface;
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
        _ = allocator;
        
        if (inputs.len != 2 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const a = &inputs[0];
        const b = &inputs[1];
        const c = &outputs[0];

        // Check shapes are compatible
        if (!shapesCompatible(a.shape(), b.shape())) {
            return error.IncompatibleShapes;
        }

        // Perform element-wise addition
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
                    c_f32[i] = a_f32[i] + b_f32[i];
                }
            },
            .f16 => {
                const a_f16 = std.mem.bytesAsSlice(f16, a_data);
                const b_f16 = std.mem.bytesAsSlice(f16, b_data);
                const c_f16 = std.mem.bytesAsSlice(f16, c_data);
                
                for (0..numel) |i| {
                    c_f16[i] = a_f16[i] + b_f16[i];
                }
            },
            .i32 => {
                const a_i32 = std.mem.bytesAsSlice(i32, a_data);
                const b_i32 = std.mem.bytesAsSlice(i32, b_data);
                const c_i32 = std.mem.bytesAsSlice(i32, c_data);
                
                for (0..numel) |i| {
                    c_i32[i] = a_i32[i] + b_i32[i];
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
                    c_f32[i] = a_f32[i] - b_f32[i];
                }
            },
            .f16 => {
                const a_f16 = std.mem.bytesAsSlice(f16, a_data);
                const b_f16 = std.mem.bytesAsSlice(f16, b_data);
                const c_f16 = std.mem.bytesAsSlice(f16, c_data);
                
                for (0..numel) |i| {
                    c_f16[i] = a_f16[i] - b_f16[i];
                }
            },
            .i32 => {
                const a_i32 = std.mem.bytesAsSlice(i32, a_data);
                const b_i32 = std.mem.bytesAsSlice(i32, b_data);
                const c_i32 = std.mem.bytesAsSlice(i32, c_data);
                
                for (0..numel) |i| {
                    c_i32[i] = a_i32[i] - b_i32[i];
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

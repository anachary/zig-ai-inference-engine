const std = @import("std");
const Allocator = std.mem.Allocator;

// Import common interfaces and registry
const common_interfaces = @import("common-interfaces");
const TensorInterface = common_interfaces.TensorInterface;
const OperatorInfo = @import("registry.zig").OperatorInfo;
const OperatorFn = @import("registry.zig").OperatorFn;
const ValidatorFn = @import("registry.zig").ValidatorFn;

/// LayerNorm operator for transformer models
pub const LayerNorm = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "LayerNorm",
            .description = "Layer normalization for transformer models",
            .min_inputs = 1,
            .max_inputs = 3,
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

        if (inputs.len < 1 or inputs.len > 3 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        // Get epsilon from attributes (default: 1e-5)
        const epsilon = parseFloatAttribute(attributes, "epsilon") orelse 1e-5;

        // Get axis from attributes (default: -1, meaning last dimension)
        const axis = parseIntAttribute(attributes, "axis") orelse -1;

        switch (input.dtype()) {
            .f32 => try layerNormF32(input, output, epsilon, axis, inputs),
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        if (input_shapes.len < 1 or input_shapes.len > 3) {
            return error.InvalidInputCount;
        }

        // Output shape is same as input shape
        var output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, input_shapes[0]);

        return output_shapes;
    }
};

/// Embedding operator for token embeddings
pub const Embedding = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Embedding",
            .description = "Token embedding lookup for language models",
            .min_inputs = 2,
            .max_inputs = 2,
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
        _ = attributes;

        if (inputs.len != 2 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const indices = &inputs[0]; // Token indices
        const weights = &inputs[1]; // Embedding weights
        const output = &outputs[0];

        switch (indices.dtype()) {
            .i32 => try embeddingI32(indices, weights, output),
            .i16 => try embeddingI16(indices, weights, output),
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

        const indices_shape = input_shapes[0];
        const weights_shape = input_shapes[1];

        if (weights_shape.len != 2) {
            return error.InvalidWeightsShape;
        }

        // Output shape: indices_shape + [embedding_dim]
        const embedding_dim = weights_shape[1];
        var output_shape = try allocator.alloc(usize, indices_shape.len + 1);

        for (indices_shape, 0..) |dim, i| {
            output_shape[i] = dim;
        }
        output_shape[indices_shape.len] = embedding_dim;

        var output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = output_shape;

        return output_shapes;
    }
};

/// Multi-Head Attention operator
pub const MultiHeadAttention = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "MultiHeadAttention",
            .description = "Multi-head attention mechanism for transformers",
            .min_inputs = 3,
            .max_inputs = 5,
            .min_outputs = 1,
            .max_outputs = 2,
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

        if (inputs.len < 3 or inputs.len > 5 or outputs.len < 1 or outputs.len > 2) {
            return error.InvalidInputOutput;
        }

        const query = &inputs[0];
        const key = &inputs[1];
        const value = &inputs[2];
        const output = &outputs[0];

        // Get number of heads from attributes
        const num_heads = parseIntAttribute(attributes, "num_heads") orelse 8;

        switch (query.dtype()) {
            .f32 => try multiHeadAttentionF32(query, key, value, output, num_heads),
            else => return error.UnsupportedDataType,
        }
    }

    fn validate(
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
        allocator: Allocator,
    ) anyerror![][]usize {
        _ = attributes;

        if (input_shapes.len < 3 or input_shapes.len > 5) {
            return error.InvalidInputCount;
        }

        const query_shape = input_shapes[0];

        // Output shape is same as query shape
        var output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, query_shape);

        return output_shapes;
    }
};

/// GELU activation function
pub const GELU = struct {
    pub fn getInfo() OperatorInfo {
        return OperatorInfo{
            .name = "Gelu",
            .description = "Gaussian Error Linear Unit activation",
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
        _ = allocator;
        _ = attributes;

        if (inputs.len != 1 or outputs.len != 1) {
            return error.InvalidInputOutput;
        }

        const input = &inputs[0];
        const output = &outputs[0];

        switch (input.dtype()) {
            .f32 => try geluF32(input, output),
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

        // Output shape is same as input shape
        var output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, input_shapes[0]);

        return output_shapes;
    }
};

// Helper functions for attribute parsing
fn parseFloatAttribute(attributes: std.StringHashMap([]const u8), name: []const u8) ?f32 {
    if (attributes.get(name)) |value_str| {
        return std.fmt.parseFloat(f32, value_str) catch null;
    }
    return null;
}

fn parseIntAttribute(attributes: std.StringHashMap([]const u8), name: []const u8) ?i32 {
    if (attributes.get(name)) |value_str| {
        return std.fmt.parseInt(i32, value_str, 10) catch null;
    }
    return null;
}

// Implementation functions (simplified for now)
fn layerNormF32(input: *const TensorInterface, output: *const TensorInterface, epsilon: f32, axis: i32, inputs: []const TensorInterface) !void {
    _ = input;
    _ = output;
    _ = epsilon;
    _ = axis;
    _ = inputs;
    // TODO: Implement LayerNorm computation
    return error.NotImplemented;
}

fn embeddingI32(indices: *const TensorInterface, weights: *const TensorInterface, output: *const TensorInterface) !void {
    _ = indices;
    _ = weights;
    _ = output;
    // TODO: Implement embedding lookup
    return error.NotImplemented;
}

fn embeddingI16(indices: *const TensorInterface, weights: *const TensorInterface, output: *const TensorInterface) !void {
    _ = indices;
    _ = weights;
    _ = output;
    // TODO: Implement embedding lookup
    return error.NotImplemented;
}

fn multiHeadAttentionF32(query: *const TensorInterface, key: *const TensorInterface, value: *const TensorInterface, output: *const TensorInterface, num_heads: i32) !void {
    _ = query;
    _ = key;
    _ = value;
    _ = output;
    _ = num_heads;
    // TODO: Implement multi-head attention
    return error.NotImplemented;
}

fn geluF32(input: *const TensorInterface, output: *const TensorInterface) !void {
    _ = input;
    _ = output;
    // TODO: Implement GELU activation
    return error.NotImplemented;
}

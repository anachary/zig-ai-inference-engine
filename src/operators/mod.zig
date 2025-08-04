const std = @import("std");

// Neural network operators
pub const linear = @import("linear.zig");
pub const activation = @import("activation.zig");
pub const normalization = @import("normalization.zig");
pub const attention = @import("attention.zig");
pub const embedding = @import("embedding.zig");
pub const pooling = @import("pooling.zig");
pub const convolution = @import("convolution.zig");

// Re-export commonly used operators
pub const LinearOp = linear.LinearOp;
pub const ActivationOp = activation.ActivationOp;
pub const NormalizationOp = normalization.NormalizationOp;
pub const AttentionOp = attention.AttentionOp;
pub const EmbeddingOp = embedding.EmbeddingOp;

/// Operator execution context
pub const OpContext = struct {
    allocator: std.mem.Allocator,
    device: Device,
    
    pub const Device = enum {
        cpu,
        gpu,
    };
    
    pub fn init(allocator: std.mem.Allocator, device: Device) OpContext {
        return OpContext{
            .allocator = allocator,
            .device = device,
        };
    }
};

/// Base operator interface
pub const Operator = struct {
    name: []const u8,
    op_type: OpType,
    vtable: *const VTable,
    impl: *anyopaque,
    
    pub const OpType = enum {
        linear,
        activation,
        normalization,
        attention,
        embedding,
        pooling,
        convolution,
        reshape,
        transpose,
        concat,
        split,
    };
    
    pub const VTable = struct {
        deinit: *const fn (impl: *anyopaque, allocator: std.mem.Allocator) void,
        forward: *const fn (impl: *anyopaque, inputs: []const Tensor, outputs: []Tensor, context: *OpContext) anyerror!void,
        getOutputShape: *const fn (impl: *anyopaque, input_shapes: []const []const usize) []const usize,
    };
    
    pub fn init(
        name: []const u8,
        op_type: OpType,
        vtable: *const VTable,
        impl: *anyopaque,
    ) Operator {
        return Operator{
            .name = name,
            .op_type = op_type,
            .vtable = vtable,
            .impl = impl,
        };
    }
    
    pub fn deinit(self: *Operator, allocator: std.mem.Allocator) void {
        self.vtable.deinit(self.impl, allocator);
    }
    
    pub fn forward(
        self: *Operator,
        inputs: []const Tensor,
        outputs: []Tensor,
        context: *OpContext,
    ) !void {
        return self.vtable.forward(self.impl, inputs, outputs, context);
    }
    
    pub fn getOutputShape(self: *Operator, input_shapes: []const []const usize) []const usize {
        return self.vtable.getOutputShape(self.impl, input_shapes);
    }
};

/// Tensor data structure for operators
pub const Tensor = struct {
    data: []f32,
    shape: []const usize,
    strides: []const usize,
    allocator: ?std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        var total_size: usize = 1;
        for (shape) |dim| {
            total_size *= dim;
        }
        
        const data = try allocator.alloc(f32, total_size);
        const shape_copy = try allocator.dupe(usize, shape);
        const strides = try computeStrides(allocator, shape);
        
        return Tensor{
            .data = data,
            .shape = shape_copy,
            .strides = strides,
            .allocator = allocator,
        };
    }
    
    pub fn initFromData(
        allocator: std.mem.Allocator,
        data: []f32,
        shape: []const usize,
    ) !Tensor {
        const shape_copy = try allocator.dupe(usize, shape);
        const strides = try computeStrides(allocator, shape);
        
        return Tensor{
            .data = data,
            .shape = shape_copy,
            .strides = strides,
            .allocator = null, // Data not owned by this tensor
        };
    }
    
    pub fn deinit(self: *Tensor) void {
        if (self.allocator) |allocator| {
            allocator.free(self.data);
            allocator.free(self.shape);
            allocator.free(self.strides);
        }
    }
    
    pub fn get(self: Tensor, indices: []const usize) f32 {
        std.debug.assert(indices.len == self.shape.len);
        
        var offset: usize = 0;
        for (indices, self.strides) |idx, stride| {
            offset += idx * stride;
        }
        
        return self.data[offset];
    }
    
    pub fn set(self: *Tensor, indices: []const usize, value: f32) void {
        std.debug.assert(indices.len == self.shape.len);
        
        var offset: usize = 0;
        for (indices, self.strides) |idx, stride| {
            offset += idx * stride;
        }
        
        self.data[offset] = value;
    }
    
    pub fn numel(self: Tensor) usize {
        var total: usize = 1;
        for (self.shape) |dim| {
            total *= dim;
        }
        return total;
    }
    
    pub fn reshape(self: Tensor, allocator: std.mem.Allocator, new_shape: []const usize) !Tensor {
        // Verify that total elements remain the same
        var new_total: usize = 1;
        for (new_shape) |dim| {
            new_total *= dim;
        }
        
        if (new_total != self.numel()) {
            return error.IncompatibleShape;
        }
        
        const shape_copy = try allocator.dupe(usize, new_shape);
        const strides = try computeStrides(allocator, new_shape);
        
        return Tensor{
            .data = self.data,
            .shape = shape_copy,
            .strides = strides,
            .allocator = allocator,
        };
    }
    
    fn computeStrides(allocator: std.mem.Allocator, shape: []const usize) ![]usize {
        var strides = try allocator.alloc(usize, shape.len);
        
        if (shape.len == 0) return strides;
        
        strides[shape.len - 1] = 1;
        if (shape.len > 1) {
            var i = shape.len - 1;
            while (i > 0) {
                i -= 1;
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        
        return strides;
    }
};

/// Operator registry for dynamic operator creation
pub const OperatorRegistry = struct {
    operators: std.StringHashMap(OperatorFactory),
    allocator: std.mem.Allocator,
    
    const OperatorFactory = struct {
        create_fn: *const fn (allocator: std.mem.Allocator, params: std.json.Value) anyerror!Operator,
    };
    
    pub fn init(allocator: std.mem.Allocator) OperatorRegistry {
        return OperatorRegistry{
            .operators = std.StringHashMap(OperatorFactory).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *OperatorRegistry) void {
        self.operators.deinit();
    }
    
    pub fn register(
        self: *OperatorRegistry,
        name: []const u8,
        create_fn: *const fn (allocator: std.mem.Allocator, params: std.json.Value) anyerror!Operator,
    ) !void {
        try self.operators.put(name, OperatorFactory{ .create_fn = create_fn });
    }
    
    pub fn create(
        self: *OperatorRegistry,
        name: []const u8,
        params: std.json.Value,
    ) !Operator {
        if (self.operators.get(name)) |factory| {
            return factory.create_fn(self.allocator, params);
        }
        return error.OperatorNotFound;
    }
};

/// Initialize default operator registry
pub fn createDefaultRegistry(allocator: std.mem.Allocator) !OperatorRegistry {
    var registry = OperatorRegistry.init(allocator);
    
    // Register built-in operators
    try registry.register("Linear", linear.createLinearOp);
    try registry.register("ReLU", activation.createReLUOp);
    try registry.register("GELU", activation.createGELUOp);
    try registry.register("Softmax", activation.createSoftmaxOp);
    try registry.register("LayerNorm", normalization.createLayerNormOp);
    try registry.register("Attention", attention.createAttentionOp);
    try registry.register("Embedding", embedding.createEmbeddingOp);
    
    return registry;
}

test "tensor creation and access" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const shape = [_]usize{ 2, 3, 4 };
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();
    
    try testing.expect(tensor.numel() == 24);
    
    const indices = [_]usize{ 1, 2, 3 };
    tensor.set(&indices, 42.0);
    try testing.expect(tensor.get(&indices) == 42.0);
}

test "tensor reshape" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const shape = [_]usize{ 2, 6 };
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();
    
    const new_shape = [_]usize{ 3, 4 };
    var reshaped = try tensor.reshape(allocator, &new_shape);
    defer {
        allocator.free(reshaped.shape);
        allocator.free(reshaped.strides);
    }
    
    try testing.expect(reshaped.numel() == 12);
    try testing.expect(reshaped.shape[0] == 3);
    try testing.expect(reshaped.shape[1] == 4);
}

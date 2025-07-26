const std = @import("std");
const Allocator = std.mem.Allocator;

// Import common interfaces
const common_interfaces = @import("common-interfaces");
const TensorInterface = common_interfaces.TensorInterface;
// ModelInterface will be defined locally for now

/// Operator function signature
pub const OperatorFn = *const fn (
    inputs: []const TensorInterface,
    outputs: []TensorInterface,
    attributes: std.StringHashMap([]const u8),
    allocator: Allocator,
) anyerror!void;

/// Operator validation function signature
pub const ValidatorFn = *const fn (
    input_shapes: []const []const usize,
    attributes: std.StringHashMap([]const u8),
    allocator: Allocator,
) anyerror![][]usize;

/// Operator information
pub const OperatorInfo = struct {
    name: []const u8,
    description: []const u8,
    min_inputs: usize,
    max_inputs: usize,
    min_outputs: usize,
    max_outputs: usize,
    supports_inplace: bool,
    supports_broadcasting: bool,
    compute_fn: OperatorFn,
    validate_fn: ValidatorFn,
};

/// Operator registry errors
pub const RegistryError = error{
    OperatorNotFound,
    OperatorAlreadyExists,
    InvalidOperator,
    ValidationFailed,
};

/// Registry for managing operators
pub const OperatorRegistry = struct {
    allocator: Allocator,
    operators: std.StringHashMap(OperatorInfo),

    const Self = @This();

    /// Initialize the operator registry
    pub fn init(allocator: Allocator) !Self {
        return Self{
            .allocator = allocator,
            .operators = std.StringHashMap(OperatorInfo).init(allocator),
        };
    }

    /// Deinitialize the registry
    pub fn deinit(self: *Self) void {
        self.operators.deinit();
    }

    /// Register a new operator
    pub fn registerOperator(self: *Self, info: OperatorInfo) !void {
        if (self.operators.contains(info.name)) {
            return RegistryError.OperatorAlreadyExists;
        }

        try self.operators.put(info.name, info);
    }

    /// Get operator information
    pub fn getOperator(self: *const Self, name: []const u8) ?OperatorInfo {
        return self.operators.get(name);
    }

    /// Execute an operator
    pub fn executeOperator(
        self: *const Self,
        name: []const u8,
        inputs: []const TensorInterface,
        outputs: []TensorInterface,
        attributes: std.StringHashMap([]const u8),
    ) !void {
        const op_info = self.getOperator(name) orelse return RegistryError.OperatorNotFound;

        // Validate input/output counts
        if (inputs.len < op_info.min_inputs or inputs.len > op_info.max_inputs) {
            return RegistryError.ValidationFailed;
        }

        if (outputs.len < op_info.min_outputs or outputs.len > op_info.max_outputs) {
            return RegistryError.ValidationFailed;
        }

        // Execute the operator
        try op_info.compute_fn(inputs, outputs, attributes, self.allocator);
    }

    /// Validate operator with given shapes
    pub fn validateOperator(
        self: *const Self,
        name: []const u8,
        input_shapes: []const []const usize,
        attributes: std.StringHashMap([]const u8),
    ) ![][]usize {
        const op_info = self.getOperator(name) orelse return RegistryError.OperatorNotFound;
        return op_info.validate_fn(input_shapes, attributes, self.allocator);
    }

    /// Register all built-in operators
    pub fn registerBuiltinOperators(self: *Self) !void {
        // Import operator implementations
        const arithmetic = @import("arithmetic.zig");
        const matrix = @import("matrix.zig");
        const conv = @import("conv.zig");
        const activation = @import("activation.zig");
        const pooling = @import("pooling.zig");
        const normalization = @import("normalization.zig");
        const reduction = @import("reduction.zig");
        const shape_ops = @import("shape.zig");
        const advanced = @import("advanced.zig");

        // Register arithmetic operators
        try self.registerOperator(arithmetic.Add.getInfo());
        try self.registerOperator(arithmetic.Sub.getInfo());
        try self.registerOperator(arithmetic.Mul.getInfo());
        try self.registerOperator(arithmetic.Div.getInfo());
        try self.registerOperator(arithmetic.Pow.getInfo());
        try self.registerOperator(arithmetic.Sqrt.getInfo());
        try self.registerOperator(arithmetic.Exp.getInfo());
        try self.registerOperator(arithmetic.Log.getInfo());

        // Register matrix operators
        try self.registerOperator(matrix.MatMul.getInfo());
        try self.registerOperator(matrix.Gemm.getInfo());
        try self.registerOperator(matrix.Transpose.getInfo());

        // Register convolution operators
        try self.registerOperator(conv.Conv2D.getInfo());
        try self.registerOperator(conv.Conv3D.getInfo());
        try self.registerOperator(conv.DepthwiseConv2D.getInfo());

        // Register activation functions
        try self.registerOperator(activation.ReLU.getInfo());
        try self.registerOperator(activation.Sigmoid.getInfo());
        try self.registerOperator(activation.Tanh.getInfo());
        try self.registerOperator(activation.Softmax.getInfo());
        try self.registerOperator(activation.GELU.getInfo());
        try self.registerOperator(activation.Swish.getInfo());

        // Register pooling operators
        try self.registerOperator(pooling.MaxPool.getInfo());
        try self.registerOperator(pooling.AvgPool.getInfo());
        try self.registerOperator(pooling.GlobalAvgPool.getInfo());

        // Register normalization operators
        try self.registerOperator(normalization.BatchNorm.getInfo());
        try self.registerOperator(normalization.LayerNorm.getInfo());
        try self.registerOperator(normalization.InstanceNorm.getInfo());

        // Register reduction operators
        try self.registerOperator(reduction.ReduceSum.getInfo());
        try self.registerOperator(reduction.ReduceMean.getInfo());
        try self.registerOperator(reduction.ReduceMax.getInfo());
        try self.registerOperator(reduction.ReduceMin.getInfo());

        // Register shape manipulation operators
        try self.registerOperator(shape_ops.Concat.getInfo());
        try self.registerOperator(shape_ops.Split.getInfo());
        try self.registerOperator(shape_ops.Slice.getInfo());
        try self.registerOperator(shape_ops.Squeeze.getInfo());
        try self.registerOperator(shape_ops.Unsqueeze.getInfo());
        try self.registerOperator(shape_ops.Gather.getInfo());
        try self.registerOperator(shape_ops.Scatter.getInfo());
        try self.registerOperator(shape_ops.Reshape.getInfo());
        try self.registerOperator(shape_ops.Constant.getInfo());

        // Register advanced operators for modern architectures
        try self.registerOperator(advanced.LayerNorm.getInfo());
        try self.registerOperator(advanced.Embedding.getInfo());
        try self.registerOperator(advanced.MultiHeadAttention.getInfo());
        try self.registerOperator(advanced.GELU.getInfo());

        std.log.info("Registered {} built-in operators", .{self.operators.count()});
    }

    /// List all registered operators
    pub fn listOperators(self: *const Self, allocator: Allocator) ![][]const u8 {
        var names = std.ArrayList([]const u8).init(allocator);
        defer names.deinit();

        var iterator = self.operators.iterator();
        while (iterator.next()) |entry| {
            try names.append(entry.key_ptr.*);
        }

        return names.toOwnedSlice();
    }

    /// Get operator count
    pub fn getOperatorCount(self: *const Self) usize {
        return self.operators.count();
    }

    /// Check if operator exists
    pub fn hasOperator(self: *const Self, name: []const u8) bool {
        return self.operators.contains(name);
    }
};

/// Registry statistics
pub const RegistryStats = struct {
    total_operators: usize,
    arithmetic_ops: usize,
    matrix_ops: usize,
    conv_ops: usize,
    activation_ops: usize,
    pooling_ops: usize,
    normalization_ops: usize,
    reduction_ops: usize,
    shape_ops: usize,
    custom_ops: usize,
};

/// Get registry statistics
pub fn getRegistryStats(registry: *const OperatorRegistry) RegistryStats {
    var stats = RegistryStats{
        .total_operators = registry.getOperatorCount(),
        .arithmetic_ops = 0,
        .matrix_ops = 0,
        .conv_ops = 0,
        .activation_ops = 0,
        .pooling_ops = 0,
        .normalization_ops = 0,
        .reduction_ops = 0,
        .shape_ops = 0,
        .custom_ops = 0,
    };

    var iterator = registry.operators.iterator();
    while (iterator.next()) |entry| {
        const name = entry.key_ptr.*;

        if (std.mem.startsWith(u8, name, "Add") or
            std.mem.startsWith(u8, name, "Sub") or
            std.mem.startsWith(u8, name, "Mul") or
            std.mem.startsWith(u8, name, "Div"))
        {
            stats.arithmetic_ops += 1;
        } else if (std.mem.startsWith(u8, name, "MatMul") or
            std.mem.startsWith(u8, name, "Gemm") or
            std.mem.startsWith(u8, name, "Transpose"))
        {
            stats.matrix_ops += 1;
        } else if (std.mem.startsWith(u8, name, "Conv")) {
            stats.conv_ops += 1;
        } else if (std.mem.startsWith(u8, name, "ReLU") or
            std.mem.startsWith(u8, name, "Sigmoid") or
            std.mem.startsWith(u8, name, "Tanh"))
        {
            stats.activation_ops += 1;
        } else if (std.mem.startsWith(u8, name, "Pool")) {
            stats.pooling_ops += 1;
        } else if (std.mem.startsWith(u8, name, "Norm")) {
            stats.normalization_ops += 1;
        } else if (std.mem.startsWith(u8, name, "Reduce")) {
            stats.reduction_ops += 1;
        } else if (std.mem.startsWith(u8, name, "Concat") or
            std.mem.startsWith(u8, name, "Split") or
            std.mem.startsWith(u8, name, "Slice"))
        {
            stats.shape_ops += 1;
        } else {
            stats.custom_ops += 1;
        }
    }

    return stats;
}

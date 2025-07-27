const std = @import("std");
const framework = @import("../../framework/lib.zig");

// Import all operator implementations
const arithmetic = struct {
    const add = @import("arithmetic/add.zig");
    const sub = @import("arithmetic/sub.zig");
    
    pub const Add = add.Add;
    pub const Sub = sub.Sub;
    pub const Mul = sub.Mul;
    pub const Div = sub.Div;
};

const activation = struct {
    const relu = @import("activation/relu.zig");
    
    pub const ReLU = relu.ReLU;
    pub const Sigmoid = relu.Sigmoid;
    pub const Tanh = relu.Tanh;
    pub const GELU = relu.GELU;
};

const matrix = struct {
    const matmul = @import("matrix/matmul.zig");
    
    pub const MatMul = matmul.MatMul;
    pub const Transpose = matmul.Transpose;
};

const OperatorRegistry = framework.OperatorRegistry;
const OperatorInterface = framework.OperatorInterface;

/// Register all built-in operators with the registry
pub fn registerBuiltinOperators(registry: *OperatorRegistry) !void {
    std.log.info("Registering built-in operators...");

    // Register arithmetic operators
    try registry.registerOperator(arithmetic.Add.getDefinition());
    try registry.registerOperator(arithmetic.Sub.getDefinition());
    try registry.registerOperator(arithmetic.Mul.getDefinition());
    try registry.registerOperator(arithmetic.Div.getDefinition());

    // Register activation operators
    try registry.registerOperator(activation.ReLU.getDefinition());
    try registry.registerOperator(activation.Sigmoid.getDefinition());
    try registry.registerOperator(activation.Tanh.getDefinition());
    try registry.registerOperator(activation.GELU.getDefinition());

    // Register matrix operators
    try registry.registerOperator(matrix.MatMul.getDefinition());
    try registry.registerOperator(matrix.Transpose.getDefinition());

    std.log.info("Registered {} built-in operators", .{registry.operators.count()});
}

/// Get operator by name and version
pub fn getOperator(name: []const u8, version: ?[]const u8) ?OperatorInterface.Definition {
    // This is a compile-time lookup for built-in operators
    // In a real implementation, this would use the registry
    
    const actual_version = version orelse "1.0.0";
    _ = actual_version; // For now, we ignore version
    
    // Arithmetic operators
    if (std.mem.eql(u8, name, "Add")) return arithmetic.Add.getDefinition();
    if (std.mem.eql(u8, name, "Sub")) return arithmetic.Sub.getDefinition();
    if (std.mem.eql(u8, name, "Mul")) return arithmetic.Mul.getDefinition();
    if (std.mem.eql(u8, name, "Div")) return arithmetic.Div.getDefinition();
    
    // Activation operators
    if (std.mem.eql(u8, name, "Relu")) return activation.ReLU.getDefinition();
    if (std.mem.eql(u8, name, "Sigmoid")) return activation.Sigmoid.getDefinition();
    if (std.mem.eql(u8, name, "Tanh")) return activation.Tanh.getDefinition();
    if (std.mem.eql(u8, name, "Gelu")) return activation.GELU.getDefinition();
    
    // Matrix operators
    if (std.mem.eql(u8, name, "MatMul")) return matrix.MatMul.getDefinition();
    if (std.mem.eql(u8, name, "Transpose")) return matrix.Transpose.getDefinition();
    
    return null;
}

/// List all available built-in operators
pub fn listBuiltinOperators() []const OperatorInfo {
    const operators = [_]OperatorInfo{
        // Arithmetic operators
        OperatorInfo{
            .name = "Add",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Element-wise addition of two tensors with broadcasting support",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16, .i32, .i16, .i8 },
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
        },
        OperatorInfo{
            .name = "Sub",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Element-wise subtraction of two tensors with broadcasting support",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16, .i32, .i16, .i8 },
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
        },
        OperatorInfo{
            .name = "Mul",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Element-wise multiplication of two tensors with broadcasting support",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16, .i32, .i16, .i8 },
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
        },
        OperatorInfo{
            .name = "Div",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Element-wise division of two tensors with broadcasting support",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16 },
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
        },
        
        // Activation operators
        OperatorInfo{
            .name = "Relu",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Rectified Linear Unit activation function: f(x) = max(0, x)",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16 },
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
        },
        OperatorInfo{
            .name = "Sigmoid",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Sigmoid activation function: f(x) = 1 / (1 + exp(-x))",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16 },
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
        },
        OperatorInfo{
            .name = "Tanh",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Hyperbolic tangent activation function: f(x) = tanh(x)",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16 },
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
        },
        OperatorInfo{
            .name = "Gelu",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Gaussian Error Linear Unit activation function",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16 },
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
        },
        
        // Matrix operators
        OperatorInfo{
            .name = "MatMul",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Matrix multiplication with broadcasting support for batch dimensions",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16 },
            .min_inputs = 2,
            .max_inputs = 2,
            .min_outputs = 1,
            .max_outputs = 1,
        },
        OperatorInfo{
            .name = "Transpose",
            .version = "1.0.0",
            .domain = "ai.onnx",
            .description = "Transpose the input tensor similar to numpy.transpose",
            .supported_types = &[_]framework.Tensor.DataType{ .f32, .f16, .i32, .i16, .i8, .u8 },
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
        },
    };
    
    return &operators;
}

/// Operator information structure
pub const OperatorInfo = struct {
    name: []const u8,
    version: []const u8,
    domain: []const u8,
    description: []const u8,
    supported_types: []const framework.Tensor.DataType,
    min_inputs: u32,
    max_inputs: u32,
    min_outputs: u32,
    max_outputs: u32,
};

/// Create a registry with all built-in operators pre-registered
pub fn createBuiltinRegistry(allocator: std.mem.Allocator) !OperatorRegistry {
    var registry = OperatorRegistry.init(allocator);
    try registerBuiltinOperators(&registry);
    return registry;
}

/// Operator categories for organization
pub const OperatorCategory = enum {
    arithmetic,
    activation,
    matrix,
    convolution,
    pooling,
    normalization,
    attention,
    embedding,
    shape,
    reduction,
    control_flow,
    custom,
};

/// Get operators by category
pub fn getOperatorsByCategory(category: OperatorCategory) []const OperatorInfo {
    const all_operators = listBuiltinOperators();
    
    return switch (category) {
        .arithmetic => all_operators[0..4], // Add, Sub, Mul, Div
        .activation => all_operators[4..8], // ReLU, Sigmoid, Tanh, GELU
        .matrix => all_operators[8..10],    // MatMul, Transpose
        else => &[_]OperatorInfo{},
    };
}

/// Check if an operator is supported
pub fn isOperatorSupported(name: []const u8) bool {
    return getOperator(name, null) != null;
}

/// Get operator metadata by name
pub fn getOperatorMetadata(name: []const u8) ?OperatorInfo {
    const all_operators = listBuiltinOperators();
    
    for (all_operators) |op_info| {
        if (std.mem.eql(u8, op_info.name, name)) {
            return op_info;
        }
    }
    
    return null;
}

/// Validate operator compatibility with given types
pub fn validateOperatorTypes(name: []const u8, input_types: []const framework.Tensor.DataType) bool {
    const op_info = getOperatorMetadata(name) orelse return false;
    
    for (input_types) |input_type| {
        var type_supported = false;
        for (op_info.supported_types) |supported_type| {
            if (input_type == supported_type) {
                type_supported = true;
                break;
            }
        }
        if (!type_supported) {
            return false;
        }
    }
    
    return true;
}

// Tests
test "operator registry" {
    const allocator = std.testing.allocator;
    
    var registry = try createBuiltinRegistry(allocator);
    defer registry.deinit();
    
    // Test that operators are registered
    try std.testing.expect(registry.hasOperator("Add", null));
    try std.testing.expect(registry.hasOperator("Relu", null));
    try std.testing.expect(registry.hasOperator("MatMul", null));
    
    // Test operator lookup
    const add_def = registry.getOperator("Add", null);
    try std.testing.expect(add_def != null);
    try std.testing.expectEqualStrings("Add", add_def.?.metadata.name);
}

test "operator categories" {
    const arithmetic_ops = getOperatorsByCategory(.arithmetic);
    try std.testing.expect(arithmetic_ops.len == 4);
    
    const activation_ops = getOperatorsByCategory(.activation);
    try std.testing.expect(activation_ops.len == 4);
    
    const matrix_ops = getOperatorsByCategory(.matrix);
    try std.testing.expect(matrix_ops.len == 2);
}

test "operator type validation" {
    const input_types = [_]framework.Tensor.DataType{ .f32, .f32 };
    
    try std.testing.expect(validateOperatorTypes("Add", &input_types));
    try std.testing.expect(validateOperatorTypes("MatMul", &input_types));
    
    const invalid_types = [_]framework.Tensor.DataType{ .bool, .bool };
    try std.testing.expect(!validateOperatorTypes("Add", &invalid_types));
}

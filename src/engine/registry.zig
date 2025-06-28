const std = @import("std");
const Allocator = std.mem.Allocator;
const operators = @import("operators.zig");
const conv = @import("operators/conv.zig");
const pool = @import("operators/pool.zig");
const activation = @import("operators/activation.zig");

pub const RegistryError = error{
    OperatorNotFound,
    OperatorAlreadyExists,
    OutOfMemory,
};

/// Simple operator entry for registry
const OperatorEntry = struct {
    name: []const u8,
    operator: operators.Operator,
};

/// Registry for managing available operators
pub const OperatorRegistry = struct {
    operators: std.ArrayList(OperatorEntry),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .operators = std.ArrayList(OperatorEntry).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free all operator name keys
        for (self.operators.items) |entry| {
            self.allocator.free(entry.name);
        }
        self.operators.deinit();
    }

    /// Register a new operator
    pub fn register(self: *Self, name: []const u8, op: operators.Operator) !void {
        // Check if operator already exists
        for (self.operators.items) |entry| {
            if (std.mem.eql(u8, entry.name, name)) {
                return RegistryError.OperatorAlreadyExists;
            }
        }

        // Clone the name to ensure it's owned by the registry
        const owned_name = try self.allocator.dupe(u8, name);
        try self.operators.append(OperatorEntry{
            .name = owned_name,
            .operator = op,
        });
    }

    /// Get an operator by name
    pub fn get(self: *const Self, name: []const u8) ?operators.Operator {
        for (self.operators.items) |entry| {
            if (std.mem.eql(u8, entry.name, name)) {
                return entry.operator;
            }
        }
        return null;
    }

    /// Check if an operator is registered
    pub fn has(self: *const Self, name: []const u8) bool {
        for (self.operators.items) |entry| {
            if (std.mem.eql(u8, entry.name, name)) {
                return true;
            }
        }
        return false;
    }

    /// Get list of all registered operator names
    pub fn list_operators(self: *const Self, allocator: Allocator) ![][]const u8 {
        const names = try allocator.alloc([]const u8, self.operators.items.len);

        for (self.operators.items, 0..) |entry, i| {
            names[i] = entry.name;
        }

        return names;
    }

    /// Remove an operator from the registry
    pub fn unregister(self: *Self, name: []const u8) !void {
        for (self.operators.items, 0..) |entry, i| {
            if (std.mem.eql(u8, entry.name, name)) {
                self.allocator.free(entry.name);
                _ = self.operators.swapRemove(i);
                return;
            }
        }
        return RegistryError.OperatorNotFound;
    }

    /// Register all built-in operators
    pub fn register_builtin_operators(self: *Self) !void {
        // Basic arithmetic operators
        try self.register("Add", operators.Add.op);
        try self.register("Sub", operators.Sub.op);
        try self.register("Mul", operators.Mul.op);
        try self.register("MatMul", operators.MatMul.op);

        // Basic activation functions
        try self.register("ReLU", operators.ReLU.op);
        try self.register("Softmax", operators.Softmax.op);

        // Enhanced activation functions
        try self.register("Sigmoid", activation.Sigmoid.op);
        try self.register("Tanh", activation.Tanh.op);
        try self.register("GELU", activation.GELU.op);
        try self.register("Swish", activation.Swish.op);
        try self.register("LeakyReLU", activation.LeakyReLU.op);
        try self.register("ELU", activation.ELU.op);

        // Convolution operators
        try self.register("Conv2D", conv.Conv2D.op);
        try self.register("DepthwiseConv2D", conv.DepthwiseConv2D.op);
        try self.register("ConvTranspose2D", conv.ConvTranspose2D.op);

        // Pooling operators
        try self.register("MaxPool2D", pool.MaxPool2D.op);
        try self.register("AvgPool2D", pool.AvgPool2D.op);
        try self.register("GlobalAvgPool2D", pool.GlobalAvgPool2D.op);
        try self.register("AdaptiveAvgPool2D", pool.AdaptiveAvgPool2D.op);
    }

    /// Get registry statistics
    pub fn get_stats(self: *const Self) RegistryStats {
        return RegistryStats{
            .total_operators = @as(u32, @intCast(self.operators.items.len)),
        };
    }
};

pub const RegistryStats = struct {
    total_operators: u32,
};

/// Global operator registry instance
var global_registry: ?OperatorRegistry = null;
var global_registry_mutex: std.Thread.Mutex = std.Thread.Mutex{};

/// Get the global operator registry (thread-safe)
pub fn get_global_registry(allocator: Allocator) !*OperatorRegistry {
    global_registry_mutex.lock();
    defer global_registry_mutex.unlock();

    if (global_registry == null) {
        global_registry = OperatorRegistry.init(allocator);
        try global_registry.?.register_builtin_operators();
    }

    return &global_registry.?;
}

/// Cleanup global registry
pub fn cleanup_global_registry() void {
    global_registry_mutex.lock();
    defer global_registry_mutex.unlock();

    if (global_registry) |*registry| {
        registry.deinit();
        global_registry = null;
    }
}

/// Operator factory for creating operators with parameters
pub const OperatorFactory = struct {
    registry: *OperatorRegistry,

    const Self = @This();

    pub fn init(registry: *OperatorRegistry) Self {
        return Self{
            .registry = registry,
        };
    }

    /// Create an operator instance with parameters
    pub fn create_operator(self: *Self, name: []const u8, params: ?std.json.Value) !operators.Operator {
        _ = params; // TODO: Use parameters for operator configuration

        if (self.registry.get(name)) |op| {
            return op;
        } else {
            return RegistryError.OperatorNotFound;
        }
    }

    /// Validate operator parameters
    pub fn validate_params(self: *Self, name: []const u8, params: std.json.Value) !bool {
        _ = self;
        _ = name;
        _ = params;

        // TODO: Implement parameter validation for each operator type
        return true;
    }
};

/// Operator execution context
pub const ExecutionContext = struct {
    allocator: Allocator,
    temp_tensors: std.ArrayList(operators.tensor.Tensor),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .temp_tensors = std.ArrayList(operators.tensor.Tensor).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        // Clean up temporary tensors
        for (self.temp_tensors.items) |*tensor_item| {
            tensor_item.deinit();
        }
        self.temp_tensors.deinit();
    }

    /// Execute an operator by name
    pub fn execute_operator(
        self: *Self,
        registry: *OperatorRegistry,
        name: []const u8,
        inputs: []const operators.tensor.Tensor,
        outputs: []operators.tensor.Tensor,
    ) !void {
        if (registry.get(name)) |op| {
            try op.forward(inputs, outputs, self.allocator);
        } else {
            return RegistryError.OperatorNotFound;
        }
    }

    /// Create a temporary tensor for intermediate results
    pub fn create_temp_tensor(self: *Self, shape: []const usize, dtype: operators.tensor.DataType) !operators.tensor.Tensor {
        const temp_tensor = try operators.tensor.Tensor.init(self.allocator, shape, dtype);
        try self.temp_tensors.append(temp_tensor);
        return temp_tensor;
    }

    /// Reset temporary tensors (for reuse)
    pub fn reset_temp_tensors(self: *Self) void {
        for (self.temp_tensors.items) |*tensor_item| {
            tensor_item.deinit();
        }
        self.temp_tensors.clearRetainingCapacity();
    }
};

test "operator registry" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var registry = OperatorRegistry.init(allocator);
    defer registry.deinit();

    // Register built-in operators
    try registry.register_builtin_operators();

    // Test operator retrieval
    const add_op = registry.get("Add");
    try testing.expect(add_op != null);
    try testing.expect(std.mem.eql(u8, add_op.?.name, "Add"));

    // Test operator listing
    const op_names = try registry.list_operators(allocator);
    defer allocator.free(op_names);

    try testing.expect(op_names.len >= 6); // At least 6 built-in operators

    // Test operator existence check
    try testing.expect(registry.has("MatMul"));
    try testing.expect(!registry.has("NonExistentOp"));
}

test "execution context" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var registry = OperatorRegistry.init(allocator);
    defer registry.deinit();
    try registry.register_builtin_operators();

    var context = ExecutionContext.init(allocator);
    defer context.deinit();

    // Create test tensors
    const shape = [_]usize{ 2, 2 };
    var a = try operators.tensor.Tensor.init(allocator, &shape, .f32);
    defer a.deinit();
    var b = try operators.tensor.Tensor.init(allocator, &shape, .f32);
    defer b.deinit();
    var result = try operators.tensor.Tensor.init(allocator, &shape, .f32);
    defer result.deinit();

    // Fill test data
    try a.set_f32(&[_]usize{ 0, 0 }, 1.0);
    try a.set_f32(&[_]usize{ 0, 1 }, 2.0);
    try a.set_f32(&[_]usize{ 1, 0 }, 3.0);
    try a.set_f32(&[_]usize{ 1, 1 }, 4.0);

    try b.set_f32(&[_]usize{ 0, 0 }, 0.5);
    try b.set_f32(&[_]usize{ 0, 1 }, 1.5);
    try b.set_f32(&[_]usize{ 1, 0 }, 2.5);
    try b.set_f32(&[_]usize{ 1, 1 }, 3.5);

    // Execute addition through context
    const inputs = [_]operators.tensor.Tensor{ a, b };
    var outputs = [_]operators.tensor.Tensor{result};

    try context.execute_operator(&registry, "Add", &inputs, &outputs);

    // Verify results
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 0, 0 }), 1.5, 1e-6);
    try testing.expectApproxEqAbs(try result.get_f32(&[_]usize{ 1, 1 }), 7.5, 1e-6);
}

const std = @import("std");

// Core framework modules
pub const interfaces = @import("core/interfaces.zig");
pub const execution = @import("core/execution.zig");

// Operator framework modules
pub const operator_base = @import("operators/base.zig");
pub const operator_registry = @import("operators/registry.zig");

// Re-export core types for convenience
pub const Tensor = interfaces.Tensor;
pub const Attributes = interfaces.Attributes;
pub const ExecutionContext = interfaces.ExecutionContext;
pub const FrameworkError = interfaces.FrameworkError;
pub const Profiler = interfaces.Profiler;
pub const MemoryManager = interfaces.MemoryManager;

// Re-export operator types
pub const OperatorInterface = operator_base.OperatorInterface;
pub const BaseOperator = operator_base.BaseOperator;
pub const OperatorUtils = operator_base.OperatorUtils;
pub const OperatorRegistry = operator_registry.OperatorRegistry;

// Re-export execution types
pub const ExecutionEngine = execution.ExecutionEngine;
pub const Graph = execution.ExecutionEngine.Graph;
pub const GraphNode = execution.ExecutionEngine.GraphNode;

/// Framework version information
pub const VERSION = "0.1.0";
pub const VERSION_MAJOR = 0;
pub const VERSION_MINOR = 1;
pub const VERSION_PATCH = 0;

/// Framework initialization
pub const Framework = struct {
    allocator: std.mem.Allocator,
    operator_registry: OperatorRegistry,
    execution_engine: ExecutionEngine,

    const Self = @This();

    /// Framework configuration
    pub const Config = struct {
        device: ExecutionContext.Device = .auto,
        optimization_level: ExecutionContext.OptimizationLevel = .basic,
        enable_profiling: bool = false,
        enable_memory_tracking: bool = true,
        max_memory_mb: ?usize = null,
        num_threads: ?u32 = null,
        operator_search_paths: []const []const u8 = &.{},
    };

    /// Initialize the framework
    pub fn init(allocator: std.mem.Allocator, config: Config) !Self {
        var operator_registry = OperatorRegistry.init(allocator);
        
        // Add search paths for operators
        for (config.operator_search_paths) |path| {
            try operator_registry.addSearchPath(path);
        }

        // Discover operators in search paths
        try operator_registry.discoverOperators();

        const execution_config = ExecutionEngine.Config{
            .device = config.device,
            .optimization_level = config.optimization_level,
            .enable_profiling = config.enable_profiling,
            .enable_memory_tracking = config.enable_memory_tracking,
            .max_memory_mb = config.max_memory_mb,
            .num_threads = config.num_threads,
        };

        const execution_engine = try ExecutionEngine.init(allocator, &operator_registry, execution_config);

        return Self{
            .allocator = allocator,
            .operator_registry = operator_registry,
            .execution_engine = execution_engine,
        };
    }

    /// Deinitialize the framework
    pub fn deinit(self: *Self) void {
        self.execution_engine.deinit();
        self.operator_registry.deinit();
    }

    /// Register a new operator
    pub fn registerOperator(self: *Self, definition: OperatorInterface.Definition) !void {
        try self.operator_registry.registerOperator(definition);
    }

    /// Override an existing operator
    pub fn overrideOperator(self: *Self, definition: OperatorInterface.Definition) !void {
        try self.operator_registry.overrideOperator(definition);
    }

    /// Execute a computational graph
    pub fn executeGraph(self: *Self, graph: *Graph) !void {
        try self.execution_engine.validateGraph(graph);
        try self.execution_engine.executeGraph(graph);
    }

    /// Create a new computational graph
    pub fn createGraph(self: *Self) Graph {
        return Graph.init(self.allocator);
    }

    /// Get framework statistics
    pub fn getStats(self: *const Self) FrameworkStats {
        const execution_stats = self.execution_engine.getExecutionStats();
        return FrameworkStats{
            .registered_operators = self.operator_registry.operators.count(),
            .total_memory_used = execution_stats.total_memory_used,
            .peak_memory_used = execution_stats.peak_memory_used,
            .profiling_data = execution_stats.profiling_data,
        };
    }

    /// Framework statistics
    pub const FrameworkStats = struct {
        registered_operators: u32,
        total_memory_used: usize,
        peak_memory_used: usize,
        profiling_data: []const Profiler.Event,
    };

    /// List all available operators
    pub fn listOperators(self: *const Self) ![]OperatorRegistry.OperatorInfo {
        return self.operator_registry.listOperators();
    }

    /// Check if an operator is available
    pub fn hasOperator(self: *const Self, name: []const u8, version: ?[]const u8) bool {
        return self.operator_registry.hasOperator(name, version);
    }

    /// Set execution device
    pub fn setDevice(self: *Self, device: ExecutionContext.Device) void {
        self.execution_engine.setDevice(device);
    }

    /// Enable/disable profiling
    pub fn setProfilingEnabled(self: *Self, enabled: bool) !void {
        try self.execution_engine.setProfilingEnabled(enabled);
    }
};

/// Utility functions for framework users
pub const utils = struct {
    /// Create a tensor with given shape and data type
    pub fn createTensor(allocator: std.mem.Allocator, shape: []const usize, dtype: Tensor.DataType) !Tensor {
        return Tensor.init(allocator, shape, dtype);
    }

    /// Create attributes container
    pub fn createAttributes(allocator: std.mem.Allocator) Attributes {
        return Attributes.init(allocator);
    }

    /// Create execution context
    pub fn createExecutionContext(allocator: std.mem.Allocator) ExecutionContext {
        return ExecutionContext.init(allocator);
    }

    /// Helper to set tensor data from slice
    pub fn setTensorData(tensor: *Tensor, comptime T: type, data: []const T) !void {
        const tensor_data = tensor.getMutableData(T);
        if (tensor_data.len != data.len) {
            return FrameworkError.ShapeMismatch;
        }
        @memcpy(tensor_data, data);
    }

    /// Helper to get tensor data as slice
    pub fn getTensorData(tensor: *const Tensor, comptime T: type) []const T {
        return tensor.getData(T);
    }

    /// Calculate total elements in a shape
    pub fn calculateTotalElements(shape: []const usize) usize {
        return OperatorUtils.calculateTotalElements(shape);
    }

    /// Get element size for a data type
    pub fn getElementSize(dtype: Tensor.DataType) usize {
        return OperatorUtils.getElementSize(dtype);
    }

    /// Check if two shapes are equal
    pub fn shapesEqual(shape1: []const usize, shape2: []const usize) bool {
        return OperatorUtils.shapesEqual(shape1, shape2);
    }
};

/// Test utilities for framework testing
pub const testing = struct {
    /// Create a test tensor with random data
    pub fn createTestTensor(allocator: std.mem.Allocator, shape: []const usize, dtype: Tensor.DataType) !Tensor {
        var tensor = try Tensor.init(allocator, shape, dtype);
        
        // Fill with test data based on type
        switch (dtype) {
            .f32 => {
                const data = tensor.getMutableData(f32);
                for (data, 0..) |*value, i| {
                    value.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
                }
            },
            .i32 => {
                const data = tensor.getMutableData(i32);
                for (data, 0..) |*value, i| {
                    value.* = @as(i32, @intCast(i % 100));
                }
            },
            else => {
                // Fill with zeros for other types
                @memset(tensor.data, 0);
            },
        }
        
        return tensor;
    }

    /// Create test attributes
    pub fn createTestAttributes(allocator: std.mem.Allocator) !Attributes {
        var attrs = Attributes.init(allocator);
        try attrs.set("test_int", Attributes.AttributeValue{ .int = 42 });
        try attrs.set("test_float", Attributes.AttributeValue{ .float = 3.14 });
        try attrs.set("test_string", Attributes.AttributeValue{ .string = "test" });
        return attrs;
    }

    /// Verify tensor data matches expected values
    pub fn verifyTensorData(tensor: *const Tensor, comptime T: type, expected: []const T) !void {
        const actual = tensor.getData(T);
        if (actual.len != expected.len) {
            return FrameworkError.ShapeMismatch;
        }
        
        for (actual, expected) |a, e| {
            if (a != e) {
                return FrameworkError.ValidationFailed;
            }
        }
    }

    /// Compare two tensors for equality
    pub fn tensorsEqual(tensor1: *const Tensor, tensor2: *const Tensor) bool {
        if (tensor1.dtype != tensor2.dtype) return false;
        if (!utils.shapesEqual(tensor1.shape, tensor2.shape)) return false;
        
        return std.mem.eql(u8, tensor1.data, tensor2.data);
    }
};

// Framework tests
test "framework initialization" {
    const allocator = std.testing.allocator;
    
    const config = Framework.Config{
        .device = .cpu,
        .optimization_level = .basic,
        .enable_profiling = false,
    };
    
    var framework = try Framework.init(allocator, config);
    defer framework.deinit();
    
    // Test basic functionality
    const stats = framework.getStats();
    try std.testing.expect(stats.total_memory_used == 0);
}

test "tensor creation and manipulation" {
    const allocator = std.testing.allocator;
    
    const shape = [_]usize{ 2, 3 };
    var tensor = try utils.createTensor(allocator, &shape, .f32);
    defer tensor.deinit();
    
    // Test tensor properties
    try std.testing.expect(tensor.shape.len == 2);
    try std.testing.expect(tensor.shape[0] == 2);
    try std.testing.expect(tensor.shape[1] == 3);
    try std.testing.expect(tensor.dtype == .f32);
    
    // Test data access
    const data = tensor.getMutableData(f32);
    try std.testing.expect(data.len == 6);
    
    // Set and verify data
    const test_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try utils.setTensorData(&tensor, f32, &test_data);
    
    const retrieved_data = utils.getTensorData(&tensor, f32);
    try std.testing.expectEqualSlices(f32, &test_data, retrieved_data);
}

test "attributes creation and access" {
    const allocator = std.testing.allocator;
    
    var attrs = utils.createAttributes(allocator);
    defer attrs.deinit();
    
    // Test setting and getting attributes
    try attrs.set("int_attr", Attributes.AttributeValue{ .int = 42 });
    try attrs.set("float_attr", Attributes.AttributeValue{ .float = 3.14 });
    try attrs.set("string_attr", Attributes.AttributeValue{ .string = "test" });
    
    try std.testing.expect(attrs.getInt("int_attr", 0) == 42);
    try std.testing.expect(attrs.getFloat("float_attr", 0.0) == 3.14);
    try std.testing.expectEqualStrings(attrs.getString("string_attr", ""), "test");
    
    // Test default values
    try std.testing.expect(attrs.getInt("nonexistent", 100) == 100);
};

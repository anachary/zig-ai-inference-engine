const std = @import("std");
const framework = @import("framework");

// Re-export framework for convenience
pub const Framework = framework.Framework;
pub const Tensor = framework.Tensor;
pub const Attributes = framework.Attributes;
pub const ExecutionContext = framework.ExecutionContext;
pub const OperatorRegistry = framework.OperatorRegistry;
pub const OperatorInterface = framework.OperatorInterface;
pub const BaseOperator = framework.BaseOperator;

// Import all operator implementations
pub const operators = struct {
    // Arithmetic operators
    pub const arithmetic = struct {
        const add_impl = @import("operators/arithmetic/add.zig");
        const sub_impl = @import("operators/arithmetic/sub.zig");
        
        pub const Add = add_impl.Add;
        pub const Sub = sub_impl.Sub;
        pub const Mul = sub_impl.Mul;
        pub const Div = sub_impl.Div;
    };

    // Activation operators
    pub const activation = struct {
        const relu_impl = @import("operators/activation/relu.zig");
        
        pub const ReLU = relu_impl.ReLU;
        pub const Sigmoid = relu_impl.Sigmoid;
        pub const Tanh = relu_impl.Tanh;
        pub const GELU = relu_impl.GELU;
    };

    // Matrix operators
    pub const matrix = struct {
        const matmul_impl = @import("operators/matrix/matmul.zig");
        
        pub const MatMul = matmul_impl.MatMul;
        pub const Transpose = matmul_impl.Transpose;
    };

    // Missing/new operators
    pub const control_flow = struct {
        const control_flow_impl = @import("operators/missing/control_flow.zig");
        
        pub const If = control_flow_impl.If;
        pub const Where = control_flow_impl.Where;
        pub const Loop = control_flow_impl.Loop;
        pub const Scan = control_flow_impl.Scan;
    };

    // Operator registry with all built-in operators
    const registry_impl = @import("operators/registry.zig");
    pub const registry = registry_impl;
    pub const createBuiltinRegistry = registry_impl.createBuiltinRegistry;
    pub const registerBuiltinOperators = registry_impl.registerBuiltinOperators;
};

// Import model-specific implementations
pub const models = struct {
    // Transformer models
    pub const transformers = struct {
        const common_impl = @import("models/transformers/common.zig");
        const attention_impl = @import("models/transformers/attention.zig");
        
        pub const LayerNorm = common_impl.LayerNorm;
        pub const RMSNorm = common_impl.RMSNorm;
        pub const Embedding = common_impl.Embedding;
        pub const RotaryPositionalEmbedding = common_impl.RotaryPositionalEmbedding;
        
        pub const MultiHeadAttention = attention_impl.MultiHeadAttention;
        pub const FlashAttention = attention_impl.FlashAttention;
    };

    // Vision models (placeholder for future implementation)
    pub const vision = struct {
        // TODO: Implement CNN, ViT, etc.
    };

    // Audio models (placeholder for future implementation)
    pub const audio = struct {
        // TODO: Implement Whisper, Wav2Vec, etc.
    };
};

/// Version information
pub const VERSION = "0.1.0";
pub const VERSION_MAJOR = 0;
pub const VERSION_MINOR = 1;
pub const VERSION_PATCH = 0;

/// Complete AI platform with all implementations
pub const AIPlatform = struct {
    allocator: std.mem.Allocator,
    framework: Framework,
    
    const Self = @This();

    /// Platform configuration
    pub const Config = struct {
        framework_config: Framework.Config = .{},
        enable_all_operators: bool = true,
        enable_transformer_models: bool = true,
        enable_vision_models: bool = false,
        enable_audio_models: bool = false,
    };

    /// Initialize the complete AI platform
    pub fn init(allocator: std.mem.Allocator, config: Config) !Self {
        var framework_instance = try Framework.init(allocator, config.framework_config);

        // Register all operators if enabled
        if (config.enable_all_operators) {
            try operators.registerBuiltinOperators(&framework_instance.operator_registry);
            
            // Register transformer-specific operators
            if (config.enable_transformer_models) {
                try framework_instance.registerOperator(models.transformers.LayerNorm.getDefinition());
                try framework_instance.registerOperator(models.transformers.RMSNorm.getDefinition());
                try framework_instance.registerOperator(models.transformers.Embedding.getDefinition());
                try framework_instance.registerOperator(models.transformers.RotaryPositionalEmbedding.getDefinition());
                try framework_instance.registerOperator(models.transformers.MultiHeadAttention.getDefinition());
                try framework_instance.registerOperator(models.transformers.FlashAttention.getDefinition());
            }

            // Register control flow operators
            try framework_instance.registerOperator(operators.control_flow.If.getDefinition());
            try framework_instance.registerOperator(operators.control_flow.Where.getDefinition());
            try framework_instance.registerOperator(operators.control_flow.Loop.getDefinition());
            try framework_instance.registerOperator(operators.control_flow.Scan.getDefinition());
        }

        return Self{
            .allocator = allocator,
            .framework = framework_instance,
        };
    }

    /// Deinitialize the platform
    pub fn deinit(self: *Self) void {
        self.framework.deinit();
    }

    /// Get the underlying framework
    pub fn getFramework(self: *Self) *Framework {
        return &self.framework;
    }

    /// Create a computational graph
    pub fn createGraph(self: *Self) framework.Graph {
        return self.framework.createGraph();
    }

    /// Execute a computational graph
    pub fn executeGraph(self: *Self, graph: *framework.Graph) !void {
        try self.framework.executeGraph(graph);
    }

    /// List all available operators
    pub fn listOperators(self: *const Self) ![]framework.OperatorRegistry.OperatorInfo {
        return self.framework.listOperators();
    }

    /// Get platform statistics
    pub fn getStats(self: *const Self) PlatformStats {
        const framework_stats = self.framework.getStats();
        return PlatformStats{
            .framework_stats = framework_stats,
            .total_operators = framework_stats.registered_operators,
            .transformer_operators_enabled = true, // TODO: Track this properly
            .vision_operators_enabled = false,
            .audio_operators_enabled = false,
        };
    }

    /// Platform statistics
    pub const PlatformStats = struct {
        framework_stats: Framework.FrameworkStats,
        total_operators: u32,
        transformer_operators_enabled: bool,
        vision_operators_enabled: bool,
        audio_operators_enabled: bool,
    };

    /// Check operator support
    pub fn supportsOperator(self: *const Self, name: []const u8, version: ?[]const u8) bool {
        return self.framework.hasOperator(name, version);
    }

    /// Get operator categories
    pub fn getOperatorCategories(self: *const Self) []const operators.registry.OperatorCategory {
        _ = self;
        return &[_]operators.registry.OperatorCategory{
            .arithmetic,
            .activation,
            .matrix,
            .control_flow,
        };
    }

    /// Get operators by category
    pub fn getOperatorsByCategory(self: *const Self, category: operators.registry.OperatorCategory) []const operators.registry.OperatorInfo {
        _ = self;
        return operators.registry.getOperatorsByCategory(category);
    }
};

/// Utility functions for the complete platform
pub const utils = struct {
    /// Create a complete AI platform with default configuration
    pub fn createDefaultPlatform(allocator: std.mem.Allocator) !AIPlatform {
        const config = AIPlatform.Config{
            .framework_config = .{
                .device = .auto,
                .optimization_level = .basic,
                .enable_profiling = false,
            },
            .enable_all_operators = true,
            .enable_transformer_models = true,
        };
        
        return AIPlatform.init(allocator, config);
    }

    /// Create a platform optimized for transformers
    pub fn createTransformerPlatform(allocator: std.mem.Allocator) !AIPlatform {
        const config = AIPlatform.Config{
            .framework_config = .{
                .device = .auto,
                .optimization_level = .aggressive,
                .enable_profiling = true,
            },
            .enable_all_operators = true,
            .enable_transformer_models = true,
            .enable_vision_models = false,
            .enable_audio_models = false,
        };
        
        return AIPlatform.init(allocator, config);
    }

    /// Create a minimal platform for edge deployment
    pub fn createEdgePlatform(allocator: std.mem.Allocator) !AIPlatform {
        const config = AIPlatform.Config{
            .framework_config = .{
                .device = .cpu,
                .optimization_level = .basic,
                .enable_profiling = false,
                .enable_memory_tracking = true,
                .max_memory_mb = 128, // Limit memory for edge devices
            },
            .enable_all_operators = false, // Only register operators as needed
            .enable_transformer_models = false,
            .enable_vision_models = false,
            .enable_audio_models = false,
        };
        
        return AIPlatform.init(allocator, config);
    }

    /// Register a custom operator with the platform
    pub fn registerCustomOperator(platform: *AIPlatform, definition: framework.OperatorInterface.Definition) !void {
        try platform.framework.registerOperator(definition);
    }

    /// Override an existing operator
    pub fn overrideOperator(platform: *AIPlatform, definition: framework.OperatorInterface.Definition) !void {
        try platform.framework.overrideOperator(definition);
    }
};

/// Testing utilities for the complete platform
pub const testing = struct {
    /// Create a test platform with minimal configuration
    pub fn createTestPlatform(allocator: std.mem.Allocator) !AIPlatform {
        const config = AIPlatform.Config{
            .framework_config = .{
                .device = .cpu,
                .optimization_level = .none,
                .enable_profiling = false,
            },
            .enable_all_operators = true,
            .enable_transformer_models = true,
        };
        
        return AIPlatform.init(allocator, config);
    }

    /// Test operator execution
    pub fn testOperatorExecution(
        platform: *AIPlatform,
        operator_name: []const u8,
        inputs: []const Tensor,
        expected_outputs: []const Tensor,
    ) !void {
        // Create a simple graph with the operator
        var graph = platform.createGraph();
        defer graph.deinit();

        // TODO: Build graph and execute
        // This would require graph building utilities
        _ = operator_name;
        _ = inputs;
        _ = expected_outputs;
        
        std.log.info("Test operator execution not yet implemented");
    }

    /// Benchmark operator performance
    pub fn benchmarkOperator(
        platform: *AIPlatform,
        operator_name: []const u8,
        inputs: []const Tensor,
        iterations: u32,
    ) !f64 {
        _ = platform;
        _ = operator_name;
        _ = inputs;
        _ = iterations;
        
        // TODO: Implement benchmarking
        std.log.info("Operator benchmarking not yet implemented");
        return 0.0;
    }
};

// Platform tests
test "AI platform initialization" {
    const allocator = std.testing.allocator;
    
    var platform = try utils.createDefaultPlatform(allocator);
    defer platform.deinit();
    
    // Test basic functionality
    const stats = platform.getStats();
    try std.testing.expect(stats.total_operators > 0);
    try std.testing.expect(stats.transformer_operators_enabled);
}

test "operator registration and lookup" {
    const allocator = std.testing.allocator;
    
    var platform = try utils.createDefaultPlatform(allocator);
    defer platform.deinit();
    
    // Test that operators are registered
    try std.testing.expect(platform.supportsOperator("Add", null));
    try std.testing.expect(platform.supportsOperator("Relu", null));
    try std.testing.expect(platform.supportsOperator("MatMul", null));
    try std.testing.expect(platform.supportsOperator("LayerNormalization", null));
    try std.testing.expect(platform.supportsOperator("MultiHeadAttention", null));
}

test "operator categories" {
    const allocator = std.testing.allocator;
    
    var platform = try utils.createDefaultPlatform(allocator);
    defer platform.deinit();
    
    const categories = platform.getOperatorCategories();
    try std.testing.expect(categories.len > 0);
    
    const arithmetic_ops = platform.getOperatorsByCategory(.arithmetic);
    try std.testing.expect(arithmetic_ops.len > 0);
}

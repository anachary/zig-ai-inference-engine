const std = @import("std");

// Core model types
pub const Model = @import("formats/model.zig").Model;
pub const ModelMetadata = @import("formats/model.zig").ModelMetadata;
pub const ModelFormat = @import("formats/model.zig").ModelFormat;
pub const IOSpec = @import("formats/model.zig").IOSpec;
pub const ComputationGraph = @import("formats/model.zig").ComputationGraph;
pub const GraphNode = @import("formats/model.zig").GraphNode;
pub const ModelStats = @import("formats/model.zig").ModelStats;

// ONNX specific types and parser
pub const onnx = @import("formats/onnx/parser.zig");
pub const ONNXParser = onnx.ONNXParser;
pub const ONNXError = onnx.ONNXError;
pub const ParserConfig = onnx.ParserConfig;

// ONNX types
pub const ONNXDataType = @import("formats/onnx/types.zig").ONNXDataType;
pub const ONNXModel = @import("formats/onnx/types.zig").ONNXModel;
pub const ONNXGraph = @import("formats/onnx/types.zig").ONNXGraph;
pub const ONNXNode = @import("formats/onnx/types.zig").ONNXNode;
pub const ONNXTensor = @import("formats/onnx/types.zig").ONNXTensor;

// Protobuf utilities
pub const protobuf = @import("formats/onnx/protobuf.zig");

/// Main parser interface that supports multiple formats
pub const Parser = struct {
    allocator: std.mem.Allocator,
    onnx_parser: ONNXParser,

    const Self = @This();

    /// Initialize parser with default configuration
    pub fn init(allocator: std.mem.Allocator) Self {
        const config = ParserConfig{};
        return Self{
            .allocator = allocator,
            .onnx_parser = ONNXParser.init(allocator, config),
        };
    }

    /// Initialize parser with custom configuration
    pub fn initWithConfig(allocator: std.mem.Allocator, config: ParserConfig) Self {
        return Self{
            .allocator = allocator,
            .onnx_parser = ONNXParser.init(allocator, config),
        };
    }

    /// Parse model from file (auto-detects format)
    pub fn parseFile(self: *Self, path: []const u8) !Model {
        const format = ModelFormat.fromPath(path);
        
        std.log.info("🔍 Detected format: {s} for file: {s}", .{ format.toString(), path });
        
        return switch (format) {
            .onnx => self.onnx_parser.parseFile(path),
            .onnx_text => error.UnsupportedFormat, // TODO: Implement text format
            .tensorflow_lite => error.UnsupportedFormat, // TODO: Implement TFLite
            .pytorch_jit => error.UnsupportedFormat, // TODO: Implement PyTorch
            else => error.UnsupportedFormat,
        };
    }

    /// Parse model from byte array with explicit format
    pub fn parseBytes(self: *Self, data: []const u8, format: ModelFormat) !Model {
        std.log.info("🔍 Parsing {} bytes as {s} format", .{ data.len, format.toString() });
        
        return switch (format) {
            .onnx => self.onnx_parser.parseBytes(data),
            .onnx_text => error.UnsupportedFormat,
            .tensorflow_lite => error.UnsupportedFormat,
            .pytorch_jit => error.UnsupportedFormat,
            else => error.UnsupportedFormat,
        };
    }

    /// Parse ONNX model specifically
    pub fn parseONNX(self: *Self, path: []const u8) !Model {
        return self.onnx_parser.parseFile(path);
    }

    /// Parse ONNX model from bytes
    pub fn parseONNXBytes(self: *Self, data: []const u8) !Model {
        return self.onnx_parser.parseBytes(data);
    }
};

/// Validation utilities
pub const validation = struct {
    /// Validate model format and structure
    pub fn validateModel(model: *const Model) !void {
        try model.validate();
        
        // Additional format-specific validations
        const metadata = model.getMetadata();
        
        switch (metadata.format) {
            .onnx => try validateONNXModel(model),
            else => {}, // No additional validation for other formats yet
        }
    }

    /// ONNX-specific validation
    fn validateONNXModel(model: *const Model) !void {
        const metadata = model.getMetadata();
        
        // Check IR version
        if (metadata.ir_version < 3) {
            std.log.warn("Old ONNX IR version: {}, consider upgrading", .{metadata.ir_version});
        }
        
        // Check opset version
        if (metadata.opset_version < 11) {
            std.log.warn("Old ONNX opset version: {}, some operators may not be supported", .{metadata.opset_version});
        }
        
        // Validate node types
        for (model.graph.nodes.items) |node| {
            if (node.op_type.len == 0) {
                std.log.err("Node {s} has empty op_type", .{node.name});
                return error.InvalidNode;
            }
        }
    }
};

/// Model optimization utilities
pub const optimization = struct {
    /// Optimization configuration
    pub const OptimizationConfig = struct {
        remove_unused_nodes: bool = true,
        constant_folding: bool = true,
        dead_code_elimination: bool = true,
        operator_fusion: bool = false, // Advanced optimization
    };

    /// Optimize model graph
    pub fn optimizeModel(model: *Model, config: OptimizationConfig) !void {
        std.log.info("🔧 Starting model optimization...", .{});
        
        if (config.remove_unused_nodes) {
            try removeUnusedNodes(model);
        }
        
        if (config.dead_code_elimination) {
            try eliminateDeadCode(model);
        }
        
        std.log.info("✅ Model optimization completed", .{});
    }

    /// Remove nodes that don't contribute to outputs
    fn removeUnusedNodes(model: *Model) !void {
        // Mark nodes that contribute to outputs
        var used_nodes = std.AutoHashMap(usize, void).init(model.allocator);
        defer used_nodes.deinit();
        
        var value_producers = std.StringHashMap(usize).init(model.allocator);
        defer value_producers.deinit();
        
        // Build value producer map
        for (model.graph.nodes.items, 0..) |node, i| {
            for (node.outputs) |output| {
                try value_producers.put(output, i);
            }
        }
        
        // Mark nodes needed for outputs
        var to_visit = std.ArrayList([]const u8).init(model.allocator);
        defer to_visit.deinit();
        
        for (model.graph.outputs.items) |output| {
            try to_visit.append(output.name);
        }
        
        while (to_visit.items.len > 0) {
            const value_name = to_visit.pop();
            
            if (value_producers.get(value_name)) |producer_idx| {
                if (!used_nodes.contains(producer_idx)) {
                    try used_nodes.put(producer_idx, {});
                    
                    // Add inputs of this node to visit list
                    const node = model.graph.nodes.items[producer_idx];
                    for (node.inputs) |input| {
                        try to_visit.append(input);
                    }
                }
            }
        }
        
        // Remove unused nodes
        var i: usize = 0;
        while (i < model.graph.nodes.items.len) {
            if (!used_nodes.contains(i)) {
                var removed_node = model.graph.nodes.orderedRemove(i);
                removed_node.deinit(model.allocator);
                std.log.info("Removed unused node: {s}", .{removed_node.name});
            } else {
                i += 1;
            }
        }
    }

    /// Eliminate dead code (nodes with no outputs)
    fn eliminateDeadCode(model: *Model) !void {
        var i: usize = 0;
        while (i < model.graph.nodes.items.len) {
            const node = model.graph.nodes.items[i];
            if (node.outputs.len == 0) {
                var removed_node = model.graph.nodes.orderedRemove(i);
                removed_node.deinit(model.allocator);
                std.log.info("Eliminated dead code node: {s}", .{removed_node.name});
            } else {
                i += 1;
            }
        }
    }
};

/// Utility functions
pub const utils = struct {
    /// Print model information
    pub fn printModelInfo(model: *const Model) void {
        const metadata = model.getMetadata();
        const stats = model.getStats();
        
        std.log.info("=== Model Information ===");
        std.log.info("Name: {s}", .{metadata.name});
        std.log.info("Version: {s}", .{metadata.version});
        std.log.info("Format: {s}", .{metadata.format.toString()});
        std.log.info("Producer: {s} v{s}", .{ metadata.producer_name, metadata.producer_version });
        std.log.info("IR Version: {}", .{metadata.ir_version});
        std.log.info("Opset Version: {}", .{metadata.opset_version});
        
        stats.print();
    }

    /// Export model to different format (placeholder)
    pub fn exportModel(model: *const Model, path: []const u8, format: ModelFormat) !void {
        _ = model;
        _ = path;
        _ = format;
        return error.NotImplemented;
    }

    /// Compare two models for structural similarity
    pub fn compareModels(model1: *const Model, model2: *const Model) ModelComparison {
        const stats1 = model1.getStats();
        const stats2 = model2.getStats();
        
        return ModelComparison{
            .nodes_match = stats1.node_count == stats2.node_count,
            .inputs_match = stats1.input_count == stats2.input_count,
            .outputs_match = stats1.output_count == stats2.output_count,
            .similarity_score = calculateSimilarity(&stats1, &stats2),
        };
    }

    const ModelComparison = struct {
        nodes_match: bool,
        inputs_match: bool,
        outputs_match: bool,
        similarity_score: f32,
    };

    fn calculateSimilarity(stats1: *const ModelStats, stats2: *const ModelStats) f32 {
        var score: f32 = 0.0;
        var total_checks: f32 = 4.0;
        
        if (stats1.node_count == stats2.node_count) score += 1.0;
        if (stats1.input_count == stats2.input_count) score += 1.0;
        if (stats1.output_count == stats2.output_count) score += 1.0;
        if (stats1.parameter_count == stats2.parameter_count) score += 1.0;
        
        return score / total_checks;
    }
};

/// Version information
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;
    pub const string = "0.1.0";
    
    pub const supported_onnx_versions = [_]i64{ 11, 12, 13, 14, 15, 16, 17, 18 };
    pub const default_onnx_version = 17;
};

// Tests
test "parser initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var parser = Parser.init(allocator);
    _ = parser;
}

test "model format detection" {
    const testing = std.testing;
    
    try testing.expect(ModelFormat.fromPath("model.onnx") == .onnx);
    try testing.expect(ModelFormat.fromPath("model.onnx.txt") == .onnx_text);
    try testing.expect(ModelFormat.fromPath("model.tflite") == .tensorflow_lite);
    try testing.expect(ModelFormat.fromPath("model.pt") == .pytorch_jit);
    try testing.expect(ModelFormat.fromPath("model.unknown") == .custom);
}

test "model metadata" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var metadata = try ModelMetadata.init(allocator, "test_model", "1.0");
    defer metadata.deinit(allocator);
    
    try testing.expect(std.mem.eql(u8, metadata.name, "test_model"));
    try testing.expect(std.mem.eql(u8, metadata.version, "1.0"));
    try testing.expect(metadata.format == .onnx);
}

test "computation graph" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var graph = ComputationGraph.init(allocator);
    defer graph.deinit();
    
    // Add a simple node
    var node = try GraphNode.init(allocator, "test_node", "Add");
    defer node.deinit(allocator);
    
    try graph.addNode(node);
    try testing.expect(graph.nodes.items.len == 1);
}

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Model format enumeration
pub const ModelFormat = enum {
    onnx,
    onnx_text,
    tensorflow_lite,
    pytorch_jit,
    internal,
    custom,

    pub fn fromPath(path: []const u8) ModelFormat {
        std.log.info("🔍 Detecting format for: {s}", .{path});

        if (std.mem.endsWith(u8, path, ".onnx")) {
            std.log.info("✅ Detected ONNX format from extension", .{});
            return .onnx;
        }
        if (std.mem.endsWith(u8, path, ".onnx.txt")) {
            std.log.info("✅ Detected ONNX text format from extension", .{});
            return .onnx_text;
        }
        if (std.mem.endsWith(u8, path, ".tflite")) {
            std.log.info("✅ Detected TensorFlow Lite format from extension", .{});
            return .tensorflow_lite;
        }
        if (std.mem.endsWith(u8, path, ".pt") or std.mem.endsWith(u8, path, ".pth")) {
            std.log.info("✅ Detected PyTorch JIT format from extension", .{});
            return .pytorch_jit;
        }

        std.log.warn("⚠️  Unknown format for {s}, defaulting to custom", .{path});
        return .custom;
    }

    /// Enhanced format detection with content analysis
    pub fn detectFormat(file_path: []const u8, data: []const u8) ModelFormat {
        std.log.info("🔍 Detecting format for: {s}", .{file_path});
        std.log.info("📊 File size: {d} bytes", .{data.len});

        // First try extension-based detection (most reliable)
        const ext_format = fromPath(file_path);
        if (ext_format != .custom) {
            return ext_format;
        }

        // Enhanced content-based detection for ONNX
        if (data.len >= 8) {
            // ONNX files are protobuf format, check for common protobuf patterns
            // Field 1 (ir_version) is usually first in ONNX ModelProto
            if ((data[0] == 0x08 and data[1] >= 0x01 and data[1] <= 0x10) or // varint field 1
                (data[0] == 0x0A) or // length-delimited field 1
                (data[0] == 0x12)) // length-delimited field 2 (graph)
            {
                std.log.info("✅ Detected ONNX format from protobuf content", .{});
                return .onnx;
            }

            // Check for ONNX magic patterns in first few bytes
            for (data[0..@min(data.len, 16)], 0..) |byte, i| {
                _ = i;
                // Look for typical ONNX protobuf field numbers (1-15 are common)
                if (byte >= 0x08 and byte <= 0x7A and (byte & 0x07) <= 0x05) {
                    std.log.info("✅ Detected potential ONNX format from protobuf patterns");
                    return .onnx;
                }
            }
        }

        std.log.warn("⚠️  Unknown format, defaulting to custom");
        return .custom;
    }

    pub fn toString(self: ModelFormat) []const u8 {
        return switch (self) {
            .onnx => "ONNX",
            .onnx_text => "ONNX Text",
            .tensorflow_lite => "TensorFlow Lite",
            .pytorch_jit => "PyTorch JIT",
            .internal => "Internal",
            .custom => "Custom",
        };
    }
};

/// Model metadata information
pub const ModelMetadata = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    author: []const u8,
    format: ModelFormat,
    ir_version: i64,
    opset_version: i64,
    producer_name: []const u8,
    producer_version: []const u8,
    domain: []const u8,
    model_size_bytes: usize,
    parameter_count: usize,
    input_count: usize,
    output_count: usize,

    pub fn init(allocator: Allocator, name: []const u8, version: []const u8) !ModelMetadata {
        return ModelMetadata{
            .name = try allocator.dupe(u8, name),
            .version = try allocator.dupe(u8, version),
            .description = try allocator.dupe(u8, ""),
            .author = try allocator.dupe(u8, ""),
            .format = .onnx,
            .ir_version = 7,
            .opset_version = 17,
            .producer_name = try allocator.dupe(u8, "zig-onnx-parser"),
            .producer_version = try allocator.dupe(u8, "0.1.0"),
            .domain = try allocator.dupe(u8, ""),
            .model_size_bytes = 0,
            .parameter_count = 0,
            .input_count = 0,
            .output_count = 0,
        };
    }

    pub fn deinit(self: *ModelMetadata, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.version);
        allocator.free(self.description);
        allocator.free(self.author);
        allocator.free(self.producer_name);
        allocator.free(self.producer_version);
        allocator.free(self.domain);
    }
};

/// Input/Output specification
pub const IOSpec = struct {
    name: []const u8,
    shape: []const i64, // -1 for dynamic dimensions
    dtype: DataType,
    description: []const u8,

    pub const DataType = enum {
        f32,
        f16,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64,
        bool,
        string,
    };

    pub fn init(allocator: Allocator, name: []const u8, shape: []const i64, dtype: DataType) !IOSpec {
        return IOSpec{
            .name = try allocator.dupe(u8, name),
            .shape = try allocator.dupe(i64, shape),
            .dtype = dtype,
            .description = try allocator.dupe(u8, ""),
        };
    }

    pub fn deinit(self: *IOSpec, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.shape);
        allocator.free(self.description);
    }

    /// Check if shape is fully defined (no dynamic dimensions)
    pub fn isFullyDefined(self: *const IOSpec) bool {
        for (self.shape) |dim| {
            if (dim < 0) return false;
        }
        return true;
    }

    /// Get total number of elements (returns null if dynamic)
    pub fn numel(self: *const IOSpec) ?usize {
        if (!self.isFullyDefined()) return null;

        var total: usize = 1;
        for (self.shape) |dim| {
            total *= @as(usize, @intCast(dim));
        }
        return total;
    }
};

/// Graph node representation
pub const GraphNode = struct {
    name: []const u8,
    op_type: []const u8,
    inputs: [][]const u8,
    outputs: [][]const u8,
    attributes: std.StringHashMap(AttributeValue),

    pub const AttributeValue = union(enum) {
        int: i64,
        float: f64,
        string: []const u8,
        ints: []i64,
        floats: []f64,
        strings: [][]const u8,
    };

    pub fn init(allocator: Allocator, name: []const u8, op_type: []const u8) !GraphNode {
        return GraphNode{
            .name = try allocator.dupe(u8, name),
            .op_type = try allocator.dupe(u8, op_type),
            .inputs = &[_][]const u8{},
            .outputs = &[_][]const u8{},
            .attributes = std.StringHashMap(AttributeValue).init(allocator),
        };
    }

    pub fn deinit(self: *GraphNode, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.op_type);

        for (self.inputs) |input| allocator.free(input);
        allocator.free(self.inputs);

        for (self.outputs) |output| allocator.free(output);
        allocator.free(self.outputs);

        var attr_iter = self.attributes.iterator();
        while (attr_iter.next()) |entry| {
            switch (entry.value_ptr.*) {
                .string => |str| allocator.free(str),
                .ints => |ints| allocator.free(ints),
                .floats => |floats| allocator.free(floats),
                .strings => |strings| {
                    for (strings) |str| allocator.free(str);
                    allocator.free(strings);
                },
                else => {},
            }
            allocator.free(entry.key_ptr.*);
        }
        self.attributes.deinit();
    }
};

/// Computation graph
pub const ComputationGraph = struct {
    nodes: std.ArrayList(GraphNode),
    inputs: std.ArrayList(IOSpec),
    outputs: std.ArrayList(IOSpec),
    allocator: Allocator,

    pub fn init(allocator: Allocator) ComputationGraph {
        return ComputationGraph{
            .nodes = std.ArrayList(GraphNode).init(allocator),
            .inputs = std.ArrayList(IOSpec).init(allocator),
            .outputs = std.ArrayList(IOSpec).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ComputationGraph) void {
        for (self.nodes.items) |*node| node.deinit(self.allocator);
        self.nodes.deinit();

        for (self.inputs.items) |*input| input.deinit(self.allocator);
        self.inputs.deinit();

        for (self.outputs.items) |*output| output.deinit(self.allocator);
        self.outputs.deinit();
    }

    pub fn addNode(self: *ComputationGraph, node: GraphNode) !void {
        try self.nodes.append(node);
    }

    pub fn addInput(self: *ComputationGraph, input: IOSpec) !void {
        try self.inputs.append(input);
    }

    pub fn addOutput(self: *ComputationGraph, output: IOSpec) !void {
        try self.outputs.append(output);
    }

    /// Validate graph connectivity and structure
    pub fn validate(self: *const ComputationGraph) !void {
        // Check that all node inputs are either graph inputs or outputs of other nodes
        var available_values = std.StringHashMap(void).init(self.allocator);
        defer available_values.deinit();

        // Add graph inputs
        for (self.inputs.items) |input| {
            try available_values.put(input.name, {});
        }

        // Check each node
        for (self.nodes.items) |node| {
            // Check inputs are available
            for (node.inputs) |input_name| {
                if (!available_values.contains(input_name)) {
                    std.log.err("Node {s}: input {s} not available", .{ node.name, input_name });
                    return error.MissingInput;
                }
            }

            // Add outputs to available values
            for (node.outputs) |output_name| {
                try available_values.put(output_name, {});
            }
        }

        // Check that all graph outputs are available
        for (self.outputs.items) |output| {
            if (!available_values.contains(output.name)) {
                std.log.err("Graph output {s} not available", .{output.name});
                return error.MissingOutput;
            }
        }
    }

    /// Get topological ordering of nodes
    pub fn getTopologicalOrder(self: *const ComputationGraph, allocator: Allocator) ![]usize {
        const node_count = self.nodes.items.len;
        var in_degree = try allocator.alloc(usize, node_count);
        defer allocator.free(in_degree);

        var adjacency = try allocator.alloc(std.ArrayList(usize), node_count);
        defer {
            for (adjacency) |*list| list.deinit();
            allocator.free(adjacency);
        }

        // Initialize
        for (0..node_count) |i| {
            in_degree[i] = 0;
            adjacency[i] = std.ArrayList(usize).init(allocator);
        }

        // Build adjacency list and calculate in-degrees
        var name_to_index = std.StringHashMap(usize).init(allocator);
        defer name_to_index.deinit();

        for (self.nodes.items, 0..) |node, i| {
            for (node.outputs) |output| {
                try name_to_index.put(output, i);
            }
        }

        for (self.nodes.items, 0..) |node, i| {
            for (node.inputs) |input| {
                if (name_to_index.get(input)) |producer_idx| {
                    try adjacency[producer_idx].append(i);
                    in_degree[i] += 1;
                }
            }
        }

        // Kahn's algorithm
        var queue = std.ArrayList(usize).init(allocator);
        defer queue.deinit();
        var result = std.ArrayList(usize).init(allocator);
        defer result.deinit();

        // Add nodes with no incoming edges
        for (0..node_count) |i| {
            if (in_degree[i] == 0) {
                try queue.append(i);
            }
        }

        while (queue.items.len > 0) {
            const current = queue.orderedRemove(0);
            try result.append(current);

            for (adjacency[current].items) |neighbor| {
                in_degree[neighbor] -= 1;
                if (in_degree[neighbor] == 0) {
                    try queue.append(neighbor);
                }
            }
        }

        if (result.items.len != node_count) {
            return error.CyclicGraph;
        }

        return result.toOwnedSlice();
    }
};

/// Main model representation
pub const Model = struct {
    metadata: ModelMetadata,
    graph: ComputationGraph,
    allocator: Allocator,

    pub fn init(allocator: Allocator, metadata: ModelMetadata) Model {
        return Model{
            .metadata = metadata,
            .graph = ComputationGraph.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Model) void {
        self.metadata.deinit(self.allocator);
        self.graph.deinit();
    }

    /// Get model metadata
    pub fn getMetadata(self: *const Model) *const ModelMetadata {
        return &self.metadata;
    }

    /// Get input specifications
    pub fn getInputs(self: *const Model) []const IOSpec {
        return self.graph.inputs.items;
    }

    /// Get output specifications
    pub fn getOutputs(self: *const Model) []const IOSpec {
        return self.graph.outputs.items;
    }

    /// Validate the entire model
    pub fn validate(self: *const Model) !void {
        try self.graph.validate();

        // Additional model-level validations
        if (self.graph.inputs.items.len == 0) {
            return error.NoInputs;
        }

        if (self.graph.outputs.items.len == 0) {
            return error.NoOutputs;
        }
    }

    /// Get model statistics
    pub fn getStats(self: *const Model) ModelStats {
        var op_counts = std.StringHashMap(usize).init(self.allocator);
        defer op_counts.deinit();

        for (self.graph.nodes.items) |node| {
            const count = op_counts.get(node.op_type) orelse 0;
            op_counts.put(node.op_type, count + 1) catch {};
        }

        return ModelStats{
            .node_count = self.graph.nodes.items.len,
            .input_count = self.graph.inputs.items.len,
            .output_count = self.graph.outputs.items.len,
            .parameter_count = self.metadata.parameter_count,
            .model_size_bytes = self.metadata.model_size_bytes,
        };
    }
};

/// Model statistics
pub const ModelStats = struct {
    node_count: usize,
    input_count: usize,
    output_count: usize,
    parameter_count: usize,
    model_size_bytes: usize,

    pub fn print(self: *const ModelStats) void {
        std.log.info("=== Model Statistics ===");
        std.log.info("Nodes: {}", .{self.node_count});
        std.log.info("Inputs: {}", .{self.input_count});
        std.log.info("Outputs: {}", .{self.output_count});
        std.log.info("Parameters: {}", .{self.parameter_count});
        std.log.info("Size: {d:.1} MB", .{@as(f64, @floatFromInt(self.model_size_bytes)) / (1024.0 * 1024.0)});
    }
};

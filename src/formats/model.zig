const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");

pub const ModelError = error{
    InvalidFormat,
    UnsupportedVersion,
    CorruptedData,
    MissingWeights,
    InvalidGraph,
    OutOfMemory,
};

pub const ModelFormat = enum {
    onnx,
    tensorflow_lite,
    pytorch_jit,
    custom_binary,

    pub fn fromPath(path: []const u8) ModelFormat {
        if (std.mem.endsWith(u8, path, ".onnx")) return .onnx;
        if (std.mem.endsWith(u8, path, ".tflite")) return .tensorflow_lite;
        if (std.mem.endsWith(u8, path, ".pt") or std.mem.endsWith(u8, path, ".pth")) return .pytorch_jit;
        return .custom_binary;
    }
};

pub const TensorSpec = struct {
    name: []const u8,
    shape: []i32, // -1 for dynamic dimensions
    dtype: tensor.DataType,

    pub fn init(allocator: Allocator, name: []const u8, shape: []const i32, dtype: tensor.DataType) !TensorSpec {
        return TensorSpec{
            .name = try allocator.dupe(u8, name),
            .shape = try allocator.dupe(i32, shape),
            .dtype = dtype,
        };
    }

    pub fn deinit(self: *TensorSpec, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.shape);
    }

    pub fn isCompatible(self: *const TensorSpec, t: *const tensor.Tensor) bool {
        // Check data type
        if (self.dtype != t.dtype) return false;

        // Check shape compatibility (allowing dynamic dimensions)
        if (self.shape.len != t.shape.len) return false;

        for (self.shape, t.shape) |spec_dim, tensor_dim| {
            if (spec_dim != -1 and spec_dim != @as(i32, @intCast(tensor_dim))) {
                return false;
            }
        }

        return true;
    }
};

pub const ModelMetadata = struct {
    name: []const u8,
    version: []const u8,
    description: ?[]const u8,
    author: ?[]const u8,
    license: ?[]const u8,
    format: ModelFormat,

    pub fn init(allocator: Allocator, name: []const u8, version: []const u8) !ModelMetadata {
        return ModelMetadata{
            .name = try allocator.dupe(u8, name),
            .version = try allocator.dupe(u8, version),
            .description = null,
            .author = null,
            .license = null,
            .format = .custom_binary,
        };
    }

    pub fn deinit(self: *ModelMetadata, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.version);
        if (self.description) |desc| allocator.free(desc);
        if (self.author) |author| allocator.free(author);
        if (self.license) |license| allocator.free(license);
    }
};

pub const WeightMap = struct {
    weights: std.StringHashMap(tensor.Tensor),
    allocator: Allocator,

    pub fn init(allocator: Allocator) WeightMap {
        return WeightMap{
            .weights = std.StringHashMap(tensor.Tensor).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *WeightMap) void {
        var iterator = self.weights.iterator();
        while (iterator.next()) |entry| {
            var weight_tensor = entry.value_ptr;
            weight_tensor.deinit();
        }
        self.weights.deinit();
    }

    pub fn addWeight(self: *WeightMap, name: []const u8, weight: tensor.Tensor) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        try self.weights.put(name_copy, weight);
    }

    pub fn getWeight(self: *const WeightMap, name: []const u8) ?*const tensor.Tensor {
        return self.weights.getPtr(name);
    }

    pub fn hasWeight(self: *const WeightMap, name: []const u8) bool {
        return self.weights.contains(name);
    }

    pub fn getWeightCount(self: *const WeightMap) u32 {
        return @intCast(self.weights.count());
    }
};

pub const ComputationGraph = struct {
    nodes: std.ArrayList(GraphNode),
    edges: std.ArrayList(GraphEdge),
    inputs: std.ArrayList(TensorSpec),
    outputs: std.ArrayList(TensorSpec),
    allocator: Allocator,

    pub fn init(allocator: Allocator) ComputationGraph {
        return ComputationGraph{
            .nodes = std.ArrayList(GraphNode).init(allocator),
            .edges = std.ArrayList(GraphEdge).init(allocator),
            .inputs = std.ArrayList(TensorSpec).init(allocator),
            .outputs = std.ArrayList(TensorSpec).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ComputationGraph) void {
        for (self.nodes.items) |*node| {
            node.deinit(self.allocator);
        }
        self.nodes.deinit();

        for (self.edges.items) |*edge| {
            edge.deinit(self.allocator);
        }
        self.edges.deinit();

        for (self.inputs.items) |*input| {
            input.deinit(self.allocator);
        }
        self.inputs.deinit();

        for (self.outputs.items) |*output| {
            output.deinit(self.allocator);
        }
        self.outputs.deinit();
    }

    pub fn addNode(self: *ComputationGraph, node: GraphNode) !void {
        try self.nodes.append(node);
    }

    pub fn addEdge(self: *ComputationGraph, edge: GraphEdge) !void {
        try self.edges.append(edge);
    }

    pub fn addInput(self: *ComputationGraph, input_spec: TensorSpec) !void {
        try self.inputs.append(input_spec);
    }

    pub fn addOutput(self: *ComputationGraph, output_spec: TensorSpec) !void {
        try self.outputs.append(output_spec);
    }

    pub fn validate(self: *const ComputationGraph) !void {
        // Basic validation
        if (self.inputs.items.len == 0) return ModelError.InvalidGraph;
        if (self.outputs.items.len == 0) return ModelError.InvalidGraph;
        if (self.nodes.items.len == 0) return ModelError.InvalidGraph;

        // TODO: Add more sophisticated graph validation
        // - Check for cycles
        // - Verify all edges connect valid nodes
        // - Ensure all inputs/outputs are connected
    }
};

pub const GraphNode = struct {
    id: []const u8,
    op_type: []const u8,
    inputs: [][]const u8,
    outputs: [][]const u8,
    attributes: std.StringHashMap(AttributeValue),

    pub fn init(allocator: Allocator, id: []const u8, op_type: []const u8) !GraphNode {
        return GraphNode{
            .id = try allocator.dupe(u8, id),
            .op_type = try allocator.dupe(u8, op_type),
            .inputs = &[_][]const u8{},
            .outputs = &[_][]const u8{},
            .attributes = std.StringHashMap(AttributeValue).init(allocator),
        };
    }

    pub fn deinit(self: *GraphNode, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.op_type);

        for (self.inputs) |input| {
            allocator.free(input);
        }
        allocator.free(self.inputs);

        for (self.outputs) |output| {
            allocator.free(output);
        }
        allocator.free(self.outputs);

        var attr_iter = self.attributes.iterator();
        while (attr_iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(allocator);
        }
        self.attributes.deinit();
    }
};

pub const GraphEdge = struct {
    from_node: []const u8,
    to_node: []const u8,
    tensor_name: []const u8,

    pub fn init(allocator: Allocator, from: []const u8, to: []const u8, tensor_name: []const u8) !GraphEdge {
        return GraphEdge{
            .from_node = try allocator.dupe(u8, from),
            .to_node = try allocator.dupe(u8, to),
            .tensor_name = try allocator.dupe(u8, tensor_name),
        };
    }

    pub fn deinit(self: *GraphEdge, allocator: Allocator) void {
        allocator.free(self.from_node);
        allocator.free(self.to_node);
        allocator.free(self.tensor_name);
    }
};

pub const AttributeValue = union(enum) {
    int: i64,
    float: f64,
    string: []const u8,
    tensor: tensor.Tensor,
    ints: []i64,
    floats: []f64,
    strings: [][]const u8,

    pub fn deinit(self: *AttributeValue, allocator: Allocator) void {
        switch (self.*) {
            .string => |s| allocator.free(s),
            .tensor => |*t| t.deinit(),
            .ints => |ints| allocator.free(ints),
            .floats => |floats| allocator.free(floats),
            .strings => |strings| {
                for (strings) |s| allocator.free(s);
                allocator.free(strings);
            },
            else => {},
        }
    }
};

pub const Model = struct {
    graph: ComputationGraph,
    weights: WeightMap,
    metadata: ModelMetadata,
    allocator: Allocator,

    pub fn init(allocator: Allocator, metadata: ModelMetadata) Model {
        return Model{
            .graph = ComputationGraph.init(allocator),
            .weights = WeightMap.init(allocator),
            .metadata = metadata,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
        self.weights.deinit();
        // Note: metadata is owned by the Model and will be freed here
        self.metadata.deinit(self.allocator);
    }

    pub fn load(allocator: Allocator, path: []const u8) !Model {
        const format = ModelFormat.fromPath(path);

        return switch (format) {
            .onnx => loadONNX(allocator, path),
            .tensorflow_lite => ModelError.UnsupportedVersion, // TODO: Implement
            .pytorch_jit => ModelError.UnsupportedVersion, // TODO: Implement
            .custom_binary => loadCustomBinary(allocator, path),
        };
    }

    pub fn validate(self: *const Model) !void {
        try self.graph.validate();

        // Verify all required weights are present
        for (self.graph.nodes.items) |node| {
            // TODO: Check if node requires weights and verify they exist
            _ = node;
        }
    }

    pub fn execute(self: *const Model, inputs: []tensor.Tensor) ![]tensor.Tensor {
        // TODO: Implement graph execution
        _ = self;
        _ = inputs;
        return ModelError.InvalidGraph;
    }
};

// Placeholder implementations for different model formats
fn loadONNX(allocator: Allocator, path: []const u8) !Model {
    _ = allocator;
    _ = path;
    // TODO: Implement ONNX loading
    return ModelError.UnsupportedVersion;
}

fn loadCustomBinary(allocator: Allocator, path: []const u8) !Model {
    _ = allocator;
    _ = path;
    // TODO: Implement custom binary format loading
    return ModelError.UnsupportedVersion;
}

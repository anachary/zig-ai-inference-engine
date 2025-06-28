const std = @import("std");
const Allocator = std.mem.Allocator;
const model = @import("../model.zig");
const tensor = @import("../../core/tensor.zig");

pub const ONNXError = error{
    InvalidProtobuf,
    UnsupportedOpset,
    MissingGraph,
    InvalidNode,
    UnsupportedDataType,
    OutOfMemory,
};

// ONNX data type mappings
pub const ONNXDataType = enum(i32) {
    undefined = 0,
    float = 1,
    uint8 = 2,
    int8 = 3,
    uint16 = 4,
    int16 = 5,
    int32 = 6,
    int64 = 7,
    string = 8,
    bool = 9,
    float16 = 10,
    double = 11,
    uint32 = 12,
    uint64 = 13,
    complex64 = 14,
    complex128 = 15,
    bfloat16 = 16,

    pub fn toTensorDataType(self: ONNXDataType) !tensor.DataType {
        return switch (self) {
            .float => .f32,
            .float16 => .f16,
            .int32 => .i32,
            .int16 => .i16,
            .int8 => .i8,
            .uint8 => .u8,
            else => ONNXError.UnsupportedDataType,
        };
    }
};

// Simplified ONNX structures (without full protobuf dependency)
pub const ONNXTensorProto = struct {
    dims: []i64,
    data_type: ONNXDataType,
    name: []const u8,
    raw_data: []const u8,

    pub fn init(allocator: Allocator, name: []const u8, dims: []const i64, data_type: ONNXDataType) !ONNXTensorProto {
        return ONNXTensorProto{
            .dims = try allocator.dupe(i64, dims),
            .data_type = data_type,
            .name = try allocator.dupe(u8, name),
            .raw_data = &[_]u8{},
        };
    }

    pub fn deinit(self: *ONNXTensorProto, allocator: Allocator) void {
        allocator.free(self.dims);
        allocator.free(self.name);
        if (self.raw_data.len > 0) {
            allocator.free(self.raw_data);
        }
    }

    pub fn toTensor(self: *const ONNXTensorProto, allocator: Allocator) !tensor.Tensor {
        // Convert ONNX dimensions to usize
        var shape = try allocator.alloc(usize, self.dims.len);
        defer allocator.free(shape);

        for (self.dims, 0..) |dim, i| {
            if (dim < 0) return ONNXError.InvalidNode;
            shape[i] = @intCast(dim);
        }

        const dtype = try self.data_type.toTensorDataType();
        var t = try tensor.Tensor.init(allocator, shape, dtype);

        // Copy raw data if available
        if (self.raw_data.len > 0) {
            const expected_size = t.size_bytes();
            if (self.raw_data.len != expected_size) {
                t.deinit();
                return ONNXError.InvalidNode;
            }
            @memcpy(t.data, self.raw_data);
        }

        return t;
    }
};

pub const ONNXNodeProto = struct {
    name: []const u8,
    op_type: []const u8,
    input: [][]const u8,
    output: [][]const u8,
    attributes: std.StringHashMap(model.AttributeValue),

    pub fn init(allocator: Allocator, name: []const u8, op_type: []const u8) !ONNXNodeProto {
        return ONNXNodeProto{
            .name = try allocator.dupe(u8, name),
            .op_type = try allocator.dupe(u8, op_type),
            .input = &[_][]const u8{},
            .output = &[_][]const u8{},
            .attributes = std.StringHashMap(model.AttributeValue).init(allocator),
        };
    }

    pub fn deinit(self: *ONNXNodeProto, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.op_type);

        for (self.input) |input| {
            allocator.free(input);
        }
        allocator.free(self.input);

        for (self.output) |output| {
            allocator.free(output);
        }
        allocator.free(self.output);

        var attr_iter = self.attributes.iterator();
        while (attr_iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(allocator);
        }
        self.attributes.deinit();
    }

    pub fn toGraphNode(self: *const ONNXNodeProto, allocator: Allocator) !model.GraphNode {
        var node = try model.GraphNode.init(allocator, self.name, self.op_type);

        // Copy inputs
        node.inputs = try allocator.alloc([]const u8, self.input.len);
        for (self.input, 0..) |input, i| {
            node.inputs[i] = try allocator.dupe(u8, input);
        }

        // Copy outputs
        node.outputs = try allocator.alloc([]const u8, self.output.len);
        for (self.output, 0..) |output, i| {
            node.outputs[i] = try allocator.dupe(u8, output);
        }

        // Copy attributes
        var attr_iter = self.attributes.iterator();
        while (attr_iter.next()) |entry| {
            const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
            try node.attributes.put(key_copy, entry.value_ptr.*);
        }

        return node;
    }
};

pub const ONNXGraphProto = struct {
    name: []const u8,
    nodes: []ONNXNodeProto,
    initializers: []ONNXTensorProto,
    inputs: []ONNXTensorProto,
    outputs: []ONNXTensorProto,

    pub fn init(allocator: Allocator, name: []const u8) !ONNXGraphProto {
        return ONNXGraphProto{
            .name = try allocator.dupe(u8, name),
            .nodes = &[_]ONNXNodeProto{},
            .initializers = &[_]ONNXTensorProto{},
            .inputs = &[_]ONNXTensorProto{},
            .outputs = &[_]ONNXTensorProto{},
        };
    }

    pub fn deinit(self: *ONNXGraphProto, allocator: Allocator) void {
        allocator.free(self.name);

        for (self.nodes) |*node| {
            node.deinit(allocator);
        }
        allocator.free(self.nodes);

        for (self.initializers) |*initializer| {
            initializer.deinit(allocator);
        }
        allocator.free(self.initializers);

        for (self.inputs) |*input| {
            input.deinit(allocator);
        }
        allocator.free(self.inputs);

        for (self.outputs) |*output| {
            output.deinit(allocator);
        }
        allocator.free(self.outputs);
    }
};

pub const ONNXModelProto = struct {
    ir_version: i64,
    opset_import: []OpsetImport,
    producer_name: []const u8,
    producer_version: []const u8,
    graph: ONNXGraphProto,

    pub const OpsetImport = struct {
        domain: []const u8,
        version: i64,
    };

    pub fn init(allocator: Allocator) !ONNXModelProto {
        return ONNXModelProto{
            .ir_version = 7, // Default to ONNX IR version 7
            .opset_import = &[_]OpsetImport{},
            .producer_name = try allocator.dupe(u8, "zig-ai-engine"),
            .producer_version = try allocator.dupe(u8, "0.1.0"),
            .graph = try ONNXGraphProto.init(allocator, "main_graph"),
        };
    }

    pub fn deinit(self: *ONNXModelProto, allocator: Allocator) void {
        for (self.opset_import) |opset| {
            allocator.free(opset.domain);
        }
        allocator.free(self.opset_import);

        allocator.free(self.producer_name);
        allocator.free(self.producer_version);
        self.graph.deinit(allocator);
    }
};

pub const ONNXParser = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) ONNXParser {
        return ONNXParser{
            .allocator = allocator,
        };
    }

    pub fn parseFile(self: *ONNXParser, path: []const u8) !model.Model {
        // Read file
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            std.log.err("Failed to open ONNX file: {s}", .{path});
            return err;
        };
        defer file.close();

        const file_size = try file.getEndPos();
        const file_data = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(file_data);

        _ = try file.readAll(file_data);

        return self.parseBytes(file_data);
    }

    pub fn parseBytes(self: *ONNXParser, data: []const u8) !model.Model {
        // For now, implement a simplified parser
        // In a full implementation, this would use a proper protobuf parser

        // Create a dummy model for demonstration
        var metadata = try model.ModelMetadata.init(self.allocator, "onnx_model", "1.0");
        metadata.format = .onnx;

        var parsed_model = model.Model.init(self.allocator, metadata);

        // Parse basic structure (simplified)
        try self.parseSimplifiedONNX(&parsed_model, data);

        return parsed_model;
    }

    fn parseSimplifiedONNX(self: *ONNXParser, parsed_model: *model.Model, data: []const u8) !void {
        // This is a simplified parser for demonstration
        // A real implementation would use protobuf parsing

        _ = data; // Suppress unused parameter warning

        // Add some dummy nodes for testing
        var add_node = try model.GraphNode.init(self.allocator, "add_node", "Add");
        try parsed_model.graph.addNode(add_node);

        var relu_node = try model.GraphNode.init(self.allocator, "relu_node", "Relu");
        try parsed_model.graph.addNode(relu_node);

        // Add dummy input/output specs
        const input_shape = [_]i32{ -1, 3, 224, 224 }; // Batch, Channels, Height, Width
        var input_spec = try model.TensorSpec.init(self.allocator, "input", &input_shape, .f32);
        try parsed_model.graph.addInput(input_spec);

        const output_shape = [_]i32{ -1, 1000 }; // Batch, Classes
        var output_spec = try model.TensorSpec.init(self.allocator, "output", &output_shape, .f32);
        try parsed_model.graph.addOutput(output_spec);

        std.log.info("Parsed simplified ONNX model with {} nodes", .{parsed_model.graph.nodes.items.len});
    }

    pub fn validate(self: *ONNXParser, onnx_model: *const ONNXModelProto) !void {
        _ = self;

        // Check IR version
        if (onnx_model.ir_version < 3 or onnx_model.ir_version > 8) {
            return ONNXError.UnsupportedOpset;
        }

        // Check for required graph
        if (onnx_model.graph.nodes.len == 0) {
            return ONNXError.MissingGraph;
        }

        // Validate opset versions
        for (onnx_model.opset_import) |opset| {
            if (std.mem.eql(u8, opset.domain, "") and opset.version > 17) {
                std.log.warn("Unsupported opset version: {d}", .{opset.version});
            }
        }
    }

    pub fn getSupportedOps() []const []const u8 {
        return &[_][]const u8{
            "Add",         "Sub",                "Mul",                "Div",
            "MatMul",      "Gemm",               "Relu",               "Sigmoid",
            "Tanh",        "Softmax",            "Conv",               "MaxPool",
            "AveragePool", "BatchNormalization", "LayerNormalization", "Reshape",
            "Transpose",   "Squeeze",            "Unsqueeze",          "Concat",
            "Split",       "Constant",           "Identity",
        };
    }

    pub fn isOpSupported(op_type: []const u8) bool {
        const supported_ops = getSupportedOps();
        for (supported_ops) |supported_op| {
            if (std.mem.eql(u8, op_type, supported_op)) {
                return true;
            }
        }
        return false;
    }
};

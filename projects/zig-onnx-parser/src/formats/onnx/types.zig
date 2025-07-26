const std = @import("std");
const Allocator = std.mem.Allocator;

/// ONNX data types and their mappings
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

    /// Convert ONNX data type to tensor core data type
    pub fn toTensorDataType(self: ONNXDataType) !@import("../../tensor_interface.zig").DataType {
        return switch (self) {
            .float => .f32,
            .float16 => .f16,
            .double => .f64,
            .int8 => .i8,
            .int16 => .i16,
            .int32 => .i32,
            .int64 => .i64,
            .uint8 => .u8,
            .uint16 => .u16,
            .uint32 => .u32,
            .uint64 => .u64,
            .bool => .u8, // Represent bool as u8
            else => error.UnsupportedDataType,
        };
    }

    /// Get size in bytes for this data type
    pub fn sizeBytes(self: ONNXDataType) usize {
        return switch (self) {
            .float, .int32, .uint32 => 4,
            .double, .int64, .uint64, .complex64 => 8,
            .float16, .int16, .uint16, .bfloat16 => 2,
            .int8, .uint8, .bool => 1,
            .complex128 => 16,
            .string => 0, // Variable size
            .undefined => 0,
        };
    }

    /// Check if data type is supported
    pub fn isSupported(self: ONNXDataType) bool {
        return switch (self) {
            .float, .float16, .double, .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .bool => true,
            else => false,
        };
    }
};

/// ONNX tensor type information
pub const ONNXTensorType = struct {
    elem_type: ONNXDataType,
    shape: ?ONNXShape,

    pub fn init(elem_type: ONNXDataType, shape: ?ONNXShape) ONNXTensorType {
        return ONNXTensorType{
            .elem_type = elem_type,
            .shape = shape,
        };
    }

    pub fn deinit(self: *ONNXTensorType, allocator: Allocator) void {
        if (self.shape) |*shape| {
            shape.deinit(allocator);
        }
    }
};

/// ONNX shape information
pub const ONNXShape = struct {
    dims: []i64, // -1 for dynamic dimensions

    pub fn init(allocator: Allocator, dims: []const i64) !ONNXShape {
        return ONNXShape{
            .dims = try allocator.dupe(i64, dims),
        };
    }

    pub fn deinit(self: *ONNXShape, allocator: Allocator) void {
        allocator.free(self.dims);
    }

    /// Check if shape is fully defined (no dynamic dimensions)
    pub fn isFullyDefined(self: *const ONNXShape) bool {
        for (self.dims) |dim| {
            if (dim < 0) return false;
        }
        return true;
    }

    /// Get total number of elements (returns null if dynamic)
    pub fn numel(self: *const ONNXShape) ?usize {
        if (!self.isFullyDefined()) return null;

        var total: usize = 1;
        for (self.dims) |dim| {
            total *= @as(usize, @intCast(dim));
        }
        return total;
    }

    /// Convert to usize array (fails if dynamic dimensions present)
    pub fn toUsizeArray(self: *const ONNXShape, allocator: Allocator) ![]usize {
        if (!self.isFullyDefined()) return error.DynamicShape;

        const result = try allocator.alloc(usize, self.dims.len);
        for (self.dims, 0..) |dim, i| {
            result[i] = @as(usize, @intCast(dim));
        }
        return result;
    }
};

/// ONNX value information
pub const ONNXValueInfo = struct {
    name: []const u8,
    type: ?ONNXTensorType,
    doc_string: []const u8,

    pub fn init(allocator: Allocator, name: []const u8, doc_string: []const u8) !ONNXValueInfo {
        return ONNXValueInfo{
            .name = try allocator.dupe(u8, name),
            .type = null,
            .doc_string = try allocator.dupe(u8, doc_string),
        };
    }

    pub fn deinit(self: *ONNXValueInfo, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.doc_string);
        if (self.type) |*tensor_type| {
            tensor_type.deinit(allocator);
        }
    }
};

/// ONNX attribute value types
pub const ONNXAttributeValue = union(enum) {
    float: f32,
    int: i64,
    string: []const u8,
    tensor: ONNXTensor,
    graph: *ONNXGraph,
    floats: []f32,
    ints: []i64,
    strings: [][]const u8,
    tensors: []ONNXTensor,
    graphs: []*ONNXGraph,

    pub fn deinit(self: *ONNXAttributeValue, allocator: Allocator) void {
        switch (self.*) {
            .string => |str| allocator.free(str),
            .tensor => |*tensor| tensor.deinit(allocator),
            .graph => |graph| {
                graph.deinit(allocator);
                allocator.destroy(graph);
            },
            .floats => |floats| allocator.free(floats),
            .ints => |ints| allocator.free(ints),
            .strings => |strings| {
                for (strings) |str| allocator.free(str);
                allocator.free(strings);
            },
            .tensors => |tensors| {
                for (tensors) |*tensor| tensor.deinit(allocator);
                allocator.free(tensors);
            },
            .graphs => |graphs| {
                for (graphs) |graph| {
                    graph.deinit(allocator);
                    allocator.destroy(graph);
                }
                allocator.free(graphs);
            },
            else => {},
        }
    }
};

/// ONNX node attribute
pub const ONNXAttribute = struct {
    name: []const u8,
    value: ONNXAttributeValue,
    doc_string: []const u8,

    pub fn init(allocator: Allocator, name: []const u8, value: ONNXAttributeValue, doc_string: []const u8) !ONNXAttribute {
        return ONNXAttribute{
            .name = try allocator.dupe(u8, name),
            .value = value,
            .doc_string = try allocator.dupe(u8, doc_string),
        };
    }

    pub fn deinit(self: *ONNXAttribute, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.doc_string);
        self.value.deinit(allocator);
    }
};

/// ONNX tensor representation
pub const ONNXTensor = struct {
    dims: []i64,
    data_type: ONNXDataType,
    raw_data: []const u8,
    name: []const u8,
    doc_string: []const u8,

    pub fn init(allocator: Allocator, name: []const u8, data_type: ONNXDataType, dims: []const i64) !ONNXTensor {
        return ONNXTensor{
            .dims = try allocator.dupe(i64, dims),
            .data_type = data_type,
            .raw_data = &[_]u8{},
            .name = try allocator.dupe(u8, name),
            .doc_string = try allocator.dupe(u8, ""),
        };
    }

    pub fn deinit(self: *ONNXTensor, allocator: Allocator) void {
        allocator.free(self.dims);
        allocator.free(self.name);
        allocator.free(self.doc_string);
        if (self.raw_data.len > 0) {
            allocator.free(self.raw_data);
        }
    }

    /// Get total number of elements
    pub fn numel(self: *const ONNXTensor) usize {
        var total: usize = 1;
        for (self.dims) |dim| {
            total *= @as(usize, @intCast(@max(dim, 1)));
        }
        return total;
    }

    /// Get size in bytes
    pub fn sizeBytes(self: *const ONNXTensor) usize {
        return self.numel() * self.data_type.sizeBytes();
    }

    /// Extract tensor data as f32 array
    pub fn extractF32Data(self: *const ONNXTensor, allocator: Allocator) ![]f32 {
        if (self.raw_data.len == 0) {
            return error.NoTensorData;
        }

        const num_elements = self.numel();
        var result = try allocator.alloc(f32, num_elements);

        switch (self.data_type) {
            .float => {
                if (self.raw_data.len != num_elements * 4) {
                    allocator.free(result);
                    return error.InvalidDataSize;
                }
                // Convert raw bytes to f32 (little-endian)
                for (0..num_elements) |i| {
                    const bytes = self.raw_data[i * 4 .. (i + 1) * 4];
                    result[i] = @bitCast(std.mem.readIntLittle(u32, bytes[0..4]));
                }
            },
            .double => {
                if (self.raw_data.len != num_elements * 8) {
                    allocator.free(result);
                    return error.InvalidDataSize;
                }
                // Convert f64 to f32
                for (0..num_elements) |i| {
                    const bytes = self.raw_data[i * 8 .. (i + 1) * 8];
                    const f64_val: f64 = @bitCast(std.mem.readIntLittle(u64, bytes[0..8]));
                    result[i] = @floatCast(f64_val);
                }
            },
            .int32 => {
                if (self.raw_data.len != num_elements * 4) {
                    allocator.free(result);
                    return error.InvalidDataSize;
                }
                // Convert i32 to f32
                for (0..num_elements) |i| {
                    const bytes = self.raw_data[i * 4 .. (i + 1) * 4];
                    const i32_val = std.mem.readIntLittle(i32, bytes[0..4]);
                    result[i] = @floatFromInt(i32_val);
                }
            },
            .int64 => {
                if (self.raw_data.len != num_elements * 8) {
                    allocator.free(result);
                    return error.InvalidDataSize;
                }
                // Convert i64 to f32
                for (0..num_elements) |i| {
                    const bytes = self.raw_data[i * 8 .. (i + 1) * 8];
                    const i64_val = std.mem.readIntLittle(i64, bytes[0..8]);
                    result[i] = @floatFromInt(i64_val);
                }
            },
            else => {
                allocator.free(result);
                return error.UnsupportedDataType;
            },
        }

        return result;
    }

    /// Extract tensor data as i32 array
    pub fn extractI32Data(self: *const ONNXTensor, allocator: Allocator) ![]i32 {
        if (self.raw_data.len == 0) {
            return error.NoTensorData;
        }

        const num_elements = self.numel();
        var result = try allocator.alloc(i32, num_elements);

        switch (self.data_type) {
            .int32 => {
                if (self.raw_data.len != num_elements * 4) {
                    allocator.free(result);
                    return error.InvalidDataSize;
                }
                // Convert raw bytes to i32 (little-endian)
                for (0..num_elements) |i| {
                    const bytes = self.raw_data[i * 4 .. (i + 1) * 4];
                    result[i] = std.mem.readIntLittle(i32, bytes[0..4]);
                }
            },
            .int64 => {
                if (self.raw_data.len != num_elements * 8) {
                    allocator.free(result);
                    return error.InvalidDataSize;
                }
                // Convert i64 to i32 (with bounds checking)
                for (0..num_elements) |i| {
                    const bytes = self.raw_data[i * 8 .. (i + 1) * 8];
                    const i64_val = std.mem.readIntLittle(i64, bytes[0..8]);
                    if (i64_val > std.math.maxInt(i32) or i64_val < std.math.minInt(i32)) {
                        allocator.free(result);
                        return error.IntegerOverflow;
                    }
                    result[i] = @intCast(i64_val);
                }
            },
            .float => {
                if (self.raw_data.len != num_elements * 4) {
                    allocator.free(result);
                    return error.InvalidDataSize;
                }
                // Convert f32 to i32
                for (0..num_elements) |i| {
                    const bytes = self.raw_data[i * 4 .. (i + 1) * 4];
                    const f32_val: f32 = @bitCast(std.mem.readIntLittle(u32, bytes[0..4]));
                    result[i] = @intFromFloat(f32_val);
                }
            },
            else => {
                allocator.free(result);
                return error.UnsupportedDataType;
            },
        }

        return result;
    }

    /// Get shape as usize array for tensor operations
    pub fn getShape(self: *const ONNXTensor, allocator: Allocator) ![]usize {
        var shape = try allocator.alloc(usize, self.dims.len);
        for (self.dims, 0..) |dim, i| {
            shape[i] = @intCast(@max(dim, 1));
        }
        return shape;
    }
};

/// ONNX node representation
pub const ONNXNode = struct {
    name: []const u8,
    op_type: []const u8,
    domain: []const u8,
    inputs: [][]const u8,
    outputs: [][]const u8,
    attributes: std.StringHashMap(ONNXAttribute),
    doc_string: []const u8,

    pub fn init(allocator: Allocator, name: []const u8, op_type: []const u8) !ONNXNode {
        return ONNXNode{
            .name = try allocator.dupe(u8, name),
            .op_type = try allocator.dupe(u8, op_type),
            .domain = try allocator.dupe(u8, ""),
            .inputs = &[_][]const u8{},
            .outputs = &[_][]const u8{},
            .attributes = std.StringHashMap(ONNXAttribute).init(allocator),
            .doc_string = try allocator.dupe(u8, ""),
        };
    }

    pub fn deinit(self: *ONNXNode, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.op_type);
        allocator.free(self.domain);
        allocator.free(self.doc_string);

        for (self.inputs) |input| allocator.free(input);
        allocator.free(self.inputs);

        for (self.outputs) |output| allocator.free(output);
        allocator.free(self.outputs);

        var attr_iter = self.attributes.iterator();
        while (attr_iter.next()) |entry| {
            entry.value_ptr.deinit(allocator);
        }
        self.attributes.deinit();
    }

    /// Add an attribute to the node
    pub fn addAttribute(self: *ONNXNode, allocator: Allocator, name: []const u8, value: ONNXAttributeValue) !void {
        const attr = try ONNXAttribute.init(allocator, name, value, "");
        try self.attributes.put(try allocator.dupe(u8, name), attr);
    }

    /// Get attribute by name
    pub fn getAttribute(self: *const ONNXNode, name: []const u8) ?*const ONNXAttribute {
        return self.attributes.getPtr(name);
    }
};

/// ONNX graph representation
pub const ONNXGraph = struct {
    name: []const u8,
    nodes: std.ArrayList(ONNXNode),
    initializers: std.ArrayList(ONNXTensor),
    inputs: std.ArrayList(ONNXValueInfo),
    outputs: std.ArrayList(ONNXValueInfo),
    value_info: std.ArrayList(ONNXValueInfo),
    doc_string: []const u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8) !ONNXGraph {
        return ONNXGraph{
            .name = try allocator.dupe(u8, name),
            .nodes = std.ArrayList(ONNXNode).init(allocator),
            .initializers = std.ArrayList(ONNXTensor).init(allocator),
            .inputs = std.ArrayList(ONNXValueInfo).init(allocator),
            .outputs = std.ArrayList(ONNXValueInfo).init(allocator),
            .value_info = std.ArrayList(ONNXValueInfo).init(allocator),
            .doc_string = try allocator.dupe(u8, ""),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ONNXGraph, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.doc_string);

        for (self.nodes.items) |*node| node.deinit(allocator);
        self.nodes.deinit();

        for (self.initializers.items) |*tensor| tensor.deinit(allocator);
        self.initializers.deinit();

        for (self.inputs.items) |*input| input.deinit(allocator);
        self.inputs.deinit();

        for (self.outputs.items) |*output| output.deinit(allocator);
        self.outputs.deinit();

        for (self.value_info.items) |*info| info.deinit(allocator);
        self.value_info.deinit();
    }

    /// Add a node to the graph
    pub fn addNode(self: *ONNXGraph, node: ONNXNode) !void {
        try self.nodes.append(node);
    }

    /// Add an initializer tensor
    pub fn addInitializer(self: *ONNXGraph, tensor: ONNXTensor) !void {
        try self.initializers.append(tensor);
    }

    /// Add input specification
    pub fn addInput(self: *ONNXGraph, input: ONNXValueInfo) !void {
        try self.inputs.append(input);
    }

    /// Add output specification
    pub fn addOutput(self: *ONNXGraph, output: ONNXValueInfo) !void {
        try self.outputs.append(output);
    }

    /// Validate graph connectivity
    pub fn validate(self: *const ONNXGraph) !void {
        // Check that all node inputs are either graph inputs or outputs of other nodes
        var available_values = std.StringHashMap(void).init(self.allocator);
        defer available_values.deinit();

        // Add graph inputs
        for (self.inputs.items) |input| {
            try available_values.put(input.name, {});
        }

        // Add initializers
        for (self.initializers.items) |initializer| {
            try available_values.put(initializer.name, {});
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
};

/// ONNX model representation
pub const ONNXModel = struct {
    ir_version: i64,
    opset_imports: []OpsetImport,
    producer_name: []const u8,
    producer_version: []const u8,
    domain: []const u8,
    model_version: i64,
    doc_string: []const u8,
    graph: ONNXGraph,
    metadata_props: std.StringHashMap([]const u8),

    pub const OpsetImport = struct {
        domain: []const u8,
        version: i64,
    };

    pub fn init(allocator: Allocator, graph: ONNXGraph) !ONNXModel {
        return ONNXModel{
            .ir_version = 7,
            .opset_imports = &[_]OpsetImport{},
            .producer_name = try allocator.dupe(u8, "zig-onnx-parser"),
            .producer_version = try allocator.dupe(u8, "0.1.0"),
            .domain = try allocator.dupe(u8, ""),
            .model_version = 1,
            .doc_string = try allocator.dupe(u8, ""),
            .graph = graph,
            .metadata_props = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *ONNXModel, allocator: Allocator) void {
        allocator.free(self.producer_name);
        allocator.free(self.producer_version);
        allocator.free(self.domain);
        allocator.free(self.doc_string);

        for (self.opset_imports) |*import| {
            allocator.free(import.domain);
        }
        allocator.free(self.opset_imports);

        var metadata_iter = self.metadata_props.iterator();
        while (metadata_iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata_props.deinit();

        self.graph.deinit(allocator);
    }

    /// Get model metadata for inference engine
    pub fn getMetadata(self: *const ONNXModel) ModelMetadata {
        return ModelMetadata{
            .name = self.producer_name,
            .input_count = self.graph.inputs.items.len,
            .output_count = self.graph.outputs.items.len,
            .node_count = self.graph.nodes.items.len,
            .initializer_count = self.graph.initializers.items.len,
        };
    }

    /// Extract all model weights as tensor interfaces
    pub fn extractWeights(self: *const ONNXModel, allocator: Allocator) !std.StringHashMap(WeightTensor) {
        var weights = std.StringHashMap(WeightTensor).init(allocator);

        for (self.graph.initializers.items) |*tensor| {
            if (tensor.raw_data.len > 0) {
                const shape = try tensor.getShape(allocator);
                defer allocator.free(shape);

                const weight_tensor = WeightTensor{
                    .name = try allocator.dupe(u8, tensor.name),
                    .shape = try allocator.dupe(usize, shape),
                    .data_type = tensor.data_type,
                    .data = try allocator.dupe(u8, tensor.raw_data),
                };

                try weights.put(weight_tensor.name, weight_tensor);
            }
        }

        return weights;
    }

    /// Find weight tensor by name
    pub fn getWeight(self: *const ONNXModel, name: []const u8) ?*const ONNXTensor {
        for (self.graph.initializers.items) |*tensor| {
            if (std.mem.eql(u8, tensor.name, name)) {
                return tensor;
            }
        }
        return null;
    }

    /// Get input tensor info
    pub fn getInputInfo(self: *const ONNXModel, index: usize) ?InputInfo {
        if (index >= self.graph.inputs.items.len) return null;

        const input = &self.graph.inputs.items[index];
        return InputInfo{
            .name = input.name,
            .data_type = if (input.type_info) |type_info| type_info.tensor_type.elem_type else .undefined,
            .shape = if (input.type_info) |type_info| type_info.tensor_type.shape.dims else &[_]i64{},
        };
    }

    /// Get output tensor info
    pub fn getOutputInfo(self: *const ONNXModel, index: usize) ?OutputInfo {
        if (index >= self.graph.outputs.items.len) return null;

        const output = &self.graph.outputs.items[index];
        return OutputInfo{
            .name = output.name,
            .data_type = if (output.type_info) |type_info| type_info.tensor_type.elem_type else .undefined,
            .shape = if (output.type_info) |type_info| type_info.tensor_type.shape.dims else &[_]i64{},
        };
    }

    /// Validate model for inference
    pub fn validate(self: *const ONNXModel) !void {
        if (self.graph.inputs.items.len == 0) {
            return error.NoInputs;
        }
        if (self.graph.outputs.items.len == 0) {
            return error.NoOutputs;
        }
        if (self.graph.nodes.items.len == 0) {
            return error.NoNodes;
        }
        // Model is valid for inference
    }
};

/// Model metadata for inference engine
pub const ModelMetadata = struct {
    name: []const u8,
    input_count: usize,
    output_count: usize,
    node_count: usize,
    initializer_count: usize,
};

/// Weight tensor representation
pub const WeightTensor = struct {
    name: []const u8,
    shape: []usize,
    data_type: ONNXDataType,
    data: []const u8,

    pub fn deinit(self: *WeightTensor, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.shape);
        allocator.free(self.data);
    }
};

/// Input tensor information
pub const InputInfo = struct {
    name: []const u8,
    data_type: ONNXDataType,
    shape: []const i64,
};

/// Output tensor information
pub const OutputInfo = struct {
    name: []const u8,
    data_type: ONNXDataType,
    shape: []const i64,
};

const std = @import("std");
const Allocator = std.mem.Allocator;
const protobuf = @import("protobuf.zig");
const types = @import("types.zig");
const model = @import("../model.zig");

const ONNXModel = types.ONNXModel;
const ONNXGraph = types.ONNXGraph;
const ONNXNode = types.ONNXNode;
const ONNXTensor = types.ONNXTensor;
const ONNXValueInfo = types.ONNXValueInfo;
const ONNXDataType = types.ONNXDataType;
const ProtobufParser = protobuf.ProtobufParser;

pub const ONNXError = error{
    InvalidProtobuf,
    UnsupportedOpset,
    MissingGraph,
    InvalidNode,
    UnsupportedDataType,
    OutOfMemory,
    ParseError,
    UnsupportedVersion,
    InvalidModel,
    MissingInput,
    MissingOutput,
} || protobuf.ProtobufParser.ProtobufError;

/// ONNX parser configuration
pub const ParserConfig = struct {
    /// Maximum model size in MB
    max_model_size_mb: u32 = 1024,
    /// Enable strict validation
    strict_validation: bool = true,
    /// Enable graph optimizations during parsing
    enable_optimizations: bool = false,
    /// Minimum supported opset version
    min_opset_version: i64 = 11,
    /// Maximum supported opset version
    max_opset_version: i64 = 18,
    /// Buffer size for streaming parser
    buffer_size_kb: u32 = 64,
};

/// ONNX parser for protobuf format
pub const ONNXParser = struct {
    allocator: Allocator,
    config: ParserConfig,

    const Self = @This();

    pub fn init(allocator: Allocator, config: ParserConfig) Self {
        return Self{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Parse ONNX model from file
    pub fn parseFile(self: *Self, path: []const u8) ONNXError!model.Model {
        std.log.info("📂 Loading ONNX file: {s}", .{path});

        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            std.log.err("Failed to open ONNX file: {s}", .{path});
            return switch (err) {
                error.FileNotFound => ONNXError.ParseError,
                error.AccessDenied => ONNXError.ParseError,
                else => ONNXError.ParseError,
            };
        };
        defer file.close();

        const file_size = try file.getEndPos();
        
        // Check file size limit
        const max_size = @as(u64, self.config.max_model_size_mb) * 1024 * 1024;
        if (file_size > max_size) {
            std.log.err("Model file too large: {} bytes (max: {} MB)", .{ file_size, self.config.max_model_size_mb });
            return ONNXError.ParseError;
        }

        const file_data = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(file_data);

        _ = try file.readAll(file_data);
        std.log.info("📊 File size: {d:.1} MB", .{@as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0)});

        return self.parseBytes(file_data);
    }

    /// Parse ONNX model from byte array
    pub fn parseBytes(self: *Self, data: []const u8) ONNXError!model.Model {
        std.log.info("🔍 Starting ONNX protobuf parsing...", .{});

        // Initialize protobuf parser
        var pb_parser = ProtobufParser.init(self.allocator, data);

        // Parse ONNX ModelProto
        var onnx_model = try self.parseModelProto(&pb_parser);
        defer onnx_model.deinit(self.allocator);

        // Validate opset version
        try self.validateOpsetVersion(&onnx_model);

        // Validate model if strict validation is enabled
        if (self.config.strict_validation) {
            try onnx_model.graph.validate();
        }

        // Convert to internal model format
        const parsed_model = try self.convertToInternalModel(onnx_model);

        std.log.info("✅ ONNX model parsed successfully", .{});
        return parsed_model;
    }

    /// Parse ModelProto from protobuf data
    fn parseModelProto(self: *Self, parser: *ProtobufParser) ONNXError!ONNXModel {
        std.log.info("📋 Parsing ONNX ModelProto...", .{});

        var graph: ?ONNXGraph = null;
        var ir_version: i64 = 7;
        var producer_name = try self.allocator.dupe(u8, "unknown");
        var producer_version = try self.allocator.dupe(u8, "unknown");
        var opset_imports = std.ArrayList(ONNXModel.OpsetImport).init(self.allocator);
        defer opset_imports.deinit();

        while (parser.hasMoreData()) {
            const header = parser.readFieldHeader() catch break;

            switch (header.field_number) {
                1 => { // ir_version
                    if (header.wire_type == .varint) {
                        ir_version = try parser.readInt64();
                        std.log.info("IR Version: {}", .{ir_version});
                    } else {
                        try parser.skipField(header.wire_type);
                    }
                },
                2 => { // producer_name
                    if (header.wire_type == .length_delimited) {
                        self.allocator.free(producer_name);
                        producer_name = try self.allocator.dupe(u8, try parser.readString());
                        std.log.info("Producer: {s}", .{producer_name});
                    } else {
                        try parser.skipField(header.wire_type);
                    }
                },
                3 => { // producer_version
                    if (header.wire_type == .length_delimited) {
                        self.allocator.free(producer_version);
                        producer_version = try self.allocator.dupe(u8, try parser.readString());
                        std.log.info("Producer Version: {s}", .{producer_version});
                    } else {
                        try parser.skipField(header.wire_type);
                    }
                },
                7 => { // graph
                    if (header.wire_type == .length_delimited) {
                        graph = try self.parseGraphProto(parser);
                        std.log.info("Graph parsed: {s}", .{graph.?.name});
                    } else {
                        try parser.skipField(header.wire_type);
                    }
                },
                8 => { // opset_import
                    if (header.wire_type == .length_delimited) {
                        const opset = try self.parseOpsetImport(parser);
                        try opset_imports.append(opset);
                    } else {
                        try parser.skipField(header.wire_type);
                    }
                },
                else => {
                    try parser.skipField(header.wire_type);
                },
            }
        }

        if (graph == null) {
            std.log.err("No graph found in ONNX model");
            return ONNXError.MissingGraph;
        }

        var onnx_model = try ONNXModel.init(self.allocator, graph.?);
        onnx_model.ir_version = ir_version;
        onnx_model.producer_name = producer_name;
        onnx_model.producer_version = producer_version;
        onnx_model.opset_imports = try opset_imports.toOwnedSlice();

        return onnx_model;
    }

    /// Parse GraphProto from protobuf data
    fn parseGraphProto(self: *Self, parser: *ProtobufParser) ONNXError!ONNXGraph {
        const data = try parser.readBytes();
        var sub_parser = ProtobufParser.init(self.allocator, data);

        var graph = try ONNXGraph.init(self.allocator, "main_graph");

        while (sub_parser.hasMoreData()) {
            const header = sub_parser.readFieldHeader() catch break;

            switch (header.field_number) {
                1 => { // node
                    if (header.wire_type == .length_delimited) {
                        const node = try self.parseNodeProto(&sub_parser);
                        try graph.addNode(node);
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                2 => { // name
                    if (header.wire_type == .length_delimited) {
                        self.allocator.free(graph.name);
                        graph.name = try self.allocator.dupe(u8, try sub_parser.readString());
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                5 => { // initializer
                    if (header.wire_type == .length_delimited) {
                        const tensor = try self.parseTensorProto(&sub_parser);
                        try graph.addInitializer(tensor);
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                11 => { // input
                    if (header.wire_type == .length_delimited) {
                        const input = try self.parseValueInfoProto(&sub_parser);
                        try graph.addInput(input);
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                12 => { // output
                    if (header.wire_type == .length_delimited) {
                        const output = try self.parseValueInfoProto(&sub_parser);
                        try graph.addOutput(output);
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                else => {
                    try sub_parser.skipField(header.wire_type);
                },
            }
        }

        std.log.info("📊 Graph statistics:");
        std.log.info("  Nodes: {}", .{graph.nodes.items.len});
        std.log.info("  Inputs: {}", .{graph.inputs.items.len});
        std.log.info("  Outputs: {}", .{graph.outputs.items.len});
        std.log.info("  Initializers: {}", .{graph.initializers.items.len});

        return graph;
    }

    /// Parse NodeProto from protobuf data
    fn parseNodeProto(self: *Self, parser: *ProtobufParser) ONNXError!ONNXNode {
        const data = try parser.readBytes();
        var sub_parser = ProtobufParser.init(self.allocator, data);

        var node = try ONNXNode.init(self.allocator, "", "");
        var inputs = std.ArrayList([]const u8).init(self.allocator);
        defer inputs.deinit();
        var outputs = std.ArrayList([]const u8).init(self.allocator);
        defer outputs.deinit();

        while (sub_parser.hasMoreData()) {
            const header = sub_parser.readFieldHeader() catch break;

            switch (header.field_number) {
                1 => { // input
                    if (header.wire_type == .length_delimited) {
                        const input_name = try self.allocator.dupe(u8, try sub_parser.readString());
                        try inputs.append(input_name);
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                2 => { // output
                    if (header.wire_type == .length_delimited) {
                        const output_name = try self.allocator.dupe(u8, try sub_parser.readString());
                        try outputs.append(output_name);
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                3 => { // name
                    if (header.wire_type == .length_delimited) {
                        self.allocator.free(node.name);
                        node.name = try self.allocator.dupe(u8, try sub_parser.readString());
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                4 => { // op_type
                    if (header.wire_type == .length_delimited) {
                        self.allocator.free(node.op_type);
                        node.op_type = try self.allocator.dupe(u8, try sub_parser.readString());
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                else => {
                    try sub_parser.skipField(header.wire_type);
                },
            }
        }

        node.inputs = try inputs.toOwnedSlice();
        node.outputs = try outputs.toOwnedSlice();

        return node;
    }

    /// Parse ValueInfoProto from protobuf data
    fn parseValueInfoProto(self: *Self, parser: *ProtobufParser) ONNXError!ONNXValueInfo {
        const data = try parser.readBytes();
        var sub_parser = ProtobufParser.init(self.allocator, data);

        var value_info = try ONNXValueInfo.init(self.allocator, "", "");

        while (sub_parser.hasMoreData()) {
            const header = sub_parser.readFieldHeader() catch break;

            switch (header.field_number) {
                1 => { // name
                    if (header.wire_type == .length_delimited) {
                        self.allocator.free(value_info.name);
                        value_info.name = try self.allocator.dupe(u8, try sub_parser.readString());
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                else => {
                    try sub_parser.skipField(header.wire_type);
                },
            }
        }

        return value_info;
    }

    /// Parse TensorProto from protobuf data
    fn parseTensorProto(self: *Self, parser: *ProtobufParser) ONNXError!ONNXTensor {
        const data = try parser.readBytes();
        var sub_parser = ProtobufParser.init(self.allocator, data);

        var tensor = try ONNXTensor.init(self.allocator, "", .undefined, &[_]i64{});

        while (sub_parser.hasMoreData()) {
            const header = sub_parser.readFieldHeader() catch break;

            switch (header.field_number) {
                1 => { // dims
                    if (header.wire_type == .length_delimited) {
                        const dims_data = try sub_parser.readRepeatedVarint(self.allocator);
                        defer self.allocator.free(dims_data);
                        
                        self.allocator.free(tensor.dims);
                        tensor.dims = try self.allocator.alloc(i64, dims_data.len);
                        for (dims_data, 0..) |dim, i| {
                            tensor.dims[i] = @as(i64, @intCast(dim));
                        }
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                2 => { // data_type
                    if (header.wire_type == .varint) {
                        const dt = try sub_parser.readInt32();
                        tensor.data_type = @as(ONNXDataType, @enumFromInt(dt));
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                8 => { // name
                    if (header.wire_type == .length_delimited) {
                        self.allocator.free(tensor.name);
                        tensor.name = try self.allocator.dupe(u8, try sub_parser.readString());
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                9 => { // raw_data
                    if (header.wire_type == .length_delimited) {
                        const raw_data = try sub_parser.readBytes();
                        tensor.raw_data = try self.allocator.dupe(u8, raw_data);
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                else => {
                    try sub_parser.skipField(header.wire_type);
                },
            }
        }

        return tensor;
    }

    /// Parse OpsetImport from protobuf data
    fn parseOpsetImport(self: *Self, parser: *ProtobufParser) ONNXError!ONNXModel.OpsetImport {
        const data = try parser.readBytes();
        var sub_parser = ProtobufParser.init(self.allocator, data);

        var domain = try self.allocator.dupe(u8, "");
        var version: i64 = 1;

        while (sub_parser.hasMoreData()) {
            const header = sub_parser.readFieldHeader() catch break;

            switch (header.field_number) {
                1 => { // domain
                    if (header.wire_type == .length_delimited) {
                        self.allocator.free(domain);
                        domain = try self.allocator.dupe(u8, try sub_parser.readString());
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                2 => { // version
                    if (header.wire_type == .varint) {
                        version = try sub_parser.readInt64();
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                else => {
                    try sub_parser.skipField(header.wire_type);
                },
            }
        }

        return ONNXModel.OpsetImport{
            .domain = domain,
            .version = version,
        };
    }

    /// Validate opset version compatibility
    fn validateOpsetVersion(self: *Self, onnx_model: *const ONNXModel) ONNXError!void {
        for (onnx_model.opset_imports) |opset| {
            if (std.mem.eql(u8, opset.domain, "") or std.mem.eql(u8, opset.domain, "ai.onnx")) {
                if (opset.version < self.config.min_opset_version or opset.version > self.config.max_opset_version) {
                    std.log.err("Unsupported opset version: {} (supported: {}-{})", .{ opset.version, self.config.min_opset_version, self.config.max_opset_version });
                    return ONNXError.UnsupportedOpset;
                }
                std.log.info("✅ Opset version {} is supported", .{opset.version});
                return;
            }
        }
        
        std.log.warn("No standard opset found, assuming default version");
    }

    /// Convert ONNX model to internal representation
    fn convertToInternalModel(self: *Self, onnx_model: ONNXModel) ONNXError!model.Model {
        std.log.info("🔄 Converting ONNX model to internal format...", .{});

        // Create model metadata
        var metadata = try model.ModelMetadata.init(
            self.allocator,
            onnx_model.producer_name,
            onnx_model.producer_version,
        );
        metadata.format = .onnx;
        metadata.ir_version = onnx_model.ir_version;
        metadata.input_count = onnx_model.graph.inputs.items.len;
        metadata.output_count = onnx_model.graph.outputs.items.len;

        var internal_model = model.Model.init(self.allocator, metadata);

        // Convert nodes (simplified conversion)
        for (onnx_model.graph.nodes.items) |onnx_node| {
            const internal_node = try model.GraphNode.init(
                self.allocator,
                onnx_node.name,
                onnx_node.op_type,
            );
            try internal_model.graph.addNode(internal_node);
        }

        std.log.info("✅ Conversion completed", .{});
        return internal_model;
    }
};

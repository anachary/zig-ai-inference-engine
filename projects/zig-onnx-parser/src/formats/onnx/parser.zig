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
    /// Enable strict validation (set to false for real models)
    strict_validation: bool = false,
    /// Enable graph optimizations during parsing
    enable_optimizations: bool = false,
    /// Minimum supported opset version
    min_opset_version: i64 = 7, // Lower for broader compatibility
    /// Maximum supported opset version
    max_opset_version: i64 = 20, // Higher for newer models
    /// Buffer size for streaming parser
    buffer_size_kb: u32 = 64,
    /// Skip unknown operators instead of failing
    skip_unknown_ops: bool = true,
    /// Allow partial parsing with missing features
    allow_partial_parsing: bool = true,
    /// Enable verbose logging for debugging
    verbose_logging: bool = true,
    /// Continue parsing even with non-critical errors
    error_recovery: bool = true,
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

        const file_size = file.getEndPos() catch |err| {
            std.log.err("Failed to get file size: {any}", .{err});
            return ONNXError.ParseError;
        };

        // Check file size limit
        const max_size = @as(u64, self.config.max_model_size_mb) * 1024 * 1024;
        if (file_size > max_size) {
            std.log.err("Model file too large: {d} bytes (max: {d} MB)", .{ file_size, self.config.max_model_size_mb });
            return ONNXError.ParseError;
        }

        const file_data = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(file_data);

        _ = file.readAll(file_data) catch |err| {
            std.log.err("Failed to read file data: {any}", .{err});
            return ONNXError.ParseError;
        };
        std.log.info("📊 File size: {d:.1} MB", .{@as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0)});

        return self.parseBytes(file_data);
    }

    /// Parse ONNX model from byte array
    pub fn parseBytes(self: *Self, data: []const u8) ONNXError!model.Model {
        std.log.info("🔍 Starting ONNX protobuf parsing...", .{});
        std.log.info("📊 Model size: {d:.2} MB", .{@as(f64, @floatFromInt(data.len)) / (1024.0 * 1024.0)});

        // Validate protobuf magic bytes for ONNX
        if (data.len < 8) {
            std.log.err("❌ File too small to be a valid ONNX model", .{});
            return ONNXError.ParseError;
        }

        // Initialize protobuf parser with error recovery
        var pb_parser = ProtobufParser.init(self.allocator, data);

        // Parse ONNX ModelProto with enhanced error handling
        var onnx_model = self.parseModelProto(&pb_parser) catch |err| {
            std.log.err("❌ Failed to parse ModelProto: {any}", .{err});
            std.log.info("💡 This might be due to: {s}", .{"compatibility issues"});
            std.log.info("   - Unsupported ONNX version: {s}", .{"check compatibility"});
            std.log.info("   - Corrupted model file", .{});
            std.log.info("   - Complex model features not yet supported", .{});
            return err;
        };
        defer onnx_model.deinit(self.allocator);

        std.log.info("✅ ModelProto parsed successfully", .{});
        std.log.info("📋 Model info: {s} v{s}", .{ onnx_model.producer_name, onnx_model.producer_version });

        // Validate opset version with detailed feedback
        self.validateOpsetVersion(&onnx_model) catch |err| {
            std.log.warn("⚠️  Opset validation failed: {any}", .{err});
            if (!self.config.strict_validation) {
                std.log.info("🔄 Continuing with relaxed validation...", .{});
            } else {
                return err;
            }
        };

        // Validate model if strict validation is enabled
        if (self.config.strict_validation) {
            onnx_model.graph.validate() catch |err| {
                std.log.warn("⚠️  Graph validation failed: {any}", .{err});
                std.log.info("💡 Try setting strict_validation = false for experimental models", .{});
                return err;
            };
        }

        // Convert to internal model format with enhanced conversion
        const parsed_model = self.convertToInternalModel(onnx_model) catch |err| {
            std.log.err("❌ Failed to convert to internal format: {any}", .{err});
            std.log.info("💡 This model may use features not yet supported", .{});
            return err;
        };

        std.log.info("✅ ONNX model parsed successfully", .{});
        std.log.info("📊 Final model stats:", .{});
        std.log.info("   - Nodes: {d}", .{parsed_model.graph.nodes.items.len});
        std.log.info("   - Inputs: {d}", .{parsed_model.graph.inputs.items.len});
        std.log.info("   - Outputs: {d}", .{parsed_model.graph.outputs.items.len});

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
                        std.log.info("IR Version: {d}", .{ir_version});
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
                    // Enhanced unknown field handling with error recovery
                    if (self.config.verbose_logging) {
                        std.log.info("Skipping unknown ModelProto field {d} with wire type {d}", .{ header.field_number, header.wire_type });
                    }
                    parser.skipField(header.wire_type) catch |err| {
                        if (self.config.error_recovery) {
                            std.log.warn("Error skipping ModelProto field {d}: {any}, continuing", .{ header.field_number, err });
                        } else {
                            return err;
                        }
                    };
                },
            }
        }

        if (graph == null) {
            std.log.err("No graph found in ONNX model: {s}", .{"missing graph"});
            return ONNXError.MissingGraph;
        }

        var onnx_model = try ONNXModel.init(self.allocator, graph.?);
        onnx_model.ir_version = ir_version;

        // Free the default strings before overwriting them
        self.allocator.free(onnx_model.producer_name);
        self.allocator.free(onnx_model.producer_version);

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
                13 => { // value_info (intermediate values)
                    if (header.wire_type == .length_delimited) {
                        var value_info = try self.parseValueInfoProto(&sub_parser);
                        defer value_info.deinit(self.allocator); // Clean up immediately since we don't store it
                        // Store intermediate value info if needed
                        if (self.config.verbose_logging) {
                            std.log.info("Found intermediate value: {s}", .{value_info.name});
                        }
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                14 => { // quantization_annotation
                    if (header.wire_type == .length_delimited) {
                        // Skip quantization annotations for now
                        _ = try sub_parser.readBytes();
                        if (self.config.verbose_logging) {
                            std.log.info("Skipping quantization annotation: {s}", .{"not supported"});
                        }
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                else => {
                    // Enhanced unknown field handling with error recovery
                    if (self.config.verbose_logging) {
                        std.log.info("Skipping unknown GraphProto field {d} with wire type {d}", .{ header.field_number, header.wire_type });
                    }
                    sub_parser.skipField(header.wire_type) catch |err| {
                        if (self.config.error_recovery) {
                            std.log.warn("Error skipping GraphProto field {d}: {any}, continuing", .{ header.field_number, err });
                        } else {
                            return err;
                        }
                    };
                },
            }
        }

        std.log.info("📊 Graph statistics: {s}", .{"analyzing"});
        std.log.info("  Nodes: {d}", .{graph.nodes.items.len});
        std.log.info("  Inputs: {d}", .{graph.inputs.items.len});
        std.log.info("  Outputs: {d}", .{graph.outputs.items.len});
        std.log.info("  Initializers: {d}", .{graph.initializers.items.len});

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
                5 => { // attribute
                    if (header.wire_type == .length_delimited) {
                        // Parse attributes (simplified for now)
                        _ = try sub_parser.readBytes();
                        if (self.config.verbose_logging) {
                            std.log.info("Skipping node attribute for {s}", .{node.op_type});
                        }
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                6 => { // doc_string
                    if (header.wire_type == .length_delimited) {
                        _ = try sub_parser.readString();
                        if (self.config.verbose_logging) {
                            std.log.info("Skipping doc string for node {s}", .{node.name});
                        }
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                else => {
                    // Enhanced unknown field handling for nodes
                    if (self.config.verbose_logging) {
                        std.log.info("Skipping unknown NodeProto field {d} for node {s}", .{ header.field_number, node.name });
                    }
                    sub_parser.skipField(header.wire_type) catch |err| {
                        if (self.config.error_recovery) {
                            std.log.warn("Error skipping NodeProto field {d}: {any}, continuing", .{ header.field_number, err });
                        } else {
                            return err;
                        }
                    };
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

        // Initialize with default values without allocating
        var value_info = ONNXValueInfo{
            .name = "",
            .type = null,
            .doc_string = "",
        };
        var name_allocated = false;
        var doc_allocated = false;

        while (sub_parser.hasMoreData()) {
            const header = sub_parser.readFieldHeader() catch break;

            switch (header.field_number) {
                1 => { // name
                    if (header.wire_type == .length_delimited) {
                        if (name_allocated) {
                            self.allocator.free(value_info.name);
                        }
                        value_info.name = try self.allocator.dupe(u8, try sub_parser.readString());
                        name_allocated = true;
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                6 => { // doc_string
                    if (header.wire_type == .length_delimited) {
                        if (doc_allocated) {
                            self.allocator.free(value_info.doc_string);
                        }
                        value_info.doc_string = try self.allocator.dupe(u8, try sub_parser.readString());
                        doc_allocated = true;
                    } else {
                        try sub_parser.skipField(header.wire_type);
                    }
                },
                else => {
                    try sub_parser.skipField(header.wire_type);
                },
            }
        }

        // Ensure all strings are allocated (use empty string if not set)
        if (!name_allocated) {
            value_info.name = try self.allocator.dupe(u8, "");
        }
        if (!doc_allocated) {
            value_info.doc_string = try self.allocator.dupe(u8, "");
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
                    std.log.err("Unsupported opset version: {d} (supported: {d}-{d})", .{ opset.version, self.config.min_opset_version, self.config.max_opset_version });
                    return ONNXError.UnsupportedOpset;
                }
                std.log.info("✅ Opset version {d} is supported", .{opset.version});
                return;
            }
        }

        std.log.warn("No standard opset found, assuming default version", .{});
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

        // Convert nodes with enhanced operator support
        var converted_nodes: usize = 0;
        var skipped_nodes: usize = 0;

        for (onnx_model.graph.nodes.items) |onnx_node| {
            // Check if operator is supported or should be skipped
            const is_supported = self.isOperatorSupported(onnx_node.op_type);

            if (!is_supported and self.config.skip_unknown_ops) {
                std.log.warn("Skipping unsupported operator: {s} (node: {s})", .{ onnx_node.op_type, onnx_node.name });
                skipped_nodes += 1;
                continue;
            }

            const internal_node = model.GraphNode.init(
                self.allocator,
                onnx_node.name,
                onnx_node.op_type,
            ) catch |err| {
                if (self.config.error_recovery) {
                    std.log.warn("Failed to create node {s} ({s}): {any}, skipping", .{ onnx_node.name, onnx_node.op_type, err });
                    skipped_nodes += 1;
                    continue;
                } else {
                    return err;
                }
            };

            internal_model.graph.addNode(internal_node) catch |err| {
                if (self.config.error_recovery) {
                    std.log.warn("Failed to add node {s}: {any}, skipping", .{ onnx_node.name, err });
                    skipped_nodes += 1;
                    continue;
                } else {
                    return err;
                }
            };

            converted_nodes += 1;
        }

        std.log.info("📊 Node conversion summary:", .{});
        std.log.info("   Converted: {d}/{d}", .{ converted_nodes, onnx_model.graph.nodes.items.len });
        if (skipped_nodes > 0) {
            std.log.warn("   Skipped: {d} unsupported/failed nodes", .{skipped_nodes});
        }

        std.log.info("✅ Conversion completed", .{});
        return internal_model;
    }

    /// Check if an operator is supported
    fn isOperatorSupported(self: *Self, op_type: []const u8) bool {
        _ = self; // Suppress unused parameter warning

        // List of commonly supported ONNX operators
        const supported_ops = [_][]const u8{
            // Basic math operations
            "Add",      "Sub",           "Mul",                "Div",                "Pow",           "Sqrt",              "Abs",                       "Neg",
            "Min",      "Max",           "Sum",                "Mean",               "Clip",

            // Activation functions
                     "Relu",              "Sigmoid",                   "Tanh",
            "Softmax",  "LeakyRelu",     "Elu",                "Selu",               "Swish",         "Gelu",              "HardSigmoid",               "HardSwish",

            // Neural network layers
            "Conv",     "ConvTranspose", "BatchNormalization", "LayerNormalization", "Dropout",       "Flatten",           "Reshape",                   "Transpose",
            "Squeeze",  "Unsqueeze",

            // Pooling operations
                "MaxPool",            "AveragePool",        "GlobalMaxPool", "GlobalAveragePool",

            // Matrix operations
            "MatMul",                    "Gemm",
            "Dot",      "Identity",

            // Tensor operations
                 "Concat",             "Split",              "Slice",         "Gather",            "Scatter",                   "Tile",
            "Expand",   "Pad",           "Constant",           "ConstantOfShape",

            // Comparison and logical
               "Equal",         "Greater",           "Less",                      "And",
            "Or",       "Not",           "Where",

            // Reduction operations
                         "ReduceSum",          "ReduceMean",    "ReduceMax",         "ReduceMin",                 "ReduceProd",
            "ReduceL1", "ReduceL2",      "ReduceLogSum",

            // Shape operations
                  "Shape",              "Size",          "Cast",              "Range",

            // Control flow (basic support)
                                "If",
            "Loop",     "Scan",

            // Common LLM operators
                     "Attention",          "MultiHeadAttention", "LayerNorm",     "RMSNorm",           "RotaryPositionalEmbedding", "Embedding",
            "Linear",
        };

        // Check if operator is in supported list
        for (supported_ops) |supported_op| {
            if (std.mem.eql(u8, op_type, supported_op)) {
                return true;
            }
        }

        // Check for common operator patterns
        if (std.mem.startsWith(u8, op_type, "Reduce") or
            std.mem.startsWith(u8, op_type, "Conv") or
            std.mem.startsWith(u8, op_type, "Pool") or
            std.mem.startsWith(u8, op_type, "Batch") or
            std.mem.startsWith(u8, op_type, "Layer"))
        {
            return true;
        }

        return false;
    }
};

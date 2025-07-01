const std = @import("std");
const Allocator = std.mem.Allocator;
const onnx_parser = @import("zig-onnx-parser");
const ModelParserFactory = @import("../model_parser_factory.zig");
const ModelTypeIdentifier = @import("../model_type_identifier.zig");
const ParserConfig = ModelParserFactory.ParserConfig;
const ParsedModel = ModelParserFactory.ParsedModel;
const ParserError = ModelParserFactory.ParserError;
const ModelCharacteristics = ModelTypeIdentifier.ModelCharacteristics;
const ModelArchitecture = ModelTypeIdentifier.ModelArchitecture;

/// Generic parser for unknown or unsupported model architectures
pub const GenericModelParser = struct {
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn parse(ctx: *anyopaque, config: ParserConfig) ParserError!ParsedModel {
        const self = @as(*Self, @ptrCast(@alignCast(ctx)));

        std.log.info("🔧 Parsing generic model: {s}", .{config.model_path});
        const start_time = std.time.milliTimestamp();

        // Validate memory constraints first
        try self.validateMemoryConstraints(config);

        // Parse ONNX model with basic validation
        var parser = onnx_parser.Parser.init(self.allocator);
        var model = parser.parseFile(config.model_path) catch |err| {
            std.log.err("❌ ONNX parsing failed: {any}", .{err});
            return switch (err) {
                error.ParseError => ParserError.InvalidModelFormat,
                error.OutOfMemory => ParserError.OutOfMemory,
                error.InvalidProtobuf => ParserError.InvalidModelFormat,
                error.UnsupportedOpset => ParserError.UnsupportedArchitecture,
                error.MissingGraph => ParserError.CorruptedModel,
                error.InvalidNode => ParserError.CorruptedModel,
                error.UnsupportedDataType => ParserError.UnsupportedArchitecture,
                error.UnsupportedVersion => ParserError.UnsupportedArchitecture,
                error.InvalidModel => ParserError.CorruptedModel,
                error.MissingInput => ParserError.CorruptedModel,
                error.MissingOutput => ParserError.CorruptedModel,
                else => ParserError.CorruptedModel,
            };
        };
        // Ensure model is cleaned up on any error from this point forward
        errdefer model.deinit();

        // Perform basic model analysis
        var characteristics = try self.analyzeGenericModel(&model);
        characteristics.architecture = .unknown;

        // Basic validation
        if (config.validate_model) {
            self.validateGenericModel(&model) catch |err| {
                std.log.warn("⚠️  Model validation failed, but continuing: {any}", .{err});
                // Don't fail completely, just warn and continue
            };
        }

        // Calculate memory usage
        const memory_usage = try self.calculateMemoryUsage(&model);

        // Check if model fits in memory constraints
        if (!config.memory_constraints.canLoadModel(memory_usage)) {
            model.deinit();
            return ParserError.InsufficientMemory;
        }

        const load_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));

        std.log.info("✅ Generic model parsed successfully in {d}ms", .{load_time});
        std.log.info("📊 Memory usage: {d:.2} MB", .{@as(f64, @floatFromInt(memory_usage)) / (1024.0 * 1024.0)});

        return ParsedModel{
            .model = model,
            .characteristics = characteristics,
            .parser_type = .unknown,
            .memory_usage_bytes = memory_usage,
            .load_time_ms = load_time,
        };
    }

    pub fn validate(ctx: *anyopaque, model_path: []const u8) ParserError!void {
        const self = @as(*Self, @ptrCast(@alignCast(ctx)));
        _ = self;

        std.log.info("🔍 Validating generic model: {s}", .{model_path});

        // Check file existence and basic format
        const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => ParserError.ModelNotFound,
                error.AccessDenied => ParserError.ModelNotFound,
                else => ParserError.InvalidModelFormat,
            };
        };
        defer file.close();

        // Basic file size check
        const file_size = file.getEndPos() catch return ParserError.InvalidModelFormat;
        if (file_size == 0) {
            return ParserError.CorruptedModel;
        }

        // Check if it's a valid ONNX file by reading the header
        var buffer: [16]u8 = undefined;
        _ = file.readAll(&buffer) catch return ParserError.InvalidModelFormat;

        // Reset file position (ignore errors for validation)
        file.seekTo(0) catch {};

        // Basic protobuf format validation
        if (buffer.len >= 2) {
            const first_byte = buffer[0];
            // Check for protobuf field markers
            if (first_byte < 0x08 or first_byte > 0x7A) {
                std.log.warn("⚠️  File may not be a valid ONNX model", .{});
            }
        }

        std.log.info("✅ Generic model validation passed", .{});
    }

    pub fn deinitParser(ctx: *anyopaque) void {
        const self = @as(*Self, @ptrCast(@alignCast(ctx)));
        const allocator = self.allocator;
        self.deinit();
        allocator.destroy(self);
    }

    /// Analyze generic model characteristics
    fn analyzeGenericModel(self: *Self, model: *const onnx_parser.Model) !ModelCharacteristics {
        var characteristics = ModelCharacteristics.init();
        characteristics.architecture = .unknown;

        // Count different operator types
        var operator_counts = std.StringHashMap(usize).init(self.allocator);
        defer operator_counts.deinit();

        for (model.graph.nodes.items) |node| {
            const result = try operator_counts.getOrPut(node.op_type);
            if (!result.found_existing) {
                result.value_ptr.* = 0;
            }
            result.value_ptr.* += 1;
        }

        // Analyze operator patterns to infer characteristics
        var has_conv = false;
        var has_matmul = false;
        var has_gather = false;
        var has_attention_ops = false;
        var has_normalization = false;

        var op_iter = operator_counts.iterator();
        while (op_iter.next()) |entry| {
            const op_type = entry.key_ptr.*;
            const count = entry.value_ptr.*;

            std.log.info("   - {s}: {d} operations", .{ op_type, count });

            if (std.mem.eql(u8, op_type, "Conv") or
                std.mem.eql(u8, op_type, "Conv2D") or
                std.mem.eql(u8, op_type, "Conv3D"))
            {
                has_conv = true;
                characteristics.has_convolution = true;
            } else if (std.mem.eql(u8, op_type, "MatMul")) {
                has_matmul = true;
            } else if (std.mem.eql(u8, op_type, "Gather")) {
                has_gather = true;
                characteristics.has_embedding = true;
            } else if (std.mem.eql(u8, op_type, "Softmax") or
                std.mem.eql(u8, op_type, "Attention"))
            {
                has_attention_ops = true;
                characteristics.has_attention = true;
            } else if (std.mem.eql(u8, op_type, "LayerNormalization") or
                std.mem.eql(u8, op_type, "BatchNormalization"))
            {
                has_normalization = true;
                characteristics.has_normalization = true;
            }
        }

        // Provide basic architecture hints
        if (has_conv and has_normalization) {
            std.log.info("💡 Hint: Model appears to have vision-like characteristics", .{});
        } else if (has_gather and has_attention_ops) {
            std.log.info("💡 Hint: Model appears to have language-like characteristics", .{});
        } else if (has_matmul and !has_conv) {
            std.log.info("💡 Hint: Model appears to be fully-connected or transformer-like", .{});
        }

        // Set a low confidence score since we can't definitively identify the architecture
        characteristics.confidence_score = 0.3;

        std.log.info("📊 Generic model analysis:", .{});
        std.log.info("   - Total operators: {d}", .{model.graph.nodes.items.len});
        std.log.info("   - Unique operator types: {d}", .{operator_counts.count()});
        std.log.info("   - Has convolution: {}", .{has_conv});
        std.log.info("   - Has matrix multiplication: {}", .{has_matmul});
        std.log.info("   - Has embedding (Gather): {}", .{has_gather});
        std.log.info("   - Has attention patterns: {}", .{has_attention_ops});

        return characteristics;
    }

    /// Validate generic model structure
    fn validateGenericModel(self: *Self, model: *const onnx_parser.Model) !void {
        _ = self;

        std.log.info("🔍 Validating generic model structure...", .{});

        // Check basic model structure
        const inputs = model.getInputs();
        const outputs = model.getOutputs();

        if (inputs.len == 0) {
            std.log.warn("⚠️  Model has no inputs - this may be unusual", .{});
            // Don't fail validation, just warn
        }

        if (outputs.len == 0) {
            std.log.warn("⚠️  Model has no outputs - this may be unusual", .{});
            // Don't fail validation, just warn
        }

        if (model.graph.nodes.items.len == 0) {
            std.log.warn("⚠️  Model has no computation nodes - this may be unusual", .{});
            // Don't fail validation, just warn
        }

        // Log model structure
        std.log.info("   - Inputs: {d}", .{inputs.len});
        for (inputs) |input| {
            std.log.info("     * {s}", .{input.name});
        }

        std.log.info("   - Outputs: {d}", .{outputs.len});
        for (outputs) |output| {
            std.log.info("     * {s}", .{output.name});
        }

        std.log.info("   - Computation nodes: {d}", .{model.graph.nodes.items.len});

        // Check for common issues
        var orphaned_nodes: usize = 0;
        for (model.graph.nodes.items) |node| {
            if (node.inputs.len == 0 and node.outputs.len == 0) {
                orphaned_nodes += 1;
            }
        }

        if (orphaned_nodes > 0) {
            std.log.warn("⚠️  Found {d} orphaned nodes (no inputs/outputs)", .{orphaned_nodes});
        }

        std.log.info("✅ Generic model structure validation passed", .{});
    }

    /// Calculate memory usage for the model
    fn calculateMemoryUsage(self: *Self, model: *const onnx_parser.Model) !usize {
        _ = self;

        // Basic memory calculation
        const metadata = model.getMetadata();
        var total_memory: usize = metadata.model_size_bytes;

        // Add overhead for inference (conservative estimate)
        total_memory += total_memory / 2; // 50% overhead for unknown models

        // Add base overhead for runtime structures
        total_memory += 100 * 1024 * 1024; // 100MB base overhead

        return total_memory;
    }

    /// Validate memory constraints
    fn validateMemoryConstraints(self: *Self, config: ParserConfig) !void {
        _ = self;

        const file = std.fs.cwd().openFile(config.model_path, .{}) catch {
            return ParserError.ModelNotFound;
        };
        defer file.close();

        const file_size = file.getEndPos() catch return ParserError.InvalidModelFormat;

        if (!config.memory_constraints.canLoadModel(file_size)) {
            std.log.err("❌ Model too large: {d} MB > {d} MB limit", .{
                file_size / (1024 * 1024),
                config.memory_constraints.max_model_size_bytes / (1024 * 1024),
            });
            return ParserError.ModelTooLarge;
        }
    }
};

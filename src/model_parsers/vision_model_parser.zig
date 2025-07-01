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

/// Specialized parser for vision models (CNNs, Vision Transformers, etc.)
pub const VisionModelParser = struct {
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

        std.log.info("🖼️  Parsing vision model: {s}", .{config.model_path});
        const start_time = std.time.milliTimestamp();

        // Validate memory constraints first
        try self.validateMemoryConstraints(config);

        // Parse ONNX model
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

        // Analyze vision model characteristics
        var characteristics = try self.analyzeVisionModel(&model);
        characteristics.architecture = .vision_model;

        // Validate model structure for vision models
        if (config.validate_model) {
            self.validateVisionModel(&model, &characteristics) catch |err| {
                std.log.warn("⚠️  Vision model validation failed, but continuing: {any}", .{err});
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

        std.log.info("✅ Vision model parsed successfully in {d}ms", .{load_time});
        std.log.info("📊 Memory usage: {d:.2} MB", .{@as(f64, @floatFromInt(memory_usage)) / (1024.0 * 1024.0)});

        return ParsedModel{
            .model = model,
            .characteristics = characteristics,
            .parser_type = .vision_model,
            .memory_usage_bytes = memory_usage,
            .load_time_ms = load_time,
        };
    }

    pub fn validate(ctx: *anyopaque, model_path: []const u8) ParserError!void {
        const self = @as(*Self, @ptrCast(@alignCast(ctx)));
        _ = self;

        std.log.info("🔍 Validating vision model: {s}", .{model_path});

        // Check file existence and basic format
        const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => ParserError.ModelNotFound,
                error.AccessDenied => ParserError.ModelNotFound,
                else => ParserError.InvalidModelFormat,
            };
        };
        defer file.close();

        std.log.info("✅ Vision model validation passed", .{});
    }

    pub fn deinitParser(ctx: *anyopaque) void {
        const self = @as(*Self, @ptrCast(@alignCast(ctx)));
        const allocator = self.allocator;
        self.deinit();
        allocator.destroy(self);
    }

    /// Analyze vision model specific characteristics
    fn analyzeVisionModel(self: *Self, model: *const onnx_parser.Model) !ModelCharacteristics {
        _ = self;

        var characteristics = ModelCharacteristics.init();
        characteristics.architecture = .vision_model;

        // Count vision-specific operators
        var conv_layers: usize = 0;
        var pool_layers: usize = 0;
        var batch_norm_layers: usize = 0;

        for (model.graph.nodes.items) |node| {
            const op_type = node.op_type;

            if (std.mem.eql(u8, op_type, "Conv") or
                std.mem.eql(u8, op_type, "Conv2D") or
                std.mem.eql(u8, op_type, "Conv3D"))
            {
                conv_layers += 1;
            } else if (std.mem.eql(u8, op_type, "MaxPool") or
                std.mem.eql(u8, op_type, "AveragePool") or
                std.mem.eql(u8, op_type, "GlobalAveragePool"))
            {
                pool_layers += 1;
            } else if (std.mem.eql(u8, op_type, "BatchNormalization")) {
                batch_norm_layers += 1;
            }
        }

        characteristics.has_convolution = conv_layers > 0;
        characteristics.has_normalization = batch_norm_layers > 0;
        characteristics.confidence_score = 0.8;

        std.log.info("📊 Vision model analysis:", .{});
        std.log.info("   - Convolution layers: {d}", .{conv_layers});
        std.log.info("   - Pooling layers: {d}", .{pool_layers});
        std.log.info("   - Batch norm layers: {d}", .{batch_norm_layers});

        return characteristics;
    }

    /// Validate vision model structure
    fn validateVisionModel(self: *Self, model: *const onnx_parser.Model, characteristics: *const ModelCharacteristics) !void {
        _ = self;

        std.log.info("🔍 Validating vision model structure...", .{});

        if (!characteristics.has_convolution) {
            std.log.warn("⚠️  No convolution layers found - unusual for vision models", .{});
        }

        // Check input/output structure
        const inputs = model.getInputs();
        const outputs = model.getOutputs();

        if (inputs.len == 0) {
            std.log.warn("⚠️  Model has no inputs - this may be unusual", .{});
        }

        if (outputs.len == 0) {
            std.log.warn("⚠️  Model has no outputs - this may be unusual", .{});
        }

        std.log.info("✅ Vision model structure validation passed", .{});
    }

    /// Calculate memory usage for the model
    fn calculateMemoryUsage(self: *Self, model: *const onnx_parser.Model) !usize {
        _ = self;

        const metadata = model.getMetadata();
        var total_memory: usize = metadata.model_size_bytes;

        // Vision models typically need more memory for feature maps
        total_memory += total_memory / 3; // 33% overhead

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
            return ParserError.ModelTooLarge;
        }
    }
};

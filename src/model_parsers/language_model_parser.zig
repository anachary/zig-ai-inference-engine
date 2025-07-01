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

/// Specialized parser for language models (Transformers, RNNs, etc.)
pub const LanguageModelParser = struct {
    allocator: Allocator,
    vocabulary_path: ?[]const u8,
    model_file_path: ?[]const u8,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .vocabulary_path = null,
            .model_file_path = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.vocabulary_path) |path| {
            self.allocator.free(path);
        }
        if (self.model_file_path) |path| {
            self.allocator.free(path);
        }
    }

    /// Set the vocabulary file path for this parser
    pub fn setVocabularyPath(self: *Self, vocab_path: []const u8) void {
        // Free existing path if any
        if (self.vocabulary_path) |existing_path| {
            self.allocator.free(existing_path);
        }
        // Store a copy of the vocabulary path
        self.vocabulary_path = self.allocator.dupe(u8, vocab_path) catch null;
    }

    /// Set the model file path for this parser
    pub fn setModelPath(self: *Self, model_path: []const u8) void {
        // Free existing path if any
        if (self.model_file_path) |existing_path| {
            self.allocator.free(existing_path);
        }
        // Store a copy of the model file path
        self.model_file_path = self.allocator.dupe(u8, model_path) catch null;
    }

    pub fn parse(ctx: *anyopaque, config: ParserConfig) ParserError!ParsedModel {
        const self = @as(*Self, @ptrCast(@alignCast(ctx)));

        // Use discovered model file path if available, otherwise use config path
        const actual_model_path = self.model_file_path orelse config.model_path;

        std.log.info("🔤 Parsing language model: {s}", .{actual_model_path});
        const start_time = std.time.milliTimestamp();

        // Validate memory constraints first
        try self.validateMemoryConstraints(actual_model_path, config.memory_constraints);

        // Parse ONNX model
        var parser = onnx_parser.Parser.init(self.allocator);
        var model = parser.parseFile(actual_model_path) catch |err| {
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

        // Analyze language model characteristics
        var characteristics = try self.analyzeLanguageModel(&model);
        characteristics.architecture = .language_model;

        // Validate model structure for language models
        if (config.validate_model) {
            self.validateLanguageModel(&model, &characteristics) catch |err| {
                std.log.err("❌ Language model validation failed: {any}", .{err});
                return err;
            };
        }

        // Extract vocabulary if requested
        if (config.extract_vocabulary) {
            try self.extractVocabulary(&model, &characteristics);
        }

        // Calculate memory usage
        const memory_usage = try self.calculateMemoryUsage(&model);

        // Check if model fits in memory constraints
        if (!config.memory_constraints.canLoadModel(memory_usage)) {
            model.deinit();
            return ParserError.InsufficientMemory;
        }

        const load_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));

        std.log.info("✅ Language model parsed successfully in {d}ms", .{load_time});
        std.log.info("📊 Memory usage: {d:.2} MB", .{@as(f64, @floatFromInt(memory_usage)) / (1024.0 * 1024.0)});

        return ParsedModel{
            .model = model,
            .characteristics = characteristics,
            .parser_type = .language_model,
            .memory_usage_bytes = memory_usage,
            .load_time_ms = load_time,
        };
    }

    pub fn validate(ctx: *anyopaque, model_path: []const u8) ParserError!void {
        const self = @as(*Self, @ptrCast(@alignCast(ctx)));
        _ = self;

        std.log.info("🔍 Validating language model: {s}", .{model_path});

        // Check file existence and basic format
        const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => ParserError.ModelNotFound,
                error.AccessDenied => ParserError.ModelNotFound,
                else => ParserError.InvalidModelFormat,
            };
        };
        defer file.close();

        // Check for required language model files in the same directory
        const dir_path = std.fs.path.dirname(model_path) orelse ".";
        var dir = std.fs.cwd().openIterableDir(dir_path, .{}) catch {
            return ParserError.MissingRequiredFiles;
        };
        defer dir.close();

        // Look for common language model files
        var has_vocab_file = false;
        var has_config_file = false;

        var iterator = dir.iterate();
        while (iterator.next() catch null) |entry| {
            if (entry.kind != .file) continue;

            const name = entry.name;
            if (std.mem.endsWith(u8, name, "vocab.txt") or
                std.mem.endsWith(u8, name, "tokenizer.json") or
                std.mem.endsWith(u8, name, "vocab.json"))
            {
                has_vocab_file = true;
            }

            if (std.mem.endsWith(u8, name, "config.json") or
                std.mem.endsWith(u8, name, "model_config.json"))
            {
                has_config_file = true;
            }
        }

        // Language models typically need vocabulary files
        if (!has_vocab_file) {
            std.log.warn("⚠️  No vocabulary file found - may affect tokenization", .{});
        }

        if (!has_config_file) {
            std.log.warn("⚠️  No config file found - using default parameters", .{});
        }

        std.log.info("✅ Language model validation passed", .{});
    }

    pub fn deinitParser(ctx: *anyopaque) void {
        const self = @as(*Self, @ptrCast(@alignCast(ctx)));
        const allocator = self.allocator;
        self.deinit();
        allocator.destroy(self);
    }

    /// Analyze language model specific characteristics
    fn analyzeLanguageModel(self: *Self, model: *const onnx_parser.Model) !ModelCharacteristics {
        _ = self;

        var characteristics = ModelCharacteristics.init();
        characteristics.architecture = .language_model;

        // Analyze computation graph for language model patterns
        var embedding_layers: usize = 0;
        var attention_layers: usize = 0;
        var layer_norm_count: usize = 0;
        var matmul_count: usize = 0;
        var max_sequence_length: ?usize = null;
        var estimated_vocab_size: ?usize = null;

        for (model.graph.nodes.items) |node| {
            const op_type = node.op_type;

            if (std.mem.eql(u8, op_type, "Gather")) {
                embedding_layers += 1;
                // Try to estimate vocab size from Gather operations
                // This is heuristic and may not be accurate
                if (estimated_vocab_size == null) {
                    estimated_vocab_size = 50000; // Default estimate
                }
            } else if (std.mem.eql(u8, op_type, "MatMul")) {
                matmul_count += 1;
            } else if (std.mem.eql(u8, op_type, "Softmax")) {
                // Softmax often indicates attention mechanism
                attention_layers += 1;
            } else if (std.mem.eql(u8, op_type, "LayerNormalization") or
                std.mem.eql(u8, op_type, "LayerNorm"))
            {
                layer_norm_count += 1;
            }
        }

        // Estimate model architecture parameters
        characteristics.has_embedding = embedding_layers > 0;
        characteristics.has_attention = attention_layers > 0 or (matmul_count >= 6);
        characteristics.has_normalization = layer_norm_count > 0;
        characteristics.vocab_size = estimated_vocab_size;
        characteristics.sequence_length = max_sequence_length;

        // Estimate number of layers (very rough heuristic)
        if (layer_norm_count > 0) {
            characteristics.num_layers = layer_norm_count / 2; // Typical transformer has 2 layer norms per layer
        }

        // Estimate hidden size and attention heads (would need more sophisticated analysis)
        characteristics.hidden_size = 768; // Common default
        characteristics.num_attention_heads = 12; // Common default

        characteristics.confidence_score = 0.8; // High confidence for language models

        std.log.info("📊 Language model analysis:", .{});
        std.log.info("   - Embedding layers: {d}", .{embedding_layers});
        std.log.info("   - Attention patterns: {d}", .{attention_layers});
        std.log.info("   - Layer normalizations: {d}", .{layer_norm_count});
        std.log.info("   - MatMul operations: {d}", .{matmul_count});

        return characteristics;
    }

    /// Validate language model structure
    fn validateLanguageModel(self: *Self, model: *const onnx_parser.Model, characteristics: *const ModelCharacteristics) !void {
        _ = self;

        std.log.info("🔍 Validating language model structure...", .{});

        // Check for essential language model components
        if (!characteristics.has_embedding) {
            std.log.warn("⚠️  No embedding layers found - unusual for language models", .{});
        }

        if (!characteristics.has_attention and model.graph.nodes.items.len > 10) {
            std.log.warn("⚠️  No attention mechanism detected - may be RNN-based model", .{});
        }

        // Check input/output structure
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

        // Language models typically have sequence inputs
        for (inputs) |input| {
            std.log.info("   - Input: {s}", .{input.name});
        }

        for (outputs) |output| {
            std.log.info("   - Output: {s}", .{output.name});
        }

        std.log.info("✅ Language model structure validation passed", .{});
    }

    /// Extract vocabulary information
    fn extractVocabulary(self: *Self, model: *const onnx_parser.Model, characteristics: *ModelCharacteristics) !void {
        std.log.info("📚 Extracting vocabulary information...", .{});

        // Use the vocabulary extractor with discovered vocabulary path
        const VocabularyExtractor = @import("../vocabulary_extractor.zig").VocabularyExtractor;
        var vocab_extractor = VocabularyExtractor.init(self.allocator);
        defer vocab_extractor.deinit();

        // Initialize with discovered vocabulary file path if available
        if (self.vocabulary_path) |vocab_path| {
            std.log.info("   - Using discovered vocabulary file: {s}", .{vocab_path});
            vocab_extractor.initializeWithDiscoveredFiles(model, vocab_path) catch |err| {
                std.log.warn("⚠️  Failed to load vocabulary from file: {any}", .{err});
                // Continue with model-based extraction as fallback
                vocab_extractor.initializeWithLoadedModel(model) catch |model_err| {
                    std.log.warn("⚠️  Failed to extract vocabulary from model: {any}", .{model_err});
                    return; // Use estimated vocabulary size
                };
            };
        } else {
            std.log.info("   - Using model-based vocabulary extraction", .{});
            vocab_extractor.initializeWithLoadedModel(model) catch |err| {
                std.log.warn("⚠️  Failed to extract vocabulary from model: {any}", .{err});
                return; // Use estimated vocabulary size
            };
        }

        // Update characteristics with actual vocabulary size
        if (vocab_extractor.vocabulary) |vocab| {
            characteristics.vocab_size = vocab.vocab_size;
            std.log.info("   - Actual vocabulary size: {d}", .{vocab.vocab_size});
        } else {
            // Fallback to estimated vocab size
            if (characteristics.vocab_size == null) {
                characteristics.vocab_size = 50000; // Default estimate
            }
            std.log.info("   - Estimated vocabulary size: {d}", .{characteristics.vocab_size.?});
        }
    }

    /// Calculate memory usage for the model
    fn calculateMemoryUsage(self: *Self, model: *const onnx_parser.Model) !usize {
        _ = self;

        // Basic memory calculation based on model metadata
        const metadata = model.getMetadata();
        var total_memory: usize = metadata.model_size_bytes;

        // Add overhead for inference (rough estimate)
        total_memory += total_memory / 4; // 25% overhead for activations

        return total_memory;
    }

    /// Validate memory constraints
    fn validateMemoryConstraints(self: *Self, model_path: []const u8, memory_constraints: ModelParserFactory.MemoryConstraints) !void {
        _ = self;

        // Check if we have enough memory available
        const file = std.fs.cwd().openFile(model_path, .{}) catch {
            return ParserError.ModelNotFound;
        };
        defer file.close();

        const file_size = file.getEndPos() catch return ParserError.InvalidModelFormat;

        if (!memory_constraints.canLoadModel(file_size)) {
            std.log.err("❌ Model too large: {d} MB > {d} MB limit", .{
                file_size / (1024 * 1024),
                memory_constraints.max_model_size_bytes / (1024 * 1024),
            });
            return ParserError.ModelTooLarge;
        }
    }
};

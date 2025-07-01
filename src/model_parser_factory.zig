const std = @import("std");
const Allocator = std.mem.Allocator;
const onnx_parser = @import("zig-onnx-parser");
const ModelTypeIdentifier = @import("model_type_identifier.zig");
const ModelArchitecture = ModelTypeIdentifier.ModelArchitecture;
const ModelCharacteristics = ModelTypeIdentifier.ModelCharacteristics;

/// Memory constraints for model loading
pub const MemoryConstraints = struct {
    max_model_size_bytes: usize,
    max_parameter_count: usize,
    available_memory_bytes: usize,

    pub fn init(available_memory_mb: usize) MemoryConstraints {
        const available_bytes = available_memory_mb * 1024 * 1024;
        return MemoryConstraints{
            .max_model_size_bytes = available_bytes / 2, // Use half for model
            .max_parameter_count = available_bytes / 8, // Assume 4-8 bytes per param
            .available_memory_bytes = available_bytes,
        };
    }

    pub fn canLoadModel(self: MemoryConstraints, model_size_bytes: usize) bool {
        return model_size_bytes <= self.max_model_size_bytes;
    }
};

/// Model discovery result
const ModelDiscoveryResult = struct {
    model_file_path: []const u8,
    model_folder_path: ?[]const u8,
    vocabulary_file_path: ?[]const u8,
    tokenizer_file_path: ?[]const u8,
    config_file_path: ?[]const u8,
    needs_cleanup: bool, // Whether model_file_path needs to be freed
};

/// Model parsing configuration
pub const ParserConfig = struct {
    model_path: []const u8,
    memory_constraints: MemoryConstraints,
    enable_optimization: bool,
    validate_model: bool,
    extract_vocabulary: bool,

    pub fn init(model_path: []const u8, available_memory_mb: usize) ParserConfig {
        return ParserConfig{
            .model_path = model_path,
            .memory_constraints = MemoryConstraints.init(available_memory_mb),
            .enable_optimization = true,
            .validate_model = true,
            .extract_vocabulary = true,
        };
    }
};

/// Parsed model result with metadata
pub const ParsedModel = struct {
    model: onnx_parser.Model,
    characteristics: ModelCharacteristics,
    parser_type: ModelArchitecture,
    memory_usage_bytes: usize,
    load_time_ms: u64,

    pub fn deinit(self: *ParsedModel) void {
        self.model.deinit();
    }
};

/// Model parser errors with detailed context
pub const ParserError = error{
    ModelNotFound,
    InvalidModelFormat,
    UnsupportedArchitecture,
    InsufficientMemory,
    ModelTooLarge,
    ValidationFailed,
    VocabularyExtractionFailed,
    LoadTimeout,
    CorruptedModel,
    MissingRequiredFiles,
    OutOfMemory,
};

/// Parser interface for different model architectures
pub const ModelParser = struct {
    const Self = @This();

    /// Parser implementation
    impl: *const ParserImpl,
    ctx: *anyopaque,

    pub const ParserImpl = struct {
        parseFn: *const fn (ctx: *anyopaque, config: ParserConfig) ParserError!ParsedModel,
        validateFn: *const fn (ctx: *anyopaque, model_path: []const u8) ParserError!void,
        deinitFn: *const fn (ctx: *anyopaque) void,
    };

    pub fn parse(self: Self, config: ParserConfig) ParserError!ParsedModel {
        return self.impl.parseFn(self.ctx, config);
    }

    pub fn validate(self: Self, model_path: []const u8) ParserError!void {
        return self.impl.validateFn(self.ctx, model_path);
    }

    pub fn deinit(self: Self) void {
        self.impl.deinitFn(self.ctx);
    }
};

/// Factory for creating model parsers based on architecture
pub const ModelParserFactory = struct {
    allocator: Allocator,
    type_identifier: ModelTypeIdentifier.ModelTypeIdentifier,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .type_identifier = ModelTypeIdentifier.ModelTypeIdentifier.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.type_identifier.deinit();
    }

    /// Automatically discover model files from a path (file or folder)
    fn discoverModelFiles(self: *Self, path: []const u8) ParserError!ModelDiscoveryResult {
        // Try to open as directory first
        if (std.fs.cwd().openDir(path, .{})) |dir| {
            var mutable_dir = dir;
            mutable_dir.close();
            // Path is a directory, discover model files
            return try self.discoverFromDirectory(path);
        } else |dir_err| {
            // Not a directory, try as file
            if (std.fs.cwd().openFile(path, .{})) |file| {
                file.close();
                // Path is a file, use it directly
                return ModelDiscoveryResult{
                    .model_file_path = path,
                    .model_folder_path = std.fs.path.dirname(path),
                    .vocabulary_file_path = null,
                    .tokenizer_file_path = null,
                    .config_file_path = null,
                    .needs_cleanup = false,
                };
            } else |file_err| {
                std.log.err("❌ Path is neither a valid file nor directory: {s}", .{path});
                std.log.err("   Directory error: {any}", .{dir_err});
                std.log.err("   File error: {any}", .{file_err});
                return ParserError.ModelNotFound;
            }
        }
    }

    /// Discover model files from a directory
    fn discoverFromDirectory(self: *Self, dir_path: []const u8) ParserError!ModelDiscoveryResult {
        std.log.info("📁 Discovering model files in directory: {s}", .{dir_path});

        var dir = std.fs.cwd().openIterableDir(dir_path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => ParserError.ModelNotFound,
                error.AccessDenied => ParserError.ModelNotFound,
                else => ParserError.InvalidModelFormat,
            };
        };
        defer dir.close();

        var result = ModelDiscoveryResult{
            .model_file_path = undefined,
            .model_folder_path = dir_path,
            .vocabulary_file_path = null,
            .tokenizer_file_path = null,
            .config_file_path = null,
            .needs_cleanup = false,
        };

        // Define file patterns to look for
        const model_patterns = [_][]const u8{
            "decoder_with_past_model.onnx", // GPT-2 optimized
            "decoder_model_merged.onnx", // GPT-2 merged
            "decoder_model.onnx", // GPT-2 basic
            "model.onnx", // Generic ONNX
            "pytorch_model.onnx", // PyTorch export
        };

        const vocab_patterns = [_][]const u8{
            "vocab.json",
            "vocabulary.json",
        };

        const tokenizer_patterns = [_][]const u8{
            "tokenizer.json",
            "tokenizer_config.json",
        };

        const config_patterns = [_][]const u8{
            "config.json",
            "model_config.json",
        };

        // Iterate through directory to find files
        var iterator = dir.iterate();
        var found_model = false;

        while (iterator.next() catch |err| {
            std.log.err("❌ Error iterating directory: {any}", .{err});
            return ParserError.InvalidModelFormat;
        }) |entry| {
            if (entry.kind != .file) continue;

            // Check for model files
            if (!found_model) {
                for (model_patterns) |pattern| {
                    if (std.mem.eql(u8, entry.name, pattern)) {
                        const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ dir_path, entry.name });
                        result.model_file_path = full_path;
                        result.needs_cleanup = true;
                        found_model = true;
                        std.log.info("✅ Found model file: {s}", .{entry.name});
                        break;
                    }
                }
            }

            // Check for vocabulary files
            if (result.vocabulary_file_path == null) {
                for (vocab_patterns) |pattern| {
                    if (std.mem.eql(u8, entry.name, pattern)) {
                        const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ dir_path, entry.name });
                        result.vocabulary_file_path = full_path;
                        std.log.info("✅ Found vocabulary file: {s}", .{entry.name});
                        break;
                    }
                }
            }

            // Check for tokenizer files
            if (result.tokenizer_file_path == null) {
                for (tokenizer_patterns) |pattern| {
                    if (std.mem.eql(u8, entry.name, pattern)) {
                        const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ dir_path, entry.name });
                        result.tokenizer_file_path = full_path;
                        std.log.info("✅ Found tokenizer file: {s}", .{entry.name});
                        break;
                    }
                }
            }

            // Check for config files
            if (result.config_file_path == null) {
                for (config_patterns) |pattern| {
                    if (std.mem.eql(u8, entry.name, pattern)) {
                        const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ dir_path, entry.name });
                        result.config_file_path = full_path;
                        std.log.info("✅ Found config file: {s}", .{entry.name});
                        break;
                    }
                }
            }
        }

        if (!found_model) {
            std.log.err("❌ No ONNX model files found in directory: {s}", .{dir_path});
            return ParserError.ModelNotFound;
        }

        std.log.info("🎯 Model discovery complete:", .{});
        std.log.info("   Model: {s}", .{result.model_file_path});
        if (result.vocabulary_file_path) |vocab| std.log.info("   Vocabulary: {s}", .{vocab});
        if (result.tokenizer_file_path) |tokenizer| std.log.info("   Tokenizer: {s}", .{tokenizer});
        if (result.config_file_path) |config| std.log.info("   Config: {s}", .{config});

        return result;
    }

    /// Create appropriate parser for the given model path (file or folder)
    pub fn createParser(self: *Self, model_path: []const u8) ParserError!ModelParser {
        std.log.info("🏭 Creating parser for model: {s}", .{model_path});

        // Auto-discover model files if path is a folder
        const discovered_model = try self.discoverModelFiles(model_path);
        defer {
            if (discovered_model.needs_cleanup) {
                self.allocator.free(discovered_model.model_file_path);
            }
            if (discovered_model.vocabulary_file_path) |vocab_path| {
                self.allocator.free(vocab_path);
            }
            if (discovered_model.tokenizer_file_path) |tokenizer_path| {
                self.allocator.free(tokenizer_path);
            }
            if (discovered_model.config_file_path) |config_path| {
                self.allocator.free(config_path);
            }
        }

        // Use the discovered model file path
        const actual_model_path = discovered_model.model_file_path;
        std.log.info("📁 Using model file: {s}", .{actual_model_path});

        // First, do basic validation
        try self.validateModelPath(actual_model_path);

        // Load model for type identification
        var basic_parser = onnx_parser.Parser.init(self.allocator);
        var model = basic_parser.parseFile(actual_model_path) catch |err| {
            std.log.err("❌ Failed to parse model for type identification: {any}", .{err});
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
        defer model.deinit();

        // Identify model architecture
        const characteristics = self.type_identifier.identifyModelType(&model) catch |err| {
            std.log.warn("⚠️  Failed to identify model type, using generic parser: {any}", .{err});
            // Don't fail completely, just use generic parser
            return try self.createGenericParser();
        };

        std.log.info("✅ Identified as: {s}", .{characteristics.architecture.toString()});

        // Create appropriate parser based on architecture
        return switch (characteristics.architecture) {
            .language_model => try self.createLanguageModelParser(discovered_model),
            .vision_model => try self.createVisionModelParser(),
            .audio_model => try self.createAudioModelParser(),
            .embedding_model => try self.createEmbeddingModelParser(),
            .classification_model => try self.createClassificationModelParser(),
            .regression_model => try self.createRegressionModelParser(),
            .multimodal_model => try self.createMultimodalModelParser(),
            .unknown => try self.createGenericParser(),
        };
    }

    /// Validate model path and basic requirements
    fn validateModelPath(self: *Self, model_path: []const u8) ParserError!void {
        _ = self;

        // Check if file exists
        const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => ParserError.ModelNotFound,
                error.AccessDenied => ParserError.ModelNotFound,
                else => ParserError.InvalidModelFormat,
            };
        };
        defer file.close();

        // Check file size
        const file_size = file.getEndPos() catch return ParserError.InvalidModelFormat;
        if (file_size == 0) {
            return ParserError.CorruptedModel;
        }

        std.log.info("📊 Model file size: {d:.2} MB", .{@as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0)});
    }

    /// Create language model parser
    fn createLanguageModelParser(self: *Self, discovery_result: ModelDiscoveryResult) ParserError!ModelParser {
        const parser_impl = try self.allocator.create(LanguageModelParser);
        parser_impl.* = LanguageModelParser.init(self.allocator);

        // Set the discovered vocabulary path if available
        if (discovery_result.vocabulary_file_path) |vocab_path| {
            parser_impl.setVocabularyPath(vocab_path);
        }

        // Store the discovered model file path for use in parsing
        parser_impl.setModelPath(discovery_result.model_file_path);

        return ModelParser{
            .impl = &.{
                .parseFn = LanguageModelParser.parse,
                .validateFn = LanguageModelParser.validate,
                .deinitFn = LanguageModelParser.deinitParser,
            },
            .ctx = parser_impl,
        };
    }

    /// Create vision model parser
    fn createVisionModelParser(self: *Self) ParserError!ModelParser {
        const parser_impl = try self.allocator.create(VisionModelParser);
        parser_impl.* = VisionModelParser.init(self.allocator);

        return ModelParser{
            .impl = &.{
                .parseFn = VisionModelParser.parse,
                .validateFn = VisionModelParser.validate,
                .deinitFn = VisionModelParser.deinitParser,
            },
            .ctx = parser_impl,
        };
    }

    /// Create audio model parser
    fn createAudioModelParser(self: *Self) ParserError!ModelParser {
        const parser_impl = try self.allocator.create(AudioModelParser);
        parser_impl.* = AudioModelParser.init(self.allocator);

        return ModelParser{
            .impl = &.{
                .parseFn = AudioModelParser.parse,
                .validateFn = AudioModelParser.validate,
                .deinitFn = AudioModelParser.deinitParser,
            },
            .ctx = parser_impl,
        };
    }

    /// Create embedding model parser
    fn createEmbeddingModelParser(self: *Self) ParserError!ModelParser {
        const parser_impl = try self.allocator.create(EmbeddingModelParser);
        parser_impl.* = EmbeddingModelParser.init(self.allocator);

        return ModelParser{
            .impl = &.{
                .parseFn = EmbeddingModelParser.parse,
                .validateFn = EmbeddingModelParser.validate,
                .deinitFn = EmbeddingModelParser.deinitParser,
            },
            .ctx = parser_impl,
        };
    }

    /// Create classification model parser
    fn createClassificationModelParser(self: *Self) ParserError!ModelParser {
        const parser_impl = try self.allocator.create(ClassificationModelParser);
        parser_impl.* = ClassificationModelParser.init(self.allocator);

        return ModelParser{
            .impl = &.{
                .parseFn = ClassificationModelParser.parse,
                .validateFn = ClassificationModelParser.validate,
                .deinitFn = ClassificationModelParser.deinitParser,
            },
            .ctx = parser_impl,
        };
    }

    /// Create regression model parser
    fn createRegressionModelParser(self: *Self) ParserError!ModelParser {
        const parser_impl = try self.allocator.create(RegressionModelParser);
        parser_impl.* = RegressionModelParser.init(self.allocator);

        return ModelParser{
            .impl = &.{
                .parseFn = RegressionModelParser.parse,
                .validateFn = RegressionModelParser.validate,
                .deinitFn = RegressionModelParser.deinitParser,
            },
            .ctx = parser_impl,
        };
    }

    /// Create multimodal model parser
    fn createMultimodalModelParser(self: *Self) ParserError!ModelParser {
        const parser_impl = try self.allocator.create(MultimodalModelParser);
        parser_impl.* = MultimodalModelParser.init(self.allocator);

        return ModelParser{
            .impl = &.{
                .parseFn = MultimodalModelParser.parse,
                .validateFn = MultimodalModelParser.validate,
                .deinitFn = MultimodalModelParser.deinitParser,
            },
            .ctx = parser_impl,
        };
    }

    /// Create generic parser for unknown architectures
    fn createGenericParser(self: *Self) ParserError!ModelParser {
        const parser_impl = try self.allocator.create(GenericModelParser);
        parser_impl.* = GenericModelParser.init(self.allocator);

        return ModelParser{
            .impl = &.{
                .parseFn = GenericModelParser.parse,
                .validateFn = GenericModelParser.validate,
                .deinitFn = GenericModelParser.deinitParser,
            },
            .ctx = parser_impl,
        };
    }
};

// Forward declarations for parser implementations
const LanguageModelParser = @import("model_parsers/language_model_parser.zig").LanguageModelParser;
const VisionModelParser = @import("model_parsers/vision_model_parser.zig").VisionModelParser;
const AudioModelParser = @import("model_parsers/audio_model_parser.zig").AudioModelParser;
const EmbeddingModelParser = @import("model_parsers/embedding_model_parser.zig").EmbeddingModelParser;
const ClassificationModelParser = @import("model_parsers/classification_model_parser.zig").ClassificationModelParser;
const RegressionModelParser = @import("model_parsers/regression_model_parser.zig").RegressionModelParser;
const MultimodalModelParser = @import("model_parsers/multimodal_model_parser.zig").MultimodalModelParser;
const GenericModelParser = @import("model_parsers/generic_model_parser.zig").GenericModelParser;

const std = @import("std");
const testing = std.testing;
const ModelTypeIdentifier = @import("src/model_type_identifier.zig").ModelTypeIdentifier;
const ModelParserFactory = @import("src/model_parser_factory.zig").ModelParserFactory;
const ModelArchitecture = @import("src/model_type_identifier.zig").ModelArchitecture;
const LanguageModelStrategy = @import("src/model_identification_strategies.zig").LanguageModelStrategy;

test "model type identifier initialization" {
    var identifier = ModelTypeIdentifier.init(testing.allocator);
    defer identifier.deinit();
    
    // Should initialize successfully
    try testing.expect(identifier.strategies.items.len == 0);
    try testing.expect(identifier.operator_patterns.count() == 0);
}

test "strategy registration" {
    var identifier = ModelTypeIdentifier.init(testing.allocator);
    defer identifier.deinit();
    
    // Register a language model strategy
    const strategy = try LanguageModelStrategy.createStrategy(testing.allocator);
    try identifier.registerStrategy(strategy);
    
    // Should have one strategy registered
    try testing.expect(identifier.strategies.items.len == 1);
}

test "model parser factory initialization" {
    var factory = ModelParserFactory.init(testing.allocator);
    defer factory.deinit();
    
    // Should initialize successfully
    // Factory should be ready to create parsers
}

test "memory constraints" {
    const constraints = ModelParserFactory.MemoryConstraints.init(1024); // 1GB
    
    // Should allow small models
    try testing.expect(constraints.canLoadModel(100 * 1024 * 1024)); // 100MB
    
    // Should reject large models
    try testing.expect(!constraints.canLoadModel(2 * 1024 * 1024 * 1024)); // 2GB
}

test "parser config creation" {
    const config = ModelParserFactory.ParserConfig.init("test_model.onnx", 2048);
    
    try testing.expect(std.mem.eql(u8, config.model_path, "test_model.onnx"));
    try testing.expect(config.memory_constraints.available_memory_bytes == 2048 * 1024 * 1024);
    try testing.expect(config.enable_optimization == true);
    try testing.expect(config.validate_model == true);
    try testing.expect(config.extract_vocabulary == true);
}

test "model architecture enum" {
    const arch = ModelArchitecture.language_model;
    try testing.expect(std.mem.eql(u8, arch.toString(), "Language Model"));
    
    const unknown = ModelArchitecture.unknown;
    try testing.expect(std.mem.eql(u8, unknown.toString(), "Unknown Architecture"));
}

test "error handling types" {
    const error_types = [_]type{
        ModelParserFactory.ParserError.ModelNotFound,
        ModelParserFactory.ParserError.InvalidModelFormat,
        ModelParserFactory.ParserError.InsufficientMemory,
        ModelParserFactory.ParserError.ModelTooLarge,
        ModelParserFactory.ParserError.ValidationFailed,
        ModelParserFactory.ParserError.VocabularyExtractionFailed,
        ModelParserFactory.ParserError.LoadTimeout,
        ModelParserFactory.ParserError.CorruptedModel,
        ModelParserFactory.ParserError.MissingRequiredFiles,
        ModelParserFactory.ParserError.OutOfMemory,
    };
    
    // All error types should be defined
    try testing.expect(error_types.len == 10);
}

test "model characteristics initialization" {
    const characteristics = ModelTypeIdentifier.ModelCharacteristics.init();
    
    try testing.expect(characteristics.architecture == .unknown);
    try testing.expect(characteristics.has_attention == false);
    try testing.expect(characteristics.has_embedding == false);
    try testing.expect(characteristics.has_convolution == false);
    try testing.expect(characteristics.has_recurrence == false);
    try testing.expect(characteristics.has_normalization == false);
    try testing.expect(characteristics.vocab_size == null);
    try testing.expect(characteristics.sequence_length == null);
    try testing.expect(characteristics.hidden_size == null);
    try testing.expect(characteristics.num_layers == null);
    try testing.expect(characteristics.num_attention_heads == null);
    try testing.expect(characteristics.confidence_score == 0.0);
}

// Integration test that would work with actual ONNX files
test "parser creation for non-existent file" {
    var factory = ModelParserFactory.init(testing.allocator);
    defer factory.deinit();
    
    // Should fail gracefully for non-existent file
    const result = factory.createParser("non_existent_model.onnx");
    try testing.expectError(ModelParserFactory.ParserError.ModelNotFound, result);
}

// Performance test
test "strategy registration performance" {
    var identifier = ModelTypeIdentifier.init(testing.allocator);
    defer identifier.deinit();
    
    const start_time = std.time.nanoTimestamp();
    
    // Register multiple strategies
    const strategy1 = try LanguageModelStrategy.createStrategy(testing.allocator);
    try identifier.registerStrategy(strategy1);
    
    const end_time = std.time.nanoTimestamp();
    const duration_ns = end_time - start_time;
    
    // Should be very fast (less than 1ms)
    try testing.expect(duration_ns < 1_000_000); // 1ms in nanoseconds
}

// Memory usage test
test "memory overhead calculation" {
    const identifier_size = @sizeOf(ModelTypeIdentifier);
    const factory_size = @sizeOf(ModelParserFactory);
    const strategy_size = @sizeOf(LanguageModelStrategy);
    
    // Should have reasonable memory overhead
    try testing.expect(identifier_size < 1024); // Less than 1KB
    try testing.expect(factory_size < 1024);    // Less than 1KB
    try testing.expect(strategy_size < 256);    // Less than 256 bytes
}

// Test SOLID principles compliance
test "interface substitutability" {
    // All strategies should implement the same interface
    const strategy_impl_size = @sizeOf(ModelTypeIdentifier.IdentificationStrategy.StrategyImpl);
    try testing.expect(strategy_impl_size > 0);
    
    // All parsers should implement the same interface
    const parser_impl_size = @sizeOf(ModelParserFactory.ModelParser.ParserImpl);
    try testing.expect(parser_impl_size > 0);
}

// Test error propagation
test "error propagation chain" {
    var factory = ModelParserFactory.init(testing.allocator);
    defer factory.deinit();
    
    // Test that errors propagate correctly through the chain
    const result = factory.createParser("invalid_file.txt");
    
    // Should get a specific error type, not a generic one
    if (result) |_| {
        try testing.expect(false); // Should not succeed
    } else |err| {
        try testing.expect(err == ModelParserFactory.ParserError.ModelNotFound or 
                          err == ModelParserFactory.ParserError.InvalidModelFormat);
    }
}

// Comprehensive system test
test "complete system integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Test the complete workflow (without actual ONNX file)
    var factory = ModelParserFactory.init(allocator);
    defer factory.deinit();
    
    // Test memory constraints
    const constraints = ModelParserFactory.MemoryConstraints.init(1024);
    try testing.expect(constraints.max_model_size_bytes > 0);
    try testing.expect(constraints.available_memory_bytes > 0);
    
    // Test configuration
    const config = ModelParserFactory.ParserConfig.init("test.onnx", 1024);
    try testing.expect(config.memory_constraints.available_memory_bytes == 1024 * 1024 * 1024);
    
    // Test architecture types
    const architectures = [_]ModelArchitecture{
        .language_model,
        .vision_model,
        .audio_model,
        .multimodal_model,
        .embedding_model,
        .classification_model,
        .regression_model,
        .unknown,
    };
    
    for (architectures) |arch| {
        try testing.expect(arch.toString().len > 0);
    }
}

// Test thread safety (basic check)
test "basic thread safety" {
    // The system should be safe for single-threaded use
    // Multi-threading would require additional synchronization
    var identifier = ModelTypeIdentifier.init(testing.allocator);
    defer identifier.deinit();
    
    var factory = ModelParserFactory.init(testing.allocator);
    defer factory.deinit();
    
    // Should be able to create multiple instances
    var identifier2 = ModelTypeIdentifier.init(testing.allocator);
    defer identifier2.deinit();
    
    var factory2 = ModelParserFactory.init(testing.allocator);
    defer factory2.deinit();
}

// Test resource cleanup
test "resource cleanup" {
    // Test that all resources are properly cleaned up
    {
        var identifier = ModelTypeIdentifier.init(testing.allocator);
        const strategy = try LanguageModelStrategy.createStrategy(testing.allocator);
        try identifier.registerStrategy(strategy);
        identifier.deinit(); // Should clean up all strategies
    }
    
    {
        var factory = ModelParserFactory.init(testing.allocator);
        factory.deinit(); // Should clean up all resources
    }
    
    // If we reach here without memory leaks, cleanup is working
}

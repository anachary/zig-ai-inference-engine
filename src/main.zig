const std = @import("std");

// Core abstractions
pub const core = struct {
    pub const Model = @import("core/model.zig").Model;
    pub const Metadata = @import("core/model.zig").Metadata;
    pub const Tokenizer = @import("core/tokenizer.zig").Tokenizer;
    pub const Inference = @import("core/inference.zig").Inference;
    pub const InferenceConfig = @import("core/inference.zig").InferenceConfig;
    pub const InferenceContext = @import("core/inference.zig").InferenceContext;
    pub const Tensor = @import("core/tensor.zig").Tensor;
    pub const DynamicTensor = @import("core/tensor.zig").DynamicTensor;
};

// Format support
pub const formats = struct {
    pub const gguf = @import("formats/gguf/mod.zig");
    pub const registry = @import("formats/mod.zig");
};

// Tokenizers
pub const tokenizers = struct {
    pub const bpe = @import("tokenizers/bpe/mod.zig");
    pub const registry = @import("tokenizers/mod.zig");
};

// Neural network layers
pub const layers = @import("layers/mod.zig");

// Model architectures
pub const models = @import("models/mod.zig");

// Quantization algorithms
pub const quantization = @import("quantization/mod.zig");

// Inference engines
pub const inference = struct {
    pub const transformer = @import("inference/transformer/mod.zig");
    pub const graph = @import("inference/graph/mod.zig");
    pub const model_loader = @import("inference/model_loader.zig");
    pub const real_transformer = @import("inference/real_transformer.zig");
    pub const SimpleChatInference = @import("inference/simple_chat.zig").SimpleChatInference;
    pub const RealGGUFInference = @import("inference/real_gguf_inference.zig").RealGGUFInference;
    pub const ModelLoader = @import("inference/model_loader.zig").ModelLoader;
};

// Week 5: Production Polish & Testing
pub const performance = struct {
    pub const PerformanceOptimizer = @import("performance/optimization.zig").PerformanceOptimizer;
    pub const MemoryPool = @import("performance/optimization.zig").MemoryPool;
    pub const SIMDOps = @import("performance/optimization.zig").SIMDOps;
    pub const BatchProcessor = @import("performance/optimization.zig").BatchProcessor;
    pub const Profiler = @import("performance/optimization.zig").Profiler;
};

pub const test_framework = struct {
    pub const TestFramework = @import("testing/test_framework.zig").TestFramework;
    pub const ModelCompatibilityTests = @import("testing/test_framework.zig").ModelCompatibilityTests;
    pub const BenchmarkTests = @import("testing/test_framework.zig").BenchmarkTests;
    pub const RobustnessTests = @import("testing/test_framework.zig").RobustnessTests;
    pub const IntegrationTests = @import("testing/test_framework.zig").IntegrationTests;
};

pub const benchmarks = struct {
    pub const BenchmarkSuite = @import("benchmarks/benchmark_suite.zig").BenchmarkSuite;
    pub const CoreBenchmarks = @import("benchmarks/benchmark_suite.zig").CoreBenchmarks;
    pub const ModelBenchmarks = @import("benchmarks/benchmark_suite.zig").ModelBenchmarks;
};

pub const error_handling = struct {
    pub const ErrorHandler = @import("error_handling/robust_inference.zig").ErrorHandler;
    pub const InputValidator = @import("error_handling/robust_inference.zig").InputValidator;
    pub const MemoryMonitor = @import("error_handling/robust_inference.zig").MemoryMonitor;
};

// Compute backends
pub const compute = @import("compute/mod.zig");

// Memory management
pub const memory = @import("memory/mod.zig");

// Mathematical operations
pub const math = @import("math/mod.zig");

// Neural network operators
pub const operators = @import("operators/mod.zig");

// Note: CLI is a separate executable, not part of the library

// Utilities
pub const utils = @import("utils/mod.zig");

// Error types - include all possible errors from file operations and GGUF parsing
pub const Error = error{
    UnsupportedFormat,
    CorruptedFile,
    InsufficientMemory,
    InvalidModel,
    TokenizationFailed,
    InferenceFailed,
    OutOfMemory,
    FileNotFound,
    CannotOpenFile,
    AccessDenied,
    InputOutput,
    Unexpected,
    BrokenPipe,
    SystemResources,
    OperationAborted,
    WouldBlock,
    ConnectionResetByPeer,
    IsDir,
    ConnectionTimedOut,
    NotOpenForReading,
    NetNameDeleted,
    SystemFdQuotaExceeded,
    NoDevice,
    NotDir,
    FileLocksNotSupported,
    FileBusy,
    InvalidMagic,
    UnsupportedVersion,
    Unseekable,
    UnsupportedDataType,
};

// C-compatible API exports for DLL
export fn zig_ai_get_version() callconv(.C) [*:0]const u8 {
    return version.string;
}

export fn zig_ai_detect_format(path: [*:0]const u8) callconv(.C) c_int {
    const zig_path = std.mem.span(path);
    const format = formats.registry.detectFormat(zig_path) catch return -1;
    return switch (format) {
        .gguf => 0,
        .onnx => 1,
        .safetensors => 2,
        .pytorch => 3,
        .tensorflow => 4,
        .huggingface => 5,
        .mlx => 6,
        .coreml => 7,
        else => -1,
    };
}

// Zig API for loading models (for Zig consumers)
pub fn loadModel(allocator: std.mem.Allocator, path: []const u8) anyerror!core.Model {
    const format = try formats.registry.detectFormat(path);
    return switch (format) {
        .gguf => formats.gguf.load(allocator, path),
        else => error.UnsupportedFormat,
    };
}

// Zig API for creating tokenizers (for Zig consumers)
pub fn createTokenizer(allocator: std.mem.Allocator, tokenizer_type: tokenizers.registry.TokenizerType, vocab_path: []const u8) anyerror!core.Tokenizer {
    return switch (tokenizer_type) {
        .bpe => tokenizers.bpe.create(allocator, vocab_path),
        else => error.UnsupportedFormat,
    };
}

// Version information
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;
    pub const string = "0.1.0";
};

test "main module imports" {
    const testing = std.testing;
    _ = testing;
    // Basic smoke test to ensure all modules can be imported
}

const std = @import("std");

/// Zig Inference Engine - High-performance neural network inference and execution engine
///
/// This library provides a focused, high-performance inference engine following the
/// Single Responsibility Principle. It handles only model execution, operator
/// implementation, and inference scheduling.
///
/// Key Features:
/// - 25+ optimized operators (arithmetic, matrix, convolution, activation, etc.)
/// - Multi-threaded task scheduling and execution
/// - GPU acceleration backends (CUDA, Vulkan, OpenCL)
/// - Memory-efficient execution planning
/// - Interface-based architecture for extensibility
///
/// Dependencies:
/// - zig-tensor-core: For tensor operations and memory management
/// - zig-onnx-parser: For model parsing and validation
///
/// Usage:
/// ```zig
/// const std = @import("std");
/// const inference_engine = @import("zig-inference-engine");
/// const onnx_parser = @import("zig-onnx-parser");
/// const tensor_core = @import("zig-tensor-core");
///
/// pub fn main() !void {
///     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
///     defer _ = gpa.deinit();
///     const allocator = gpa.allocator();
///
///     // Initialize inference engine
///     const config = inference_engine.Config{
///         .device_type = .auto,
///         .num_threads = 4,
///         .enable_gpu = true,
///         .optimization_level = .balanced,
///     };
///
///     var engine = try inference_engine.Engine.init(allocator, config);
///     defer engine.deinit();
///
///     // Load model (parsed by zig-onnx-parser)
///     var parser = onnx_parser.Parser.init(allocator);
///     const model = try parser.parseFile("model.onnx");
///     defer model.deinit();
///
///     try engine.loadModel(model.ptr, model.interface);
///
///     // Run inference
///     const outputs = try engine.infer(&[_]tensor_core.Tensor{input});
///     defer allocator.free(outputs);
/// }
/// ```

// Re-export core engine components
pub const Engine = @import("engine/engine.zig").Engine;
pub const Config = @import("engine/engine.zig").Config;
pub const Stats = @import("engine/engine.zig").Stats;
pub const EngineError = @import("engine/engine.zig").EngineError;
pub const DeviceType = @import("engine/engine.zig").DeviceType;
pub const GPUBackendType = @import("engine/engine.zig").GPUBackendType;
pub const OptimizationLevel = @import("engine/engine.zig").OptimizationLevel;
pub const Precision = @import("engine/engine.zig").Precision;

// Re-export operator registry
pub const OperatorRegistry = @import("operators/registry.zig").OperatorRegistry;
pub const OperatorInfo = @import("operators/registry.zig").OperatorInfo;
pub const OperatorFn = @import("operators/registry.zig").OperatorFn;
pub const ValidatorFn = @import("operators/registry.zig").ValidatorFn;
pub const RegistryError = @import("operators/registry.zig").RegistryError;
pub const RegistryStats = @import("operators/registry.zig").RegistryStats;

// Re-export shape inference
pub const ShapeInference = @import("engine/shape_inference.zig").ShapeInference;

// Re-export task scheduler
pub const TaskScheduler = @import("scheduler/scheduler.zig").TaskScheduler;
pub const Task = @import("scheduler/scheduler.zig").Task;
pub const TaskContext = @import("scheduler/scheduler.zig").TaskContext;
pub const TaskType = @import("scheduler/scheduler.zig").TaskType;
pub const Priority = @import("scheduler/scheduler.zig").Priority;
pub const TaskStatus = @import("scheduler/scheduler.zig").TaskStatus;
pub const SchedulerStats = @import("scheduler/scheduler.zig").SchedulerStats;

// Re-export GPU backend
pub const GPUBackend = @import("gpu/backend.zig").GPUBackend;
pub const BackendType = @import("gpu/backend.zig").BackendType;
pub const BackendError = @import("gpu/backend.zig").BackendError;
pub const Kernel = @import("gpu/backend.zig").Kernel;
pub const ExecutionContext = @import("gpu/backend.zig").ExecutionContext;
pub const BackendStats = @import("gpu/backend.zig").BackendStats;

// Re-export operator modules
pub const operators = struct {
    pub const arithmetic = @import("operators/arithmetic.zig");
    pub const matrix = @import("operators/matrix.zig");
    pub const activation = @import("operators/activation.zig");
    pub const conv = @import("operators/conv.zig");
    pub const pooling = @import("operators/pooling.zig");
    pub const normalization = @import("operators/normalization.zig");
    pub const reduction = @import("operators/reduction.zig");
    pub const shape = @import("operators/shape.zig");
};

// Re-export tokenizer
pub const SimpleTokenizer = @import("tokenizer/simple_tokenizer.zig").SimpleTokenizer;

// Import common interfaces for re-export
const common_interfaces = @import("common-interfaces");
pub const TensorInterface = common_interfaces.TensorInterface;

// Re-export engine interfaces
const engine = @import("engine/engine.zig");
pub const ModelInterface = engine.ModelInterface;
pub const ModelImpl = engine.ModelImpl;
pub const ModelMetadata = engine.ModelMetadata;

/// Library version information
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;
    pub const string = "0.1.0";
};

/// Library information
pub const info = struct {
    pub const name = "zig-inference-engine";
    pub const description = "High-performance neural network inference and execution engine";
    pub const author = "Zig AI Ecosystem";
    pub const license = "MIT";
    pub const repository = "https://github.com/zig-ai/zig-inference-engine";
};

/// Supported operator types
pub const SupportedOperators = enum {
    // Arithmetic operators
    add,
    sub,
    mul,
    div,
    pow,
    sqrt,
    exp,
    log,

    // Matrix operations
    matmul,
    gemm,
    transpose,

    // Convolution operators
    conv2d,
    conv3d,
    depthwise_conv2d,

    // Activation functions
    relu,
    sigmoid,
    tanh,
    softmax,
    gelu,
    swish,

    // Pooling operators
    max_pool,
    avg_pool,
    global_avg_pool,

    // Normalization operators
    batch_norm,
    layer_norm,
    instance_norm,

    // Reduction operations
    reduce_sum,
    reduce_mean,
    reduce_max,
    reduce_min,

    // Shape manipulation
    concat,
    split,
    slice,
    squeeze,
    unsqueeze,
    gather,
    scatter,
};

/// Get list of all supported operators
pub fn getSupportedOperators() []const SupportedOperators {
    return std.meta.fields(SupportedOperators);
}

/// Check if an operator is supported
pub fn isOperatorSupported(op_name: []const u8) bool {
    inline for (std.meta.fields(SupportedOperators)) |field| {
        if (std.mem.eql(u8, op_name, field.name)) {
            return true;
        }
    }
    return false;
}

/// Initialize a default inference engine configuration
pub fn defaultConfig() Config {
    return Config{
        .device_type = .auto,
        .num_threads = null, // Auto-detect
        .enable_gpu = true,
        .gpu_backend = .auto,
        .optimization_level = .balanced,
        .precision = .fp32,
        .max_batch_size = 1,
        .max_sequence_length = 2048,
        .enable_profiling = false,
        .memory_limit_mb = null,
        .enable_operator_fusion = true,
        .enable_memory_optimization = true,
    };
}

/// Create a configuration optimized for IoT devices
pub fn iotConfig() Config {
    return Config{
        .device_type = .cpu,
        .num_threads = 1,
        .enable_gpu = false,
        .gpu_backend = .auto,
        .optimization_level = .aggressive,
        .precision = .fp16,
        .max_batch_size = 1,
        .max_sequence_length = 512,
        .enable_profiling = false,
        .memory_limit_mb = 64, // 64MB limit for IoT
        .enable_operator_fusion = true,
        .enable_memory_optimization = true,
    };
}

/// Create a configuration optimized for desktop applications
pub fn desktopConfig() Config {
    return Config{
        .device_type = .auto,
        .num_threads = null, // Auto-detect
        .enable_gpu = true,
        .gpu_backend = .auto,
        .optimization_level = .balanced,
        .precision = .fp32,
        .max_batch_size = 4,
        .max_sequence_length = 2048,
        .enable_profiling = false,
        .memory_limit_mb = 2048, // 2GB limit
        .enable_operator_fusion = true,
        .enable_memory_optimization = true,
    };
}

/// Create a configuration optimized for server deployment
pub fn serverConfig() Config {
    return Config{
        .device_type = .auto,
        .num_threads = null, // Auto-detect
        .enable_gpu = true,
        .gpu_backend = .auto,
        .optimization_level = .max,
        .precision = .mixed,
        .max_batch_size = 32,
        .max_sequence_length = 4096,
        .enable_profiling = true,
        .memory_limit_mb = null, // No limit
        .enable_operator_fusion = true,
        .enable_memory_optimization = true,
    };
}

/// Utility function to create an engine with default configuration
pub fn createEngine(allocator: std.mem.Allocator) !Engine {
    return Engine.init(allocator, defaultConfig());
}

/// Utility function to create an IoT-optimized engine
pub fn createIoTEngine(allocator: std.mem.Allocator) !Engine {
    return Engine.init(allocator, iotConfig());
}

/// Utility function to create a desktop-optimized engine
pub fn createDesktopEngine(allocator: std.mem.Allocator) !Engine {
    return Engine.init(allocator, desktopConfig());
}

/// Utility function to create a server-optimized engine
pub fn createServerEngine(allocator: std.mem.Allocator) !Engine {
    return Engine.init(allocator, serverConfig());
}

/// Library initialization function (optional)
pub fn init() void {
    std.log.info("Zig Inference Engine v{s} initialized", .{version.string});
}

/// Library cleanup function (optional)
pub fn deinit() void {
    std.log.info("Zig Inference Engine v{s} deinitialized", .{version.string});
}

/// Test function to verify library functionality
pub fn test_basic_functionality() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test engine creation
    var test_engine = try createEngine(allocator);
    defer test_engine.deinit();

    // Test operator registry
    const op_count = test_engine.operator_registry.getOperatorCount();
    std.log.info("Registered {} operators", .{op_count});

    // Test scheduler
    const scheduler_stats = engine.task_scheduler.getStats();
    std.log.info("Scheduler initialized with {} workers", .{scheduler_stats.worker_utilization});

    std.log.info("Basic functionality test passed");
}

// Tests
test "library initialization" {
    init();
    deinit();
}

test "configuration creation" {
    const default_cfg = defaultConfig();
    const iot_cfg = iotConfig();
    const desktop_cfg = desktopConfig();
    const server_cfg = serverConfig();

    // Basic validation
    try std.testing.expect(default_cfg.optimization_level == .balanced);
    try std.testing.expect(iot_cfg.device_type == .cpu);
    try std.testing.expect(desktop_cfg.enable_gpu == true);
    try std.testing.expect(server_cfg.max_batch_size == 32);
}

test "operator support check" {
    try std.testing.expect(isOperatorSupported("add"));
    try std.testing.expect(isOperatorSupported("relu"));
    try std.testing.expect(isOperatorSupported("matmul"));
    try std.testing.expect(!isOperatorSupported("nonexistent_op"));
}

test "engine creation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var test_engine = try createEngine(allocator);
    defer test_engine.deinit();

    const stats = test_engine.getStats();
    try std.testing.expect(!stats.model_loaded);
    try std.testing.expect(stats.total_inferences == 0);
}

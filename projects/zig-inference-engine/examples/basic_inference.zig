const std = @import("std");
const inference_engine = @import("zig-inference-engine");

/// Basic inference example demonstrating the core functionality
/// of the Zig Inference Engine
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== Zig Inference Engine - Basic Inference Example ===", .{});

    // Initialize the inference engine with default configuration
    std.log.info("Initializing inference engine...", .{});
    var engine = try inference_engine.createEngine(allocator);
    defer engine.deinit();

    std.log.info("Engine initialized successfully!", .{});
    
    // Display engine configuration
    const config = engine.config;
    std.log.info("Configuration:", .{});
    std.log.info("  Device Type: {}", .{config.device_type});
    std.log.info("  Optimization Level: {}", .{config.optimization_level});
    std.log.info("  Precision: {}", .{config.precision});
    std.log.info("  Max Batch Size: {}", .{config.max_batch_size});
    std.log.info("  Enable GPU: {}", .{config.enable_gpu});

    // Display operator registry information
    const op_count = engine.operator_registry.getOperatorCount();
    std.log.info("Operator Registry:", .{});
    std.log.info("  Total Operators: {}", .{op_count});

    // List some supported operators
    std.log.info("  Supported Operators:");
    const supported_ops = [_][]const u8{
        "Add", "Sub", "Mul", "Div", "MatMul", "ReLU", "Sigmoid", "Softmax"
    };
    
    for (supported_ops) |op_name| {
        const has_op = engine.operator_registry.hasOperator(op_name);
        std.log.info("    {s}: {}", .{ op_name, has_op });
    }

    // Display scheduler information
    const scheduler_stats = engine.task_scheduler.getStats();
    std.log.info("Task Scheduler:", .{});
    std.log.info("  Total Tasks: {}", .{scheduler_stats.total_tasks});
    std.log.info("  Completed Tasks: {}", .{scheduler_stats.completed_tasks});
    std.log.info("  Worker Utilization: {d:.2}%", .{scheduler_stats.worker_utilization * 100});

    // Display GPU backend information
    if (engine.gpu_backend) |*backend| {
        const backend_stats = backend.getStats();
        std.log.info("GPU Backend:", .{});
        std.log.info("  Backend Type: {}", .{backend_stats.backend_type});
        std.log.info("  Device Count: {}", .{backend_stats.device_count});
        std.log.info("  Kernels Compiled: {}", .{backend_stats.kernels_compiled});
    } else {
        std.log.info("GPU Backend: Not available", .{});
    }

    // Display current engine statistics
    const stats = engine.getStats();
    std.log.info("Engine Statistics:", .{});
    std.log.info("  Model Loaded: {}", .{stats.model_loaded});
    std.log.info("  Total Inferences: {}", .{stats.total_inferences});
    std.log.info("  Average Latency: {d:.2}ms", .{stats.average_latency_ms});
    std.log.info("  Peak Memory: {}MB", .{stats.peak_memory_mb});

    // Demonstrate operator execution (without a loaded model)
    std.log.info("\n=== Operator Execution Demo ===", .{});
    
    // Note: In a real implementation, we would need actual tensor instances
    // from zig-tensor-core. For this demo, we'll show the operator registry functionality.
    
    if (engine.operator_registry.getOperator("Add")) |add_op| {
        std.log.info("Add Operator Info:", .{});
        std.log.info("  Name: {s}", .{add_op.name});
        std.log.info("  Description: {s}", .{add_op.description});
        std.log.info("  Min Inputs: {}", .{add_op.min_inputs});
        std.log.info("  Max Inputs: {}", .{add_op.max_inputs});
        std.log.info("  Supports Inplace: {}", .{add_op.supports_inplace});
        std.log.info("  Supports Broadcasting: {}", .{add_op.supports_broadcasting});
    }

    if (engine.operator_registry.getOperator("ReLU")) |relu_op| {
        std.log.info("ReLU Operator Info:", .{});
        std.log.info("  Name: {s}", .{relu_op.name});
        std.log.info("  Description: {s}", .{relu_op.description});
        std.log.info("  Min Inputs: {}", .{relu_op.min_inputs});
        std.log.info("  Max Inputs: {}", .{relu_op.max_inputs});
        std.log.info("  Supports Inplace: {}", .{relu_op.supports_inplace});
    }

    // Demonstrate configuration variants
    std.log.info("\n=== Configuration Variants Demo ===", .{});
    
    // IoT configuration
    const iot_config = inference_engine.iotConfig();
    std.log.info("IoT Configuration:", .{});
    std.log.info("  Device Type: {}", .{iot_config.device_type});
    std.log.info("  Memory Limit: {}MB", .{iot_config.memory_limit_mb.?});
    std.log.info("  Precision: {}", .{iot_config.precision});
    std.log.info("  Max Batch Size: {}", .{iot_config.max_batch_size});

    // Desktop configuration
    const desktop_config = inference_engine.desktopConfig();
    std.log.info("Desktop Configuration:", .{});
    std.log.info("  Device Type: {}", .{desktop_config.device_type});
    std.log.info("  Enable GPU: {}", .{desktop_config.enable_gpu});
    std.log.info("  Memory Limit: {}MB", .{desktop_config.memory_limit_mb.?});
    std.log.info("  Max Batch Size: {}", .{desktop_config.max_batch_size});

    // Server configuration
    const server_config = inference_engine.serverConfig();
    std.log.info("Server Configuration:", .{});
    std.log.info("  Optimization Level: {}", .{server_config.optimization_level});
    std.log.info("  Precision: {}", .{server_config.precision});
    std.log.info("  Max Batch Size: {}", .{server_config.max_batch_size});
    std.log.info("  Enable Profiling: {}", .{server_config.enable_profiling});

    // Test operator support checking
    std.log.info("\n=== Operator Support Check ===", .{});
    const test_operators = [_][]const u8{
        "Add", "MatMul", "Conv2D", "BatchNorm", "NonExistentOp"
    };
    
    for (test_operators) |op_name| {
        const supported = inference_engine.isOperatorSupported(op_name);
        std.log.info("  {s}: {}", .{ op_name, supported });
    }

    // Display library information
    std.log.info("\n=== Library Information ===", .{});
    std.log.info("Name: {s}", .{inference_engine.info.name});
    std.log.info("Version: {s}", .{inference_engine.version.string});
    std.log.info("Description: {s}", .{inference_engine.info.description});
    std.log.info("Author: {s}", .{inference_engine.info.author});
    std.log.info("License: {s}", .{inference_engine.info.license});

    // Demonstrate engine statistics reset
    std.log.info("\n=== Statistics Reset Demo ===", .{});
    std.log.info("Before reset - Total Inferences: {}", .{engine.getStats().total_inferences});
    engine.resetStats();
    std.log.info("After reset - Total Inferences: {}", .{engine.getStats().total_inferences});

    std.log.info("\n=== Basic Inference Example Complete ===", .{});
    std.log.info("The inference engine is ready for model loading and inference!", .{});
    std.log.info("Next steps:", .{});
    std.log.info("  1. Use zig-onnx-parser to load a model", .{});
    std.log.info("  2. Call engine.loadModel() with the parsed model", .{});
    std.log.info("  3. Create input tensors using zig-tensor-core", .{});
    std.log.info("  4. Call engine.infer() to run inference", .{});
    std.log.info("  5. Process the output tensors", .{});
}

/// Test function for the basic inference example
pub fn test_basic_inference() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test engine creation
    var engine = try inference_engine.createEngine(allocator);
    defer engine.deinit();

    // Verify engine is properly initialized
    const stats = engine.getStats();
    try std.testing.expect(!stats.model_loaded);
    try std.testing.expect(stats.total_inferences == 0);

    // Test operator registry
    const op_count = engine.operator_registry.getOperatorCount();
    try std.testing.expect(op_count > 0);

    // Test specific operators
    try std.testing.expect(engine.operator_registry.hasOperator("Add"));
    try std.testing.expect(engine.operator_registry.hasOperator("ReLU"));
    try std.testing.expect(!engine.operator_registry.hasOperator("NonExistentOp"));

    // Test configuration variants
    const iot_config = inference_engine.iotConfig();
    try std.testing.expect(iot_config.device_type == .cpu);
    try std.testing.expect(iot_config.precision == .fp16);

    const server_config = inference_engine.serverConfig();
    try std.testing.expect(server_config.optimization_level == .max);
    try std.testing.expect(server_config.max_batch_size == 32);

    std.log.info("Basic inference test passed!", .{});
}

test "basic inference functionality" {
    try test_basic_inference();
}

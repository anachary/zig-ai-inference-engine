const std = @import("std");
const inference_engine = @import("zig-inference-engine");

/// Integration tests for the Zig Inference Engine
/// These tests verify that all components work together correctly

test "engine initialization and cleanup" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test default engine creation
    var engine = try inference_engine.createEngine(allocator);
    defer engine.deinit();

    // Verify initial state
    const stats = engine.getStats();
    try std.testing.expect(!stats.model_loaded);
    try std.testing.expect(stats.total_inferences == 0);
    try std.testing.expect(stats.average_latency_ms == 0.0);
}

test "operator registry functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var engine = try inference_engine.createEngine(allocator);
    defer engine.deinit();

    // Test operator count
    const op_count = engine.operator_registry.getOperatorCount();
    try std.testing.expect(op_count > 0);

    // Test specific operators exist
    try std.testing.expect(engine.operator_registry.hasOperator("Add"));
    try std.testing.expect(engine.operator_registry.hasOperator("Sub"));
    try std.testing.expect(engine.operator_registry.hasOperator("Mul"));
    try std.testing.expect(engine.operator_registry.hasOperator("Div"));
    try std.testing.expect(engine.operator_registry.hasOperator("MatMul"));
    try std.testing.expect(engine.operator_registry.hasOperator("ReLU"));
    try std.testing.expect(engine.operator_registry.hasOperator("Sigmoid"));
    try std.testing.expect(engine.operator_registry.hasOperator("Softmax"));

    // Test non-existent operator
    try std.testing.expect(!engine.operator_registry.hasOperator("NonExistentOperator"));

    // Test operator information retrieval
    if (engine.operator_registry.getOperator("Add")) |add_op| {
        try std.testing.expectEqualStrings("Add", add_op.name);
        try std.testing.expect(add_op.min_inputs == 2);
        try std.testing.expect(add_op.max_inputs == 2);
        try std.testing.expect(add_op.min_outputs == 1);
        try std.testing.expect(add_op.max_outputs == 1);
        try std.testing.expect(add_op.supports_inplace == true);
        try std.testing.expect(add_op.supports_broadcasting == true);
    } else {
        try std.testing.expect(false); // Add operator should exist
    }
}

test "task scheduler functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var engine = try inference_engine.createEngine(allocator);
    defer engine.deinit();

    // Test initial scheduler state
    const initial_stats = engine.task_scheduler.getStats();
    try std.testing.expect(initial_stats.total_tasks == 0);
    try std.testing.expect(initial_stats.completed_tasks == 0);
    try std.testing.expect(initial_stats.failed_tasks == 0);
    try std.testing.expect(initial_stats.pending_tasks == 0);
    try std.testing.expect(initial_stats.running_tasks == 0);
}

test "gpu backend initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var engine = try inference_engine.createEngine(allocator);
    defer engine.deinit();

    // GPU backend may or may not be available depending on the system
    // We just test that the engine handles it gracefully
    if (engine.gpu_backend) |*backend| {
        const stats = backend.getStats();
        // Basic validation that stats are initialized
        try std.testing.expect(stats.operations_executed == 0);
        try std.testing.expect(stats.kernels_compiled == 0);
    }
    // If no GPU backend, that's also fine
}

test "configuration variants" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test IoT configuration
    {
        var iot_engine = try inference_engine.createIoTEngine(allocator);
        defer iot_engine.deinit();

        const config = iot_engine.config;
        try std.testing.expect(config.device_type == .cpu);
        try std.testing.expect(config.enable_gpu == false);
        try std.testing.expect(config.precision == .fp16);
        try std.testing.expect(config.max_batch_size == 1);
        try std.testing.expect(config.memory_limit_mb.? == 64);
    }

    // Test desktop configuration
    {
        var desktop_engine = try inference_engine.createDesktopEngine(allocator);
        defer desktop_engine.deinit();

        const config = desktop_engine.config;
        try std.testing.expect(config.device_type == .auto);
        try std.testing.expect(config.enable_gpu == true);
        try std.testing.expect(config.precision == .fp32);
        try std.testing.expect(config.max_batch_size == 4);
        try std.testing.expect(config.memory_limit_mb.? == 2048);
    }

    // Test server configuration
    {
        var server_engine = try inference_engine.createServerEngine(allocator);
        defer server_engine.deinit();

        const config = server_engine.config;
        try std.testing.expect(config.device_type == .auto);
        try std.testing.expect(config.enable_gpu == true);
        try std.testing.expect(config.precision == .mixed);
        try std.testing.expect(config.max_batch_size == 32);
        try std.testing.expect(config.optimization_level == .max);
        try std.testing.expect(config.enable_profiling == true);
    }
}

test "statistics tracking" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var engine = try inference_engine.createEngine(allocator);
    defer engine.deinit();

    // Test initial statistics
    const initial_stats = engine.getStats();
    try std.testing.expect(initial_stats.total_inferences == 0);
    try std.testing.expect(initial_stats.average_latency_ms == 0.0);
    try std.testing.expect(initial_stats.peak_memory_mb == 0);
    try std.testing.expect(!initial_stats.model_loaded);

    // Test statistics reset
    engine.resetStats();
    const reset_stats = engine.getStats();
    try std.testing.expect(reset_stats.total_inferences == 0);
    try std.testing.expect(reset_stats.average_latency_ms == 0.0);
    try std.testing.expect(!reset_stats.model_loaded);
}

test "operator support checking" {
    // Test the global operator support function
    try std.testing.expect(inference_engine.isOperatorSupported("Add"));
    try std.testing.expect(inference_engine.isOperatorSupported("MatMul"));
    try std.testing.expect(inference_engine.isOperatorSupported("ReLU"));
    try std.testing.expect(inference_engine.isOperatorSupported("Conv2D"));
    try std.testing.expect(!inference_engine.isOperatorSupported("NonExistentOp"));
    try std.testing.expect(!inference_engine.isOperatorSupported(""));
}

test "library information" {
    // Test version information
    try std.testing.expect(inference_engine.version.major == 0);
    try std.testing.expect(inference_engine.version.minor == 1);
    try std.testing.expect(inference_engine.version.patch == 0);
    try std.testing.expectEqualStrings("0.1.0", inference_engine.version.string);

    // Test library information
    try std.testing.expectEqualStrings("zig-inference-engine", inference_engine.info.name);
    try std.testing.expectEqualStrings("MIT", inference_engine.info.license);
}

test "memory management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test multiple engine creation and cleanup
    for (0..5) |_| {
        var engine = try inference_engine.createEngine(allocator);
        defer engine.deinit();

        // Verify each engine is properly initialized
        const stats = engine.getStats();
        try std.testing.expect(!stats.model_loaded);
        try std.testing.expect(stats.total_inferences == 0);
    }
}

test "concurrent engine usage" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test that multiple engines can coexist
    var engine1 = try inference_engine.createEngine(allocator);
    defer engine1.deinit();

    var engine2 = try inference_engine.createIoTEngine(allocator);
    defer engine2.deinit();

    var engine3 = try inference_engine.createServerEngine(allocator);
    defer engine3.deinit();

    // Verify they have different configurations
    try std.testing.expect(engine1.config.device_type == .auto);
    try std.testing.expect(engine2.config.device_type == .cpu);
    try std.testing.expect(engine3.config.max_batch_size == 32);

    // Verify they all have operators registered
    try std.testing.expect(engine1.operator_registry.getOperatorCount() > 0);
    try std.testing.expect(engine2.operator_registry.getOperatorCount() > 0);
    try std.testing.expect(engine3.operator_registry.getOperatorCount() > 0);
}

test "error handling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var engine = try inference_engine.createEngine(allocator);
    defer engine.deinit();

    // Test inference without loaded model should fail
    const empty_inputs: []const inference_engine.TensorInterface = &[_]inference_engine.TensorInterface{};
    const result = engine.infer(empty_inputs);
    try std.testing.expectError(inference_engine.EngineError.ModelNotLoaded, result);
}

test "basic functionality test" {
    // Run the basic functionality test from the library
    try inference_engine.test_basic_functionality();
}

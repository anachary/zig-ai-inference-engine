const std = @import("std");
const lib = @import("zig-ai-engine");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧮 Computation Graph Example - Phase 2 Implementation", .{});
    std.log.info("====================================================", .{});

    // Initialize the AI engine
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 1024,
        .num_threads = 4,
        .enable_profiling = true,
    });
    defer engine.deinit();

    std.log.info("✅ AI Engine initialized with profiling enabled", .{});

    // Create a simple computation graph manually
    std.log.info("", .{});
    std.log.info("🏗️ Creating computation graph...", .{});

    var metadata = try lib.formats.ModelMetadata.init(allocator, "test_graph", "1.0");
    metadata.format = .onnx;

    var test_model = lib.formats.Model.init(allocator, metadata);
    defer test_model.deinit();

    // Define input and output specs first
    const input_shape = [_]i32{ 2, 3 }; // 2x3 matrix
    var input_spec = try lib.formats.TensorSpec.init(allocator, "input", &input_shape, .f32);
    try test_model.graph.addInput(input_spec);

    const output_shape = [_]i32{ 2, 3 }; // Same shape as input
    var output_spec = try lib.formats.TensorSpec.init(allocator, "output", &output_shape, .f32);
    try test_model.graph.addOutput(output_spec);

    // Create a simple graph: Input -> ReLU -> Output (simplified to avoid Add complexity)
    // ReLU node: applies ReLU activation with proper input/output connections
    var relu_node = try lib.formats.GraphNode.init(allocator, "relu_1", "Relu");

    // Set up the node's inputs and outputs to connect to the graph's input/output specs
    relu_node.inputs = try allocator.alloc([]const u8, 1);
    relu_node.inputs[0] = try allocator.dupe(u8, "input");

    relu_node.outputs = try allocator.alloc([]const u8, 1);
    relu_node.outputs[0] = try allocator.dupe(u8, "output");

    try test_model.graph.addNode(relu_node);

    std.log.info("✅ Created computation graph with:", .{});
    std.log.info("  - {d} nodes: ReLU", .{test_model.graph.nodes.items.len});
    std.log.info("  - {d} edges", .{test_model.graph.edges.items.len});
    std.log.info("  - {d} inputs", .{test_model.graph.inputs.items.len});
    std.log.info("  - {d} outputs", .{test_model.graph.outputs.items.len});

    // Validate the graph
    std.log.info("", .{});
    std.log.info("✅ Validating computation graph...", .{});
    try test_model.validate();
    std.log.info("✅ Graph validation passed", .{});

    // Test graph optimization
    std.log.info("", .{});
    std.log.info("⚡ Testing graph optimization...", .{});

    var optimizer = lib.optimizer.GraphOptimizer.init(allocator, .{
        .enable_fusion = true,
        .enable_dead_code_elimination = true,
        .enable_constant_folding = true,
    });

    const opt_result = try optimizer.optimize(&test_model.graph);
    opt_result.print();

    // Test graph execution
    std.log.info("", .{});
    std.log.info("🚀 Testing graph execution...", .{});

    // Create model executor
    var model_executor = lib.executor.ModelExecutor.init(allocator, &engine.operator_registry, true);

    // Load the model
    try model_executor.loadModel(&test_model);

    // Create test input tensor
    const input_tensor_shape = [_]usize{ 2, 3 };
    var input_tensor = try lib.tensor.Tensor.init(allocator, &input_tensor_shape, .f32);
    defer input_tensor.deinit();

    // Fill input with test data
    var i: usize = 0;
    while (i < input_tensor.numel()) : (i += 1) {
        try input_tensor.set_f32_flat(i, @as(f32, @floatFromInt(i)) - 2.0); // Values: -2, -1, 0, 1, 2, 3
    }

    std.log.info("📊 Input tensor data:", .{});
    i = 0;
    while (i < input_tensor.numel()) : (i += 1) {
        const val = try input_tensor.get_f32_flat(i);
        std.log.info("  input[{d}] = {d:.1}", .{ i, val });
    }

    // Execute the graph
    var inputs = [_]lib.tensor.Tensor{input_tensor};
    const outputs = try model_executor.execute(inputs[0..]);

    // Clean up outputs properly - these are new tensors created by the executor
    defer {
        for (outputs) |output| {
            var mutable_output = output;
            mutable_output.deinit();
        }
        allocator.free(outputs);
    }

    // Clean up model executor before input tensor to avoid double-free
    defer model_executor.deinit();

    std.log.info("", .{});
    std.log.info("📊 Output tensor data:", .{});
    if (outputs.len > 0) {
        i = 0;
        while (i < outputs[0].numel()) : (i += 1) {
            const val = try outputs[0].get_f32_flat(i);
            std.log.info("  output[{d}] = {d:.1}", .{ i, val });
        }
    }

    // Print execution statistics
    if (model_executor.getExecutionStats()) |stats| {
        std.log.info("", .{});
        stats.print();
    }

    // Test benchmarking
    std.log.info("", .{});
    std.log.info("⏱️ Running benchmark (10 iterations)...", .{});

    const benchmark_result = try model_executor.benchmark(inputs[0..], 10);
    benchmark_result.print();

    std.log.info("", .{});
    std.log.info("🎊 Computation Graph Example Complete!", .{});
    std.log.info("✅ Graph creation and validation working", .{});
    std.log.info("✅ Graph optimization pipeline functional", .{});
    std.log.info("✅ Graph execution engine operational", .{});
    std.log.info("✅ Performance profiling and benchmarking ready", .{});
    std.log.info("", .{});
    std.log.info("🚀 Ready for complex model execution!", .{});
}

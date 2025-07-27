const std = @import("std");
const framework = @import("framework");
const implementations = @import("implementations");

/// Comprehensive demo of the new framework capabilities
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 Zig AI Platform Framework Demo");
    std.log.info("=====================================");

    // Demo 1: Basic Platform Initialization
    try demoBasicPlatform(allocator);
    
    // Demo 2: Operator Registry
    try demoOperatorRegistry(allocator);
    
    // Demo 3: Tensor Operations
    try demoTensorOperations(allocator);
    
    // Demo 4: Model-Specific Components
    try demoModelComponents(allocator);
    
    // Demo 5: Custom Operator
    try demoCustomOperator(allocator);
    
    // Demo 6: Performance Benchmarking
    try demoBenchmarking(allocator);

    std.log.info("✅ All demos completed successfully!");
}

fn demoBasicPlatform(allocator: std.mem.Allocator) !void {
    std.log.info("\n📋 Demo 1: Basic Platform Initialization");
    std.log.info("----------------------------------------");

    // Create platform with default configuration
    var platform = try implementations.utils.createDefaultPlatform(allocator);
    defer platform.deinit();

    // Get platform statistics
    const stats = platform.getStats();
    std.log.info("✓ Platform initialized successfully");
    std.log.info("  - Total operators: {}", .{stats.total_operators});
    std.log.info("  - Transformer support: {}", .{stats.transformer_operators_enabled});
    std.log.info("  - Memory usage: {} bytes", .{stats.framework_stats.total_memory_used});

    // List available operator categories
    const categories = platform.getOperatorCategories();
    std.log.info("  - Available categories: {}", .{categories.len});
    for (categories) |category| {
        const ops = platform.getOperatorsByCategory(category);
        std.log.info("    * {s}: {} operators", .{ @tagName(category), ops.len });
    }
}

fn demoOperatorRegistry(allocator: std.mem.Allocator) !void {
    std.log.info("\n🔧 Demo 2: Operator Registry");
    std.log.info("-----------------------------");

    // Create registry and register operators
    var registry = try implementations.operators.createBuiltinRegistry(allocator);
    defer registry.deinit();

    // Test operator lookup
    const operators_to_test = [_][]const u8{ "Add", "Relu", "MatMul", "LayerNormalization" };
    
    for (operators_to_test) |op_name| {
        if (registry.hasOperator(op_name, null)) {
            std.log.info("✓ Operator '{}' is available", .{op_name});
            
            // Get operator versions
            if (registry.getOperatorVersions(op_name)) |versions| {
                std.log.info("  - Versions: {}", .{versions.len});
                for (versions) |version| {
                    std.log.info("    * {s}", .{version});
                }
            }
        } else {
            std.log.warn("✗ Operator '{}' not found", .{op_name});
        }
    }

    // List all operators
    const all_operators = try registry.listOperators();
    defer allocator.free(all_operators);
    
    std.log.info("✓ Total registered operators: {}", .{all_operators.len});
}

fn demoTensorOperations(allocator: std.mem.Allocator) !void {
    std.log.info("\n🧮 Demo 3: Tensor Operations");
    std.log.info("-----------------------------");

    // Create tensors
    const shape = [_]usize{ 2, 3 };
    var tensor_a = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer tensor_a.deinit();
    var tensor_b = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer tensor_b.deinit();
    var result = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer result.deinit();

    // Set test data
    const data_a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const data_b = [_]f32{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
    try implementations.utils.setTensorData(&tensor_a, f32, &data_a);
    try implementations.utils.setTensorData(&tensor_b, f32, &data_b);

    std.log.info("✓ Created tensors with shape [{}, {}]", .{ shape[0], shape[1] });
    std.log.info("  - Tensor A: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]");
    std.log.info("  - Tensor B: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]");

    // Test Add operator
    const inputs = [_]implementations.Tensor{ tensor_a, tensor_b };
    var outputs = [_]implementations.Tensor{result};
    
    var attrs = implementations.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    var context = implementations.utils.createExecutionContext(allocator);
    
    try implementations.operators.arithmetic.Add.compute(&inputs, &outputs, &attrs, &context);
    
    const result_data = implementations.utils.getTensorData(&result, f32);
    std.log.info("✓ Addition result: [{d:.1}, {d:.1}, {d:.1}, {d:.1}, {d:.1}, {d:.1}]", .{
        result_data[0], result_data[1], result_data[2], result_data[3], result_data[4], result_data[5]
    });

    // Test ReLU operator
    var relu_input = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer relu_input.deinit();
    var relu_output = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer relu_output.deinit();

    const relu_data = [_]f32{ -1.0, 2.0, -3.0, 4.0, -5.0, 6.0 };
    try implementations.utils.setTensorData(&relu_input, f32, &relu_data);

    const relu_inputs = [_]implementations.Tensor{relu_input};
    var relu_outputs = [_]implementations.Tensor{relu_output};
    
    try implementations.operators.activation.ReLU.compute(&relu_inputs, &relu_outputs, &attrs, &context);
    
    const relu_result = implementations.utils.getTensorData(&relu_output, f32);
    std.log.info("✓ ReLU result: [{d:.1}, {d:.1}, {d:.1}, {d:.1}, {d:.1}, {d:.1}]", .{
        relu_result[0], relu_result[1], relu_result[2], relu_result[3], relu_result[4], relu_result[5]
    });
}

fn demoModelComponents(allocator: std.mem.Allocator) !void {
    std.log.info("\n🤖 Demo 4: Model-Specific Components");
    std.log.info("------------------------------------");

    // Test LayerNorm
    const shape = [_]usize{ 1, 4 };
    var input = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer input.deinit();
    var output = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer output.deinit();

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try implementations.utils.setTensorData(&input, f32, &input_data);

    const inputs = [_]implementations.Tensor{input};
    var outputs = [_]implementations.Tensor{output};
    
    var attrs = implementations.utils.createAttributes(allocator);
    defer attrs.deinit();
    try attrs.set("epsilon", implementations.Attributes.AttributeValue{ .float = 1e-5 });
    try attrs.set("axis", implementations.Attributes.AttributeValue{ .int = -1 });
    
    var context = implementations.utils.createExecutionContext(allocator);
    
    try implementations.models.transformers.LayerNorm.compute(&inputs, &outputs, &attrs, &context);
    
    const result_data = implementations.utils.getTensorData(&output, f32);
    std.log.info("✓ LayerNorm applied successfully");
    std.log.info("  - Input: [1.0, 2.0, 3.0, 4.0]");
    std.log.info("  - Output: [{d:.3}, {d:.3}, {d:.3}, {d:.3}]", .{
        result_data[0], result_data[1], result_data[2], result_data[3]
    });

    // Test embedding lookup
    const vocab_size = 10;
    const embed_dim = 4;
    const seq_len = 3;
    
    const embed_shape = [_]usize{ vocab_size, embed_dim };
    const indices_shape = [_]usize{seq_len};
    const embed_output_shape = [_]usize{ seq_len, embed_dim };
    
    var embeddings = try implementations.utils.createTensor(allocator, &embed_shape, .f32);
    defer embeddings.deinit();
    var indices = try implementations.utils.createTensor(allocator, &indices_shape, .i32);
    defer indices.deinit();
    var embed_output = try implementations.utils.createTensor(allocator, &embed_output_shape, .f32);
    defer embed_output.deinit();

    // Initialize embeddings with test data
    var embed_data = try allocator.alloc(f32, vocab_size * embed_dim);
    defer allocator.free(embed_data);
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            embed_data[i * embed_dim + j] = @as(f32, @floatFromInt(i * 10 + j));
        }
    }
    try implementations.utils.setTensorData(&embeddings, f32, embed_data);

    const indices_data = [_]i32{ 1, 3, 5 };
    try implementations.utils.setTensorData(&indices, i32, &indices_data);

    const embed_inputs = [_]implementations.Tensor{ embeddings, indices };
    var embed_outputs = [_]implementations.Tensor{embed_output};
    
    try implementations.models.transformers.Embedding.compute(&embed_inputs, &embed_outputs, &attrs, &context);
    
    std.log.info("✓ Embedding lookup completed");
    std.log.info("  - Looked up indices: [1, 3, 5]");
    std.log.info("  - Output shape: [{}, {}]", .{ seq_len, embed_dim });
}

fn demoCustomOperator(allocator: std.mem.Allocator) !void {
    std.log.info("\n🔧 Demo 5: Custom Operator");
    std.log.info("---------------------------");

    // Create platform
    var platform = try implementations.utils.createDefaultPlatform(allocator);
    defer platform.deinit();

    // Define a simple square operator
    const SquareOperator = implementations.BaseOperator(struct {
        pub fn getMetadata() implementations.OperatorInterface.Metadata {
            return implementations.OperatorInterface.Metadata{
                .name = "Square",
                .version = "1.0.0",
                .description = "Element-wise square operation",
                .domain = "custom",
                .min_inputs = 1,
                .max_inputs = 1,
                .min_outputs = 1,
                .max_outputs = 1,
                .type_constraints = &[_]implementations.OperatorInterface.TypeConstraint{
                    implementations.OperatorInterface.TypeConstraint{
                        .name = "T",
                        .allowed_types = &[_]implementations.Tensor.DataType{.f32},
                        .description = "Float32 tensors only",
                    },
                },
            };
        }

        pub fn validate(input_shapes: []const []const usize, input_types: []const implementations.Tensor.DataType, 
                        attributes: *const implementations.Attributes) implementations.FrameworkError!void {
            _ = attributes;
            if (input_shapes.len != 1 or input_types[0] != .f32) {
                return implementations.FrameworkError.InvalidInput;
            }
        }

        pub fn inferShapes(input_shapes: []const []const usize, attributes: *const implementations.Attributes, 
                           allocator_param: std.mem.Allocator) implementations.FrameworkError![][]usize {
            _ = attributes;
            const output_shapes = try allocator_param.alloc([]usize, 1);
            output_shapes[0] = try allocator_param.dupe(usize, input_shapes[0]);
            return output_shapes;
        }

        pub fn compute(inputs: []const implementations.Tensor, outputs: []implementations.Tensor, 
                       attributes: *const implementations.Attributes, context: *implementations.ExecutionContext) implementations.FrameworkError!void {
            _ = attributes;
            _ = context;
            
            const input_data = inputs[0].getData(f32);
            const output_data = outputs[0].getMutableData(f32);
            
            for (0..input_data.len) |i| {
                output_data[i] = input_data[i] * input_data[i];
            }
        }
    });

    // Register custom operator
    try platform.getFramework().registerOperator(SquareOperator.getDefinition());
    std.log.info("✓ Custom 'Square' operator registered");

    // Test custom operator
    const shape = [_]usize{ 1, 4 };
    var input = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer input.deinit();
    var output = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer output.deinit();

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try implementations.utils.setTensorData(&input, f32, &input_data);

    const inputs = [_]implementations.Tensor{input};
    var outputs = [_]implementations.Tensor{output};
    
    var attrs = implementations.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    var context = implementations.utils.createExecutionContext(allocator);
    
    try SquareOperator.compute(&inputs, &outputs, &attrs, &context);
    
    const result_data = implementations.utils.getTensorData(&output, f32);
    std.log.info("✓ Custom operator executed successfully");
    std.log.info("  - Input: [1.0, 2.0, 3.0, 4.0]");
    std.log.info("  - Output: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]", .{
        result_data[0], result_data[1], result_data[2], result_data[3]
    });

    // Verify the operator is registered
    if (platform.supportsOperator("Square", null)) {
        std.log.info("✓ Custom operator is now available in the platform");
    }
}

fn demoBenchmarking(allocator: std.mem.Allocator) !void {
    std.log.info("\n⚡ Demo 6: Performance Benchmarking");
    std.log.info("-----------------------------------");

    // Create platform
    var platform = try implementations.utils.createTransformerPlatform(allocator);
    defer platform.deinit();

    // Benchmark tensor creation
    const iterations = 1000;
    const shape = [_]usize{ 100, 100 };
    
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        var tensor = try implementations.utils.createTensor(allocator, &shape, .f32);
        tensor.deinit();
    }
    
    const end_time = std.time.nanoTimestamp();
    const duration_ns = end_time - start_time;
    const duration_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
    
    std.log.info("✓ Tensor creation benchmark:");
    std.log.info("  - Created {} tensors of shape [{}, {}]", .{ iterations, shape[0], shape[1] });
    std.log.info("  - Total time: {d:.2} ms", .{duration_ms});
    std.log.info("  - Average time per tensor: {d:.4} ms", .{duration_ms / @as(f64, @floatFromInt(iterations))});

    // Benchmark operator execution
    var tensor_a = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer tensor_a.deinit();
    var tensor_b = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer tensor_b.deinit();
    var result = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer result.deinit();

    // Fill with test data
    const test_data = try allocator.alloc(f32, shape[0] * shape[1]);
    defer allocator.free(test_data);
    for (0..test_data.len) |i| {
        test_data[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
    }
    try implementations.utils.setTensorData(&tensor_a, f32, test_data);
    try implementations.utils.setTensorData(&tensor_b, f32, test_data);

    const inputs = [_]implementations.Tensor{ tensor_a, tensor_b };
    var outputs = [_]implementations.Tensor{result};
    
    var attrs = implementations.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    var context = implementations.utils.createExecutionContext(allocator);

    const op_start_time = std.time.nanoTimestamp();
    
    const op_iterations = 100;
    for (0..op_iterations) |_| {
        try implementations.operators.arithmetic.Add.compute(&inputs, &outputs, &attrs, &context);
    }
    
    const op_end_time = std.time.nanoTimestamp();
    const op_duration_ns = op_end_time - op_start_time;
    const op_duration_ms = @as(f64, @floatFromInt(op_duration_ns)) / 1_000_000.0;
    
    std.log.info("✓ Add operator benchmark:");
    std.log.info("  - Executed {} additions on [{}, {}] tensors", .{ op_iterations, shape[0], shape[1] });
    std.log.info("  - Total time: {d:.2} ms", .{op_duration_ms});
    std.log.info("  - Average time per operation: {d:.4} ms", .{op_duration_ms / @as(f64, @floatFromInt(op_iterations))});
    
    const elements_per_op = shape[0] * shape[1];
    const total_elements = elements_per_op * op_iterations;
    const throughput = @as(f64, @floatFromInt(total_elements)) / (op_duration_ms / 1000.0);
    std.log.info("  - Throughput: {d:.0} elements/second", .{throughput});

    // Get final platform statistics
    const final_stats = platform.getStats();
    std.log.info("✓ Final platform statistics:");
    std.log.info("  - Peak memory usage: {} bytes", .{final_stats.framework_stats.peak_memory_used});
    std.log.info("  - Total operators available: {}", .{final_stats.total_operators});
}

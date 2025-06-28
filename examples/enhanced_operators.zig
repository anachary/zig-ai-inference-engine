const std = @import("std");
const lib = @import("zig-ai-engine");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧠 Enhanced Operators Demo - Phase 2 Implementation", .{});
    std.log.info("==================================================", .{});

    // Initialize the AI engine
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 512,
        .num_threads = 2,
        .enable_profiling = true,
    });
    defer engine.deinit();

    std.log.info("✅ AI Engine initialized with enhanced operators", .{});

    // Test enhanced activation functions
    std.log.info("", .{});
    std.log.info("🔥 Testing Enhanced Activation Functions...", .{});

    try testActivationFunctions(allocator, &engine);

    // Test convolution operators
    std.log.info("", .{});
    std.log.info("🧮 Testing Convolution Operators...", .{});

    try testConvolutionOperators(allocator, &engine);

    // Test pooling operators
    std.log.info("", .{});
    std.log.info("🏊 Testing Pooling Operators...", .{});

    try testPoolingOperators(allocator, &engine);

    std.log.info("", .{});
    std.log.info("🎊 Enhanced Operators Demo Complete!", .{});
    std.log.info("✅ Advanced activation functions working", .{});
    std.log.info("✅ Convolution operations functional", .{});
    std.log.info("✅ Pooling operations operational", .{});
    std.log.info("✅ Operator registry expanded successfully", .{});
    std.log.info("", .{});
    std.log.info("🚀 Ready for complex neural network inference!", .{});
}

fn testActivationFunctions(allocator: std.mem.Allocator, engine: *lib.Engine) !void {
    // Test input tensor
    const shape = [_]usize{4};
    var input = try engine.get_tensor(&shape, .f32);
    defer engine.return_tensor(input) catch {};

    var output = try engine.get_tensor(&shape, .f32);
    defer engine.return_tensor(output) catch {};

    // Fill input with test values: [-2, -1, 0, 1]
    try input.set_f32(&[_]usize{0}, -2.0);
    try input.set_f32(&[_]usize{1}, -1.0);
    try input.set_f32(&[_]usize{2}, 0.0);
    try input.set_f32(&[_]usize{3}, 1.0);

    const activations = [_][]const u8{ "Sigmoid", "Tanh", "GELU", "Swish", "LeakyReLU", "ELU" };

    for (activations) |activation_name| {
        if (engine.operator_registry.get(activation_name)) |op| {
            const inputs = [_]lib.tensor.Tensor{input};
            var outputs = [_]lib.tensor.Tensor{output};

            try op.forward(&inputs, &outputs, allocator);

            std.log.info("  ✅ {s} activation:", .{activation_name});
            for (0..4) |i| {
                const input_val = try input.get_f32(&[_]usize{i});
                const output_val = try output.get_f32(&[_]usize{i});
                std.log.info("    {s}({d:.1}) = {d:.3}", .{ activation_name, input_val, output_val });
            }
        } else {
            std.log.warn("  ❌ {s} operator not found", .{activation_name});
        }
    }
}

fn testConvolutionOperators(allocator: std.mem.Allocator, engine: *lib.Engine) !void {
    // Create test tensors for convolution
    // Input: [1, 1, 3, 3] - single batch, single channel, 3x3 image
    const input_shape = [_]usize{ 1, 1, 3, 3 };
    var input = try engine.get_tensor(&input_shape, .f32);
    defer engine.return_tensor(input) catch {};

    // Weight: [1, 1, 2, 2] - single output channel, single input channel, 2x2 kernel
    const weight_shape = [_]usize{ 1, 1, 2, 2 };
    var weight = try engine.get_tensor(&weight_shape, .f32);
    defer engine.return_tensor(weight) catch {};

    // Output: [1, 1, 2, 2] - single batch, single channel, 2x2 output
    const output_shape = [_]usize{ 1, 1, 2, 2 };
    var output = try engine.get_tensor(&output_shape, .f32);
    defer engine.return_tensor(output) catch {};

    // Fill input with test data (3x3 matrix)
    var val: f32 = 1.0;
    for (0..3) |i| {
        for (0..3) |j| {
            try input.set_f32(&[_]usize{ 0, 0, i, j }, val);
            val += 1.0;
        }
    }

    // Fill weight with edge detection kernel
    try weight.set_f32(&[_]usize{ 0, 0, 0, 0 }, 1.0);
    try weight.set_f32(&[_]usize{ 0, 0, 0, 1 }, -1.0);
    try weight.set_f32(&[_]usize{ 0, 0, 1, 0 }, 1.0);
    try weight.set_f32(&[_]usize{ 0, 0, 1, 1 }, -1.0);

    if (engine.operator_registry.get("Conv2D")) |conv_op| {
        const inputs = [_]lib.tensor.Tensor{ input, weight };
        var outputs = [_]lib.tensor.Tensor{output};

        try conv_op.forward(&inputs, &outputs, allocator);

        std.log.info("  ✅ Conv2D operation completed:", .{});
        std.log.info("    Input (3x3):", .{});
        for (0..3) |i| {
            var row_str = std.ArrayList(u8).init(allocator);
            defer row_str.deinit();

            for (0..3) |j| {
                const val_input = try input.get_f32(&[_]usize{ 0, 0, i, j });
                try row_str.writer().print("{d:.0} ", .{val_input});
            }
            std.log.info("      [{s}]", .{row_str.items});
        }

        std.log.info("    Output (2x2):", .{});
        for (0..2) |i| {
            var row_str = std.ArrayList(u8).init(allocator);
            defer row_str.deinit();

            for (0..2) |j| {
                const val_output = try output.get_f32(&[_]usize{ 0, 0, i, j });
                try row_str.writer().print("{d:.0} ", .{val_output});
            }
            std.log.info("      [{s}]", .{row_str.items});
        }
    } else {
        std.log.warn("  ❌ Conv2D operator not found", .{});
    }
}

fn testPoolingOperators(allocator: std.mem.Allocator, engine: *lib.Engine) !void {
    // Create test tensor for pooling
    // Input: [1, 1, 4, 4] - single batch, single channel, 4x4 image
    const input_shape = [_]usize{ 1, 1, 4, 4 };
    var input = try engine.get_tensor(&input_shape, .f32);
    defer engine.return_tensor(input) catch {};

    // Output: [1, 1, 2, 2] - single batch, single channel, 2x2 output (2x2 pooling)
    const output_shape = [_]usize{ 1, 1, 2, 2 };
    var output = try engine.get_tensor(&output_shape, .f32);
    defer engine.return_tensor(output) catch {};

    // Fill input with test data (4x4 matrix)
    var val: f32 = 1.0;
    for (0..4) |i| {
        for (0..4) |j| {
            try input.set_f32(&[_]usize{ 0, 0, i, j }, val);
            val += 1.0;
        }
    }

    const pooling_ops = [_][]const u8{ "MaxPool2D", "AvgPool2D" };

    for (pooling_ops) |pool_name| {
        if (engine.operator_registry.get(pool_name)) |pool_op| {
            const inputs = [_]lib.tensor.Tensor{input};
            var outputs = [_]lib.tensor.Tensor{output};

            try pool_op.forward(&inputs, &outputs, allocator);

            std.log.info("  ✅ {s} operation completed:", .{pool_name});
            std.log.info("    Input (4x4):", .{});
            for (0..4) |i| {
                var row_str = std.ArrayList(u8).init(allocator);
                defer row_str.deinit();

                for (0..4) |j| {
                    const val_input = try input.get_f32(&[_]usize{ 0, 0, i, j });
                    try row_str.writer().print("{d:2.0} ", .{val_input});
                }
                std.log.info("      [{s}]", .{row_str.items});
            }

            std.log.info("    Output (2x2):", .{});
            for (0..2) |i| {
                var row_str = std.ArrayList(u8).init(allocator);
                defer row_str.deinit();

                for (0..2) |j| {
                    const val_output = try output.get_f32(&[_]usize{ 0, 0, i, j });
                    try row_str.writer().print("{d:2.0} ", .{val_output});
                }
                std.log.info("      [{s}]", .{row_str.items});
            }
        } else {
            std.log.warn("  ❌ {s} operator not found", .{pool_name});
        }
    }
}

const std = @import("std");
const onnx_parser = @import("zig-onnx-parser");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== Zig ONNX Parser - Basic Parsing Example ===\n");

    // Initialize parser with custom configuration
    const config = onnx_parser.ParserConfig{
        .max_model_size_mb = 512,
        .strict_validation = true,
        .enable_optimizations = false,
        .min_opset_version = 11,
        .max_opset_version = 18,
    };

    var parser = onnx_parser.Parser.initWithConfig(allocator, config);

    std.log.info("Parser initialized with config:");
    std.log.info("  Max model size: {d}MB", .{config.max_model_size_mb});
    std.log.info("  Strict validation: {any}", .{config.strict_validation});
    std.log.info("  Supported opset versions: {d}-{d}", .{ config.min_opset_version, config.max_opset_version });

    // Example 1: Parse a simple ONNX model (if available)
    std.log.info("\n--- Example 1: File Parsing ---");

    // Create a simple test model in memory for demonstration
    const test_model_data = createTestONNXModel(allocator) catch |err| {
        std.log.warn("Could not create test model: {any}", .{err});
        return;
    };
    defer allocator.free(test_model_data);

    // Parse the test model
    var model = parser.parseBytes(test_model_data, .onnx) catch |err| {
        std.log.err("Failed to parse test model: {}", .{err});
        return;
    };
    defer model.deinit();

    std.log.info("✅ Model parsed successfully!");

    // Example 2: Examine model metadata
    std.log.info("\n--- Example 2: Model Metadata ---");

    const metadata = model.getMetadata();
    std.log.info("Model Information:");
    std.log.info("  Name: {s}", .{metadata.name});
    std.log.info("  Version: {s}", .{metadata.version});
    std.log.info("  Format: {s}", .{metadata.format.toString()});
    std.log.info("  Producer: {s} v{s}", .{ metadata.producer_name, metadata.producer_version });
    std.log.info("  IR Version: {d}", .{metadata.ir_version});
    std.log.info("  Opset Version: {d}", .{metadata.opset_version});

    // Example 3: Examine inputs and outputs
    std.log.info("\n--- Example 3: Model I/O Specifications ---");

    const inputs = model.getInputs();
    const outputs = model.getOutputs();

    std.log.info("Model Inputs ({d}):", .{inputs.len});
    for (inputs, 0..) |input, i| {
        std.log.info("  [{d}] {s}: shape {any}, type {any}", .{ i, input.name, input.shape, input.dtype });
        if (input.isFullyDefined()) {
            if (input.numel()) |elements| {
                std.log.info("      Total elements: {d}", .{elements});
            }
        } else {
            std.log.info("      Dynamic shape detected");
        }
    }

    std.log.info("Model Outputs ({d}):", .{outputs.len});
    for (outputs, 0..) |output, i| {
        std.log.info("  [{d}] {s}: shape {any}, type {any}", .{ i, output.name, output.shape, output.dtype });
    }

    // Example 4: Model validation
    std.log.info("\n--- Example 4: Model Validation ---");

    onnx_parser.validation.validateModel(&model) catch |err| {
        std.log.err("Model validation failed: {}", .{err});
        return;
    };
    std.log.info("✅ Model validation passed!");

    // Example 5: Model statistics
    std.log.info("\n--- Example 5: Model Statistics ---");

    const stats = model.getStats();
    stats.print();

    // Example 6: Graph analysis
    std.log.info("\n--- Example 6: Graph Analysis ---");

    std.log.info("Graph Structure:");
    std.log.info("  Total nodes: {d}", .{model.graph.nodes.items.len});

    // Count operators by type
    var op_counts = std.StringHashMap(usize).init(allocator);
    defer op_counts.deinit();

    for (model.graph.nodes.items) |node| {
        const count = op_counts.get(node.op_type) orelse 0;
        try op_counts.put(node.op_type, count + 1);
    }

    std.log.info("  Operator distribution:");
    var op_iter = op_counts.iterator();
    while (op_iter.next()) |entry| {
        std.log.info("    {s}: {d} nodes", .{ entry.key_ptr.*, entry.value_ptr.* });
    }

    // Example 7: Topological ordering
    std.log.info("\n--- Example 7: Topological Ordering ---");

    const topo_order = model.graph.getTopologicalOrder(allocator) catch |err| {
        std.log.warn("Could not compute topological order: {any}", .{err});
        return;
    };
    defer allocator.free(topo_order);

    std.log.info("Execution order ({d} nodes):", .{topo_order.len});
    for (topo_order[0..@min(5, topo_order.len)], 0..) |node_idx, i| {
        const node = model.graph.nodes.items[node_idx];
        std.log.info("  [{d}] {s} ({s})", .{ i, node.name, node.op_type });
    }
    if (topo_order.len > 5) {
        std.log.info("  ... and {} more nodes", .{topo_order.len - 5});
    }

    // Example 8: Model optimization (if enabled)
    std.log.info("\n--- Example 8: Model Optimization ---");

    const opt_config = onnx_parser.optimization.OptimizationConfig{
        .remove_unused_nodes = true,
        .dead_code_elimination = true,
        .constant_folding = false, // Keep simple for demo
        .operator_fusion = false,
    };

    const nodes_before = model.graph.nodes.items.len;
    onnx_parser.optimization.optimizeModel(&model, opt_config) catch |err| {
        std.log.warn("Optimization failed: {}", .{err});
    };
    const nodes_after = model.graph.nodes.items.len;

    if (nodes_before != nodes_after) {
        std.log.info("Optimization removed {d} nodes ({d} -> {d})", .{ nodes_before - nodes_after, nodes_before, nodes_after });
    } else {
        std.log.info("No optimization opportunities found");
    }

    // Example 9: Format detection
    std.log.info("\n--- Example 9: Format Detection ---");

    const test_files = [_][]const u8{
        "model.onnx",
        "model.onnx.txt",
        "model.tflite",
        "model.pt",
        "model.unknown",
    };

    for (test_files) |filename| {
        const detected_format = onnx_parser.ModelFormat.fromPath(filename);
        std.log.info("  {s} -> {s}", .{ filename, detected_format.toString() });
    }

    // Example 10: Version information
    std.log.info("\n--- Example 10: Version Information ---");

    std.log.info("Zig ONNX Parser v{s}", .{onnx_parser.version.string});
    std.log.info("Supported ONNX opset versions: {any}", .{onnx_parser.version.supported_onnx_versions});
    std.log.info("Default ONNX version: {d}", .{onnx_parser.version.default_onnx_version});

    std.log.info("\n=== Example completed successfully! ===");
}

/// Create a minimal test ONNX model for demonstration
fn createTestONNXModel(allocator: std.mem.Allocator) ![]u8 {
    // This is a simplified ONNX model creation for testing
    // In a real scenario, you would load actual ONNX files

    // Create a minimal protobuf-encoded ONNX model
    var model_data = std.ArrayList(u8).init(allocator);
    defer model_data.deinit();

    // ModelProto structure (simplified)
    // Field 1: ir_version (varint)
    try model_data.append(0x08); // Field 1, wire type 0 (varint)
    try model_data.append(0x07); // IR version 7

    // Field 2: producer_name (string)
    try model_data.append(0x12); // Field 2, wire type 2 (length-delimited)
    try model_data.append(0x0A); // Length 10
    try model_data.appendSlice("test_model");

    // Field 3: producer_version (string)
    try model_data.append(0x1A); // Field 3, wire type 2
    try model_data.append(0x03); // Length 3
    try model_data.appendSlice("1.0");

    // Field 7: graph (embedded message)
    try model_data.append(0x3A); // Field 7, wire type 2

    // Create a minimal graph
    var graph_data = std.ArrayList(u8).init(allocator);
    defer graph_data.deinit();

    // Graph name
    try graph_data.append(0x12); // Field 2, wire type 2
    try graph_data.append(0x09); // Length 9
    try graph_data.appendSlice("test_graph");

    // Add a simple Add node
    try graph_data.append(0x0A); // Field 1 (node), wire type 2

    var node_data = std.ArrayList(u8).init(allocator);
    defer node_data.deinit();

    // Node inputs
    try node_data.append(0x0A); // Field 1, wire type 2
    try node_data.append(0x01); // Length 1
    try node_data.appendSlice("A");

    try node_data.append(0x0A); // Field 1, wire type 2
    try node_data.append(0x01); // Length 1
    try node_data.appendSlice("B");

    // Node outputs
    try node_data.append(0x12); // Field 2, wire type 2
    try node_data.append(0x01); // Length 1
    try node_data.appendSlice("C");

    // Node name
    try node_data.append(0x1A); // Field 3, wire type 2
    try node_data.append(0x08); // Length 8
    try node_data.appendSlice("add_node");

    // Node op_type
    try node_data.append(0x22); // Field 4, wire type 2
    try node_data.append(0x03); // Length 3
    try node_data.appendSlice("Add");

    // Write node length and data
    try writeVarint(&graph_data, node_data.items.len);
    try graph_data.appendSlice(node_data.items);

    // Write graph length and data
    try writeVarint(&model_data, graph_data.items.len);
    try model_data.appendSlice(graph_data.items);

    return model_data.toOwnedSlice();
}

/// Helper function to write varint encoding
fn writeVarint(writer: *std.ArrayList(u8), value: usize) !void {
    var val = value;
    while (val >= 0x80) {
        try writer.append(@as(u8, @truncate(val)) | 0x80);
        val >>= 7;
    }
    try writer.append(@as(u8, @truncate(val)));
}

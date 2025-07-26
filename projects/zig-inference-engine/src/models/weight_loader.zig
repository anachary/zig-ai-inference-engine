const std = @import("std");
const Allocator = std.mem.Allocator;
const onnx_protobuf = @import("onnx_protobuf_parser.zig");

/// Real Weight Loader for ONNX Models
/// Extracts and loads transformer weights from ONNX protobuf data
pub const WeightLoader = struct {
    allocator: Allocator,
    model_proto: onnx_protobuf.ModelProto,

    const Self = @This();

    pub fn init(allocator: Allocator, model_proto: onnx_protobuf.ModelProto) Self {
        return Self{
            .allocator = allocator,
            .model_proto = model_proto,
        };
    }

    /// Load transformer weights from ONNX model
    pub fn loadTransformerWeights(self: *Self, config: ModelConfig) !TransformerWeights {
        std.log.info("⚖️  Loading transformer weights from ONNX model...", .{});

        var weights = TransformerWeights{
            .token_embedding = null,
            .position_embedding = null,
            .layers = try self.allocator.alloc(LayerWeights, config.num_layers),
            .final_layer_norm = LayerNormWeights{ .weight = null, .bias = null },
            .lm_head = null,
        };

        // Initialize layer weights
        for (weights.layers) |*layer| {
            layer.* = LayerWeights{
                .attention = AttentionWeights{
                    .query = null,
                    .key = null,
                    .value = null,
                    .output = null,
                },
                .ffn = FFNWeights{
                    .up = null,
                    .down = null,
                    .gate = null,
                },
                .layer_norm1 = LayerNormWeights{ .weight = null, .bias = null },
                .layer_norm2 = LayerNormWeights{ .weight = null, .bias = null },
            };
        }

        // Load weights from ONNX initializers
        for (self.model_proto.graph.initializers.items) |initializer| {
            try self.loadTensorWeight(&weights, initializer, config);
        }

        std.log.info("✅ Transformer weights loaded successfully", .{});
        return weights;
    }

    /// Load individual tensor weight
    fn loadTensorWeight(self: *Self, weights: *TransformerWeights, tensor_proto: onnx_protobuf.TensorProto, config: ModelConfig) !void {
        const name = tensor_proto.name;

        std.log.debug("Loading weight tensor: {s}", .{name});

        // Create tensor from protobuf data
        var tensor = try self.createTensorFromProto(tensor_proto);

        // Determine where this weight belongs based on name patterns
        if (std.mem.indexOf(u8, name, "embed") != null) {
            if (std.mem.indexOf(u8, name, "token") != null or std.mem.indexOf(u8, name, "word") != null) {
                // Token embedding
                weights.token_embedding = tensor;
                std.log.info("   Loaded token embedding: {s}", .{name});
            } else if (std.mem.indexOf(u8, name, "pos") != null) {
                // Position embedding
                weights.position_embedding = tensor;
                std.log.info("   Loaded position embedding: {s}", .{name});
            }
        } else if (std.mem.indexOf(u8, name, "lm_head") != null or
            std.mem.indexOf(u8, name, "output") != null)
        {
            // Language model head
            weights.lm_head = tensor;
            std.log.info("   Loaded LM head: {s}", .{name});
        } else if (std.mem.indexOf(u8, name, "ln_f") != null or
            std.mem.indexOf(u8, name, "final_norm") != null)
        {
            // Final layer norm
            if (std.mem.indexOf(u8, name, "weight") != null) {
                weights.final_layer_norm.weight = tensor;
            } else if (std.mem.indexOf(u8, name, "bias") != null) {
                weights.final_layer_norm.bias = tensor;
            }
            std.log.info("   Loaded final layer norm: {s}", .{name});
        } else {
            // Layer-specific weights
            const layer_idx = try self.extractLayerIndex(name, config.num_layers);
            if (layer_idx < config.num_layers) {
                try self.loadLayerWeight(&weights.layers[layer_idx], tensor, name);
            }
        }
    }

    /// Load layer-specific weight
    fn loadLayerWeight(self: *Self, layer: *LayerWeights, tensor: Tensor, name: []const u8) !void {
        _ = self;

        if (std.mem.indexOf(u8, name, "attn") != null or std.mem.indexOf(u8, name, "attention") != null) {
            // Attention weights
            if (std.mem.indexOf(u8, name, "q_proj") != null or std.mem.indexOf(u8, name, "query") != null) {
                layer.attention.query = tensor;
            } else if (std.mem.indexOf(u8, name, "k_proj") != null or std.mem.indexOf(u8, name, "key") != null) {
                layer.attention.key = tensor;
            } else if (std.mem.indexOf(u8, name, "v_proj") != null or std.mem.indexOf(u8, name, "value") != null) {
                layer.attention.value = tensor;
            } else if (std.mem.indexOf(u8, name, "o_proj") != null or std.mem.indexOf(u8, name, "out") != null) {
                layer.attention.output = tensor;
            }
            std.log.debug("   Loaded attention weight: {s}", .{name});
        } else if (std.mem.indexOf(u8, name, "mlp") != null or std.mem.indexOf(u8, name, "ffn") != null or std.mem.indexOf(u8, name, "feed_forward") != null) {
            // FFN weights
            if (std.mem.indexOf(u8, name, "up_proj") != null or std.mem.indexOf(u8, name, "fc1") != null) {
                layer.ffn.up = tensor;
            } else if (std.mem.indexOf(u8, name, "down_proj") != null or std.mem.indexOf(u8, name, "fc2") != null) {
                layer.ffn.down = tensor;
            } else if (std.mem.indexOf(u8, name, "gate_proj") != null or std.mem.indexOf(u8, name, "gate") != null) {
                layer.ffn.gate = tensor;
            }
            std.log.debug("   Loaded FFN weight: {s}", .{name});
        } else if (std.mem.indexOf(u8, name, "norm") != null or std.mem.indexOf(u8, name, "ln") != null) {
            // Layer norm weights
            if (std.mem.indexOf(u8, name, "input_layernorm") != null or std.mem.indexOf(u8, name, "ln_1") != null) {
                if (std.mem.indexOf(u8, name, "weight") != null) {
                    layer.layer_norm1.weight = tensor;
                } else if (std.mem.indexOf(u8, name, "bias") != null) {
                    layer.layer_norm1.bias = tensor;
                }
            } else if (std.mem.indexOf(u8, name, "post_attention_layernorm") != null or std.mem.indexOf(u8, name, "ln_2") != null) {
                if (std.mem.indexOf(u8, name, "weight") != null) {
                    layer.layer_norm2.weight = tensor;
                } else if (std.mem.indexOf(u8, name, "bias") != null) {
                    layer.layer_norm2.bias = tensor;
                }
            }
            std.log.debug("   Loaded layer norm weight: {s}", .{name});
        }
    }

    /// Extract layer index from weight name
    fn extractLayerIndex(self: *Self, name: []const u8, num_layers: usize) !usize {
        _ = self;

        // Look for patterns like "layers.0.", "layer.1.", "h.2.", etc.
        const patterns = [_][]const u8{ "layers.", "layer.", "h.", "blocks." };

        for (patterns) |pattern| {
            if (std.mem.indexOf(u8, name, pattern)) |start| {
                const after_pattern = name[start + pattern.len ..];

                // Find the next dot or end
                var end_idx: usize = 0;
                for (after_pattern) |char| {
                    if (char == '.' or char == '_') break;
                    if (!std.ascii.isDigit(char)) break;
                    end_idx += 1;
                }

                if (end_idx > 0) {
                    const layer_str = after_pattern[0..end_idx];
                    if (std.fmt.parseInt(usize, layer_str, 10)) |layer_idx| {
                        if (layer_idx < num_layers) {
                            return layer_idx;
                        }
                    } else |_| {}
                }
            }
        }

        return num_layers; // Return invalid index if not found
    }

    /// Create tensor from ONNX TensorProto
    fn createTensorFromProto(self: *Self, tensor_proto: onnx_protobuf.TensorProto) !Tensor {
        // Convert dimensions
        var shape = try self.allocator.alloc(usize, tensor_proto.dims.items.len);
        for (tensor_proto.dims.items, 0..) |dim, i| {
            shape[i] = @intCast(@max(0, dim));
        }

        // Determine data type
        const dtype = switch (tensor_proto.data_type) {
            1 => Tensor.DataType.f32, // FLOAT
            11 => Tensor.DataType.f64, // DOUBLE
            6 => Tensor.DataType.i32, // INT32
            7 => Tensor.DataType.i64, // INT64
            else => Tensor.DataType.f32, // Default to f32
        };

        // Calculate total size
        var total_elements: usize = 1;
        for (shape) |dim| {
            total_elements *= dim;
        }

        const element_size: usize = switch (dtype) {
            .f32, .i32 => 4,
            .f64, .i64 => 8,
        };

        // Allocate data
        var data = try self.allocator.alloc(u8, total_elements * element_size);

        // Copy data from protobuf
        if (tensor_proto.raw_data.len > 0) {
            // Raw binary data
            const copy_size = @min(data.len, tensor_proto.raw_data.len);
            @memcpy(data[0..copy_size], tensor_proto.raw_data[0..copy_size]);
        } else if (tensor_proto.float_data.items.len > 0 and dtype == .f32) {
            // Float data
            const float_data = @as([*]f32, @ptrCast(@alignCast(data.ptr)));
            const copy_count = @min(total_elements, tensor_proto.float_data.items.len);
            @memcpy(float_data[0..copy_count], tensor_proto.float_data.items[0..copy_count]);
        } else if (tensor_proto.int32_data.items.len > 0 and dtype == .i32) {
            // Int32 data
            const int_data = @as([*]i32, @ptrCast(@alignCast(data.ptr)));
            const copy_count = @min(total_elements, tensor_proto.int32_data.items.len);
            @memcpy(int_data[0..copy_count], tensor_proto.int32_data.items[0..copy_count]);
        } else if (tensor_proto.int64_data.items.len > 0 and dtype == .i64) {
            // Int64 data
            const int_data = @as([*]i64, @ptrCast(@alignCast(data.ptr)));
            const copy_count = @min(total_elements, tensor_proto.int64_data.items.len);
            @memcpy(int_data[0..copy_count], tensor_proto.int64_data.items[0..copy_count]);
        } else if (tensor_proto.double_data.items.len > 0 and dtype == .f64) {
            // Double data
            const double_data = @as([*]f64, @ptrCast(@alignCast(data.ptr)));
            const copy_count = @min(total_elements, tensor_proto.double_data.items.len);
            @memcpy(double_data[0..copy_count], tensor_proto.double_data.items[0..copy_count]);
        } else {
            // No data available, initialize with zeros
            @memset(data, 0);
            std.log.warn("No data found for tensor {s}, initialized with zeros", .{tensor_proto.name});
        }

        return Tensor{
            .data = data,
            .shape = shape,
            .dtype = dtype,
            .allocator = self.allocator,
        };
    }
};

/// Import types from transformer_inference
const transformer_inference = @import("transformer_inference.zig");
pub const ModelConfig = transformer_inference.ModelConfig;
pub const TransformerWeights = transformer_inference.TransformerWeights;
pub const LayerWeights = transformer_inference.LayerWeights;
pub const AttentionWeights = transformer_inference.AttentionWeights;
pub const FFNWeights = transformer_inference.FFNWeights;
pub const LayerNormWeights = transformer_inference.LayerNormWeights;
pub const Tensor = transformer_inference.Tensor;

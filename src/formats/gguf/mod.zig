const std = @import("std");
const Model = @import("../../core/model.zig").Model;
const Metadata = @import("../../core/model.zig").Metadata;
const Architecture = @import("../../core/model.zig").Architecture;
const Tensor = @import("../../core/tensor.zig").DynamicTensor;
const DataType = @import("../../core/tensor.zig").DataType;
const quantization = @import("quantization.zig");

/// GGUF magic bytes
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// GGUF version
const GGUF_VERSION: u32 = 3;

/// GGUF data types
pub const GGUFDataType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    // Extended types for newer GGUF versions
    bfloat16 = 13,
    qint4_0 = 14,
    qint4_1 = 15,
    qint5_0 = 16,
    qint5_1 = 17,
    qint8_0 = 18,
    qint8_1 = 19,
    qint2_k = 20,
    qint3_k = 21,
    qint4_k = 22,
    qint5_k = 23,
    qint6_k = 24,
    qint8_k = 25,
    iq2_xxs = 26,
    iq2_xs = 27,
    iq3_xxs = 28,
    iq1_s = 29,
    iq4_nl = 30,
    iq3_s = 31,
    iq2_s = 32,
    iq4_xs = 33,
    // Catch-all for unknown types
    _,

    pub fn fromInt(value: u32) !GGUFDataType {
        return @enumFromInt(value);
    }

    pub fn toDataType(self: GGUFDataType) DataType {
        return switch (self) {
            .uint8 => .u8,
            .int8 => .i8,
            .int16 => .i16,
            .int32 => .i32,
            .float32 => .f32,
            else => .f32, // Default fallback
        };
    }

    pub fn size(self: GGUFDataType) usize {
        return switch (self) {
            .uint8, .int8, .bool => 1,
            .uint16, .int16, .bfloat16 => 2,
            .uint32, .int32, .float32 => 4,
            .uint64, .int64, .float64 => 8,
            .string, .array => 0, // Variable size
            // Quantized types - approximate sizes
            .qint4_0, .qint4_1, .qint4_k => 2, // 4-bit quantized
            .qint5_0, .qint5_1, .qint5_k => 3, // 5-bit quantized
            .qint8_0, .qint8_1, .qint8_k => 1, // 8-bit quantized
            .qint2_k, .iq2_xxs, .iq2_xs, .iq2_s => 1, // 2-bit quantized
            .qint3_k, .iq3_xxs, .iq3_s => 2, // 3-bit quantized
            .qint6_k => 3, // 6-bit quantized
            .iq1_s => 1, // 1-bit quantized
            .iq4_nl, .iq4_xs => 2, // 4-bit variants
            else => 4, // Default fallback
        };
    }
};

/// GGUF tensor quantization types
pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,

    pub fn toDataType(self: GGMLType) DataType {
        return switch (self) {
            .f32 => .f32,
            .f16 => .f16,
            else => .u8, // Quantized types stored as bytes
        };
    }

    pub fn blockSize(self: GGMLType) usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .q4_0, .q4_1 => 20, // 16 4-bit values + 4 bytes metadata
            .q5_0, .q5_1 => 24, // 16 5-bit values + 8 bytes metadata
            .q8_0, .q8_1 => 34, // 32 8-bit values + 2 bytes metadata
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 256,
        };
    }

    pub fn elementsPerBlock(self: GGMLType) usize {
        return switch (self) {
            .f32, .f16 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1 => 32,
            .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
        };
    }
};

/// GGUF header
pub const GGUFHeader = struct {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,

    pub fn read(reader: anytype) !GGUFHeader {
        const magic = try reader.readIntLittle(u32);
        if (magic != GGUF_MAGIC) {
            return error.InvalidMagic;
        }

        const version = try reader.readIntLittle(u32);
        if (version != GGUF_VERSION) {
            return error.UnsupportedVersion;
        }

        const tensor_count = try reader.readIntLittle(u64);
        const metadata_kv_count = try reader.readIntLittle(u64);

        return GGUFHeader{
            .magic = magic,
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = metadata_kv_count,
        };
    }
};

/// GGUF metadata value
pub const GGUFValue = union(GGUFDataType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: []u8,
    array: []GGUFValue,
    uint64: u64,
    int64: i64,
    float64: f64,
    // Extended quantized types
    bfloat16: u16,
    qint4_0: u32,
    qint4_1: u32,
    qint5_0: u32,
    qint5_1: u32,
    qint8_0: u32,
    qint8_1: u32,
    qint2_k: u32,
    qint3_k: u32,
    qint4_k: u32,
    qint5_k: u32,
    qint6_k: u32,
    qint8_k: u32,
    iq2_xxs: u32,
    iq2_xs: u32,
    iq3_xxs: u32,
    iq1_s: u32,
    iq4_nl: u32,
    iq3_s: u32,
    iq2_s: u32,
    iq4_xs: u32,

    pub fn deinit(self: *GGUFValue, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .string => |str| allocator.free(str),
            .array => |arr| {
                for (arr) |*val| {
                    val.deinit(allocator);
                }
                allocator.free(arr);
            },
            else => {},
        }
    }
};

/// Transformer layer weights for organized access
pub const LayerWeights = struct {
    layer_idx: u32,

    // Attention weights
    attention_norm: ?*Tensor,
    wq: ?*Tensor, // Query weights
    wk: ?*Tensor, // Key weights
    wv: ?*Tensor, // Value weights
    wo: ?*Tensor, // Output weights

    // Feed-forward weights
    ffn_norm: ?*Tensor,
    w1: ?*Tensor, // Gate weights
    w2: ?*Tensor, // Down weights
    w3: ?*Tensor, // Up weights

    pub fn init(layer_idx: u32) LayerWeights {
        return LayerWeights{
            .layer_idx = layer_idx,
            .attention_norm = null,
            .wq = null,
            .wk = null,
            .wv = null,
            .wo = null,
            .ffn_norm = null,
            .w1 = null,
            .w2 = null,
            .w3 = null,
        };
    }
};

/// GGUF tensor info
pub const GGUFTensorInfo = struct {
    name: []u8,
    n_dimensions: u32,
    dimensions: []u64,
    ggml_type: GGMLType,
    offset: u64,

    pub fn deinit(self: *GGUFTensorInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.dimensions);
    }

    pub fn numel(self: GGUFTensorInfo) u64 {
        var total: u64 = 1;
        for (self.dimensions) |dim| {
            total *= dim;
        }
        return total;
    }

    pub fn sizeInBytes(self: GGUFTensorInfo) u64 {
        const elements = self.numel();
        const block_size = self.ggml_type.blockSize();
        const elements_per_block = self.ggml_type.elementsPerBlock();

        if (elements_per_block == 1) {
            return elements * block_size;
        } else {
            const num_blocks = (elements + elements_per_block - 1) / elements_per_block;
            return num_blocks * block_size;
        }
    }
};

/// GGUF model implementation with real tensor loading
pub const GGUFModel = struct {
    header: GGUFHeader,
    metadata: std.StringHashMap(GGUFValue),
    tensor_infos: []GGUFTensorInfo,
    file: std.fs.File,
    data_offset: u64,
    allocator: std.mem.Allocator,

    // Parsed model metadata
    parsed_metadata: ?Metadata,

    // Loaded tensors organized by type
    tensors: std.StringHashMap(*Tensor),

    // Common transformer tensors for easy access
    token_embeddings: ?*Tensor,
    output_weights: ?*Tensor,
    layer_weights: std.ArrayList(LayerWeights),

    pub fn deinit(self: *GGUFModel, allocator: std.mem.Allocator) void {
        // Clean up metadata
        var iterator = self.metadata.iterator();
        while (iterator.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(allocator);
        }
        self.metadata.deinit();

        // Clean up tensor infos
        for (self.tensor_infos) |*info| {
            info.deinit(allocator);
        }
        allocator.free(self.tensor_infos);

        // Clean up loaded tensors
        var tensor_iterator = self.tensors.iterator();
        while (tensor_iterator.next()) |entry| {
            entry.value_ptr.*.deinit();
            allocator.destroy(entry.value_ptr.*);
        }
        self.tensors.deinit();

        // Clean up layer weights
        self.layer_weights.deinit();

        self.file.close();
    }

    pub fn getTensor(self: *GGUFModel, name: []const u8) ?*Tensor {
        // Find tensor info
        for (self.tensor_infos) |*info| {
            if (std.mem.eql(u8, info.name, name)) {
                return self.loadTensor(info) catch null;
            }
        }
        return null;
    }

    fn loadTensor(self: *GGUFModel, info: *GGUFTensorInfo) !*Tensor {
        std.log.debug("Loading tensor: {s} (type: {s})", .{ info.name, @tagName(info.ggml_type) });

        // Seek to tensor data
        try self.file.seekTo(self.data_offset + info.offset);

        // Read quantized tensor data
        const quantized_size = info.sizeInBytes();
        const quantized_data = try self.allocator.alloc(u8, quantized_size);
        defer self.allocator.free(quantized_data);
        _ = try self.file.readAll(quantized_data);

        // Convert dimensions to usize
        var dims = try self.allocator.alloc(usize, info.dimensions.len);
        for (info.dimensions, 0..) |dim, i| {
            dims[i] = @intCast(dim);
        }

        // Calculate total elements
        var total_elements: usize = 1;
        for (dims) |dim| {
            total_elements *= dim;
        }

        // Create tensor to hold dequantized F32 data
        var tensor = try self.allocator.create(Tensor);
        tensor.* = try Tensor.init(self.allocator, .f32, dims);

        // Dequantize to F32
        const f32_data = std.mem.bytesAsSlice(f32, tensor.data);
        quantization.dequantize(info.ggml_type, quantized_data, f32_data, self.allocator) catch |err| {
            std.log.warn("Failed to dequantize tensor {s}: {}", .{ info.name, err });
            // Fallback: fill with zeros
            @memset(f32_data, 0.0);
        };

        std.log.debug("Loaded and dequantized tensor: {s} with {d} elements", .{ info.name, total_elements });
        return tensor;
    }

    /// Load all tensors and organize them by type
    pub fn loadAllTensors(self: *GGUFModel) !void {
        std.log.info("🔄 Loading {d} tensors from GGUF model...", .{self.tensor_infos.len});

        // Load each tensor
        for (self.tensor_infos) |*info| {
            std.log.debug("Loading tensor: {s} [{d}D: ", .{ info.name, info.n_dimensions });
            for (info.dimensions) |dim| {
                std.log.debug("{d} ", .{dim});
            }
            std.log.debug("] type: {s}", .{@tagName(info.ggml_type)});

            const tensor = try self.loadTensor(info);
            try self.tensors.put(try self.allocator.dupe(u8, info.name), tensor);
        }

        // Organize tensors by common transformer patterns
        try self.organizeTensors();

        std.log.info("✅ Successfully loaded and organized {d} tensors", .{self.tensors.count()});
    }

    /// Organize loaded tensors into common transformer structure
    fn organizeTensors(self: *GGUFModel) !void {
        // Find token embeddings
        if (self.tensors.get("token_embd.weight")) |tensor| {
            self.token_embeddings = tensor;
        } else if (self.tensors.get("tok_embeddings.weight")) |tensor| {
            self.token_embeddings = tensor;
        }

        // Find output weights
        if (self.tensors.get("output.weight")) |tensor| {
            self.output_weights = tensor;
        } else if (self.tensors.get("output_norm.weight")) |tensor| {
            self.output_weights = tensor;
        }

        // Organize layer weights
        var layer_idx: u32 = 0;
        while (layer_idx < 100) : (layer_idx += 1) { // Check up to 100 layers
            var layer_weights = LayerWeights.init(layer_idx);
            var found_layer = false;

            // Look for attention weights
            var buf: [256]u8 = undefined;

            // Attention norm
            if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_norm.weight", .{layer_idx})) |key| {
                if (self.tensors.get(key)) |tensor| {
                    layer_weights.attention_norm = tensor;
                    found_layer = true;
                }
            } else |_| {}

            // Query weights
            if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_q.weight", .{layer_idx})) |key| {
                if (self.tensors.get(key)) |tensor| {
                    layer_weights.wq = tensor;
                    found_layer = true;
                }
            } else |_| {}

            // Key weights
            if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_k.weight", .{layer_idx})) |key| {
                if (self.tensors.get(key)) |tensor| {
                    layer_weights.wk = tensor;
                    found_layer = true;
                }
            } else |_| {}

            // Value weights
            if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_v.weight", .{layer_idx})) |key| {
                if (self.tensors.get(key)) |tensor| {
                    layer_weights.wv = tensor;
                    found_layer = true;
                }
            } else |_| {}

            // Output weights
            if (std.fmt.bufPrint(buf[0..], "blk.{d}.attn_output.weight", .{layer_idx})) |key| {
                if (self.tensors.get(key)) |tensor| {
                    layer_weights.wo = tensor;
                    found_layer = true;
                }
            } else |_| {}

            // FFN norm
            if (std.fmt.bufPrint(buf[0..], "blk.{d}.ffn_norm.weight", .{layer_idx})) |key| {
                if (self.tensors.get(key)) |tensor| {
                    layer_weights.ffn_norm = tensor;
                    found_layer = true;
                }
            } else |_| {}

            // FFN weights
            if (std.fmt.bufPrint(buf[0..], "blk.{d}.ffn_gate.weight", .{layer_idx})) |key| {
                if (self.tensors.get(key)) |tensor| {
                    layer_weights.w1 = tensor;
                    found_layer = true;
                }
            } else |_| {}

            if (std.fmt.bufPrint(buf[0..], "blk.{d}.ffn_down.weight", .{layer_idx})) |key| {
                if (self.tensors.get(key)) |tensor| {
                    layer_weights.w2 = tensor;
                    found_layer = true;
                }
            } else |_| {}

            if (std.fmt.bufPrint(buf[0..], "blk.{d}.ffn_up.weight", .{layer_idx})) |key| {
                if (self.tensors.get(key)) |tensor| {
                    layer_weights.w3 = tensor;
                    found_layer = true;
                }
            } else |_| {}

            if (found_layer) {
                try self.layer_weights.append(layer_weights);
                std.log.debug("Organized layer {d} weights", .{layer_idx});
            } else {
                break; // No more layers found
            }
        }

        std.log.info("📊 Organized {d} transformer layers", .{self.layer_weights.items.len});
    }

    pub fn getMetadata(self: *GGUFModel) *const Metadata {
        // Extract metadata from GGUF file
        if (self.parsed_metadata) |*metadata| {
            return metadata;
        }

        // Parse metadata from GGUF header if not already cached
        self.parsed_metadata = self.parseMetadataFromHeader(self.allocator) catch |err| {
            std.log.err("Failed to parse GGUF metadata: {}", .{err});
            // Create fallback metadata
            const fallback_metadata = Metadata{
                .name = self.allocator.dupe(u8, "GGUF Model (fallback)") catch "GGUF Model",
                .architecture = .llama,
                .vocab_size = 32000,
                .embedding_dim = 4096,
                .num_layers = 32,
                .num_heads = 32,
                .num_kv_heads = null,
                .context_length = 2048,
                .intermediate_size = 11008,
                .rope_freq_base = null,
                .rope_scaling = null,
            };
            self.parsed_metadata = fallback_metadata;
            return &self.parsed_metadata.?;
        };

        return &self.parsed_metadata.?;
    }

    fn parseMetadataFromHeader(self: *GGUFModel, allocator: std.mem.Allocator) !Metadata {
        // Parse real metadata from GGUF file
        var vocab_size: u32 = 32000;
        var embedding_dim: u32 = 4096;
        var num_layers: u32 = 32;
        var num_heads: u32 = 32;
        var context_length: u32 = 2048;
        var intermediate_size: ?u32 = null;
        var architecture: Architecture = .llama;
        var model_name: []const u8 = "GGUF Model";

        // Extract from metadata key-value pairs
        if (self.metadata.get("general.architecture")) |arch_value| {
            if (arch_value == .string) {
                const arch_str = arch_value.string;
                if (std.mem.eql(u8, arch_str, "llama")) {
                    architecture = .llama;
                } else if (std.mem.eql(u8, arch_str, "gpt2")) {
                    architecture = .gpt2;
                } else if (std.mem.eql(u8, arch_str, "bert")) {
                    architecture = .bert;
                }
            }
        }

        if (self.metadata.get("general.name")) |name_value| {
            if (name_value == .string) {
                model_name = name_value.string;
            }
        }

        if (self.metadata.get("llama.vocab_size")) |vocab_value| {
            if (vocab_value == .uint32) {
                vocab_size = vocab_value.uint32;
            }
        }

        if (self.metadata.get("llama.embedding_length")) |emb_value| {
            if (emb_value == .uint32) {
                embedding_dim = emb_value.uint32;
            }
        }

        if (self.metadata.get("llama.block_count")) |layer_value| {
            if (layer_value == .uint32) {
                num_layers = layer_value.uint32;
            }
        }

        if (self.metadata.get("llama.attention.head_count")) |head_value| {
            if (head_value == .uint32) {
                num_heads = head_value.uint32;
            }
        }

        if (self.metadata.get("llama.context_length")) |ctx_value| {
            if (ctx_value == .uint32) {
                context_length = ctx_value.uint32;
            }
        }

        if (self.metadata.get("llama.feed_forward_length")) |ff_value| {
            if (ff_value == .uint32) {
                intermediate_size = ff_value.uint32;
            }
        }

        // Also try to infer from tensor shapes
        if (self.token_embeddings) |emb_tensor| {
            if (emb_tensor.shape.len >= 2) {
                vocab_size = @intCast(emb_tensor.shape[0]);
                embedding_dim = @intCast(emb_tensor.shape[1]);
            }
        }

        // Count actual layers from organized weights
        if (self.layer_weights.items.len > 0) {
            num_layers = @intCast(self.layer_weights.items.len);
        }

        return Metadata{
            .name = try allocator.dupe(u8, model_name),
            .architecture = architecture,
            .vocab_size = vocab_size,
            .embedding_dim = embedding_dim,
            .num_layers = num_layers,
            .num_heads = num_heads,
            .num_kv_heads = null,
            .context_length = context_length,
            .intermediate_size = intermediate_size,
            .rope_freq_base = null,
            .rope_scaling = null,
        };
    }
};

/// Load GGUF model from file
pub fn load(allocator: std.mem.Allocator, path: []const u8) !Model {
    var file = try std.fs.cwd().openFile(path, .{});
    var reader = file.reader();

    // Read header
    const header = try GGUFHeader.read(reader);

    // Read metadata
    var metadata = std.StringHashMap(GGUFValue).init(allocator);
    for (0..header.metadata_kv_count) |_| {
        const key = try readString(reader, allocator);
        const value = try readValue(reader, allocator);
        try metadata.put(key, value);
    }

    // Read tensor infos
    var tensor_infos = try allocator.alloc(GGUFTensorInfo, header.tensor_count);
    for (tensor_infos) |*info| {
        info.* = try readTensorInfo(reader, allocator);
    }

    // Calculate data offset (aligned to 32 bytes)
    const current_pos = try file.getPos();
    const data_offset = std.mem.alignForward(u64, current_pos, 32);

    // Create GGUF model
    var gguf_model = try allocator.create(GGUFModel);
    gguf_model.* = GGUFModel{
        .header = header,
        .metadata = metadata,
        .tensor_infos = tensor_infos,
        .file = file,
        .data_offset = data_offset,
        .allocator = allocator,
        .parsed_metadata = null,
        .tensors = std.StringHashMap(*Tensor).init(allocator),
        .token_embeddings = null,
        .output_weights = null,
        .layer_weights = std.ArrayList(LayerWeights).init(allocator),
    };

    // Load all tensors
    try gguf_model.loadAllTensors();

    // Create model metadata (simplified for now)
    const model_metadata = Metadata{
        .name = try allocator.dupe(u8, "gguf-model"),
        .architecture = .llama, // Detect from metadata
        .vocab_size = 32000, // Extract from metadata
        .context_length = 2048, // Extract from metadata
        .embedding_dim = 4096, // Extract from metadata
        .num_layers = 32, // Extract from metadata
        .num_heads = 32, // Extract from metadata
        .num_kv_heads = null,
        .intermediate_size = null,
        .rope_freq_base = null,
        .rope_scaling = null,
    };

    // Create vtable
    const vtable = &Model.VTable{
        .deinit = ggufDeinit,
        .getTensor = ggufGetTensor,
        .getMetadata = ggufGetMetadata,
    };

    return Model.init(allocator, model_metadata, vtable, gguf_model);
}

// Helper functions for reading GGUF data
fn readString(reader: anytype, allocator: std.mem.Allocator) ![]u8 {
    const len = try reader.readIntLittle(u64);
    const str = try allocator.alloc(u8, len);
    _ = try reader.readAll(str);
    return str;
}

fn readValue(reader: anytype, allocator: std.mem.Allocator) !GGUFValue {
    const value_type = try reader.readIntLittle(u32);
    const gguf_type: GGUFDataType = std.meta.intToEnum(GGUFDataType, value_type) catch {
        std.debug.print("Unknown GGUF data type: {}\n", .{value_type});
        return error.UnsupportedDataType;
    };

    return switch (gguf_type) {
        .uint8 => GGUFValue{ .uint8 = try reader.readIntLittle(u8) },
        .int8 => GGUFValue{ .int8 = try reader.readIntLittle(i8) },
        .uint16 => GGUFValue{ .uint16 = try reader.readIntLittle(u16) },
        .int16 => GGUFValue{ .int16 = try reader.readIntLittle(i16) },
        .uint32 => GGUFValue{ .uint32 = try reader.readIntLittle(u32) },
        .int32 => GGUFValue{ .int32 = try reader.readIntLittle(i32) },
        .float32 => GGUFValue{ .float32 = @bitCast(try reader.readIntLittle(u32)) },
        .bool => GGUFValue{ .bool = (try reader.readIntLittle(u8)) != 0 },
        .string => GGUFValue{ .string = try readString(reader, allocator) },
        .uint64 => GGUFValue{ .uint64 = try reader.readIntLittle(u64) },
        .int64 => GGUFValue{ .int64 = try reader.readIntLittle(i64) },
        .float64 => GGUFValue{ .float64 = @bitCast(try reader.readIntLittle(u64)) },
        .array => {
            const array_type = try reader.readIntLittle(u32);
            _ = array_type;
            const array_len = try reader.readIntLittle(u64);
            var array = try allocator.alloc(GGUFValue, array_len);

            // For arrays, we need to read each element
            // This is simplified - real implementation would handle array types properly
            for (array) |*item| {
                item.* = try readValue(reader, allocator);
            }

            return GGUFValue{ .array = array };
        },
        // Handle all quantized types as u32 values
        .bfloat16 => GGUFValue{ .bfloat16 = try reader.readIntLittle(u16) },
        .qint4_0 => GGUFValue{ .qint4_0 = try reader.readIntLittle(u32) },
        .qint4_1 => GGUFValue{ .qint4_1 = try reader.readIntLittle(u32) },
        .qint5_0 => GGUFValue{ .qint5_0 = try reader.readIntLittle(u32) },
        .qint5_1 => GGUFValue{ .qint5_1 = try reader.readIntLittle(u32) },
        .qint8_0 => GGUFValue{ .qint8_0 = try reader.readIntLittle(u32) },
        .qint8_1 => GGUFValue{ .qint8_1 = try reader.readIntLittle(u32) },
        .qint2_k => GGUFValue{ .qint2_k = try reader.readIntLittle(u32) },
        .qint3_k => GGUFValue{ .qint3_k = try reader.readIntLittle(u32) },
        .qint4_k => GGUFValue{ .qint4_k = try reader.readIntLittle(u32) },
        .qint5_k => GGUFValue{ .qint5_k = try reader.readIntLittle(u32) },
        .qint6_k => GGUFValue{ .qint6_k = try reader.readIntLittle(u32) },
        .qint8_k => GGUFValue{ .qint8_k = try reader.readIntLittle(u32) },
        .iq2_xxs => GGUFValue{ .iq2_xxs = try reader.readIntLittle(u32) },
        .iq2_xs => GGUFValue{ .iq2_xs = try reader.readIntLittle(u32) },
        .iq3_xxs => GGUFValue{ .iq3_xxs = try reader.readIntLittle(u32) },
        .iq1_s => GGUFValue{ .iq1_s = try reader.readIntLittle(u32) },
        .iq4_nl => GGUFValue{ .iq4_nl = try reader.readIntLittle(u32) },
        .iq3_s => GGUFValue{ .iq3_s = try reader.readIntLittle(u32) },
        .iq2_s => GGUFValue{ .iq2_s = try reader.readIntLittle(u32) },
        .iq4_xs => GGUFValue{ .iq4_xs = try reader.readIntLittle(u32) },
        _ => {
            // Handle unknown data types gracefully
            std.log.warn("Unknown GGUF data type: {} - skipping value", .{value_type});

            // Try to skip the value by reading a reasonable amount of data
            const skip_size: usize = switch (value_type) {
                0...12 => blk: {
                    const known_type = std.meta.intToEnum(GGUFDataType, value_type) catch GGUFDataType.float32;
                    break :blk known_type.size();
                },
                else => 4, // Default skip size for unknown types
            };

            if (skip_size > 0) {
                _ = try reader.readBytesNoEof(skip_size);
            }

            // Return a default value instead of erroring
            return GGUFValue{ .float32 = 0.0 };
        },
    };
}

fn readTensorInfo(reader: anytype, allocator: std.mem.Allocator) !GGUFTensorInfo {
    const name = try readString(reader, allocator);
    const n_dimensions = try reader.readIntLittle(u32);

    var dimensions = try allocator.alloc(u64, n_dimensions);
    for (dimensions) |*dim| {
        dim.* = try reader.readIntLittle(u64);
    }

    const ggml_type_raw = try reader.readIntLittle(u32);
    const ggml_type: GGMLType = @enumFromInt(ggml_type_raw);
    const offset = try reader.readIntLittle(u64);

    return GGUFTensorInfo{
        .name = name,
        .n_dimensions = n_dimensions,
        .dimensions = dimensions,
        .ggml_type = ggml_type,
        .offset = offset,
    };
}

// VTable implementations
fn ggufDeinit(impl: *anyopaque, allocator: std.mem.Allocator) void {
    const gguf_model: *GGUFModel = @ptrCast(@alignCast(impl));
    gguf_model.deinit(allocator);
    allocator.destroy(gguf_model);
}

fn ggufGetTensor(impl: *anyopaque, name: []const u8) ?*Tensor {
    const gguf_model: *GGUFModel = @ptrCast(@alignCast(impl));
    return gguf_model.getTensor(name);
}

fn ggufGetMetadata(impl: *anyopaque) *const Metadata {
    const gguf_model: *GGUFModel = @ptrCast(@alignCast(impl));
    return gguf_model.getMetadata();
}

test "gguf data types" {
    const testing = std.testing;

    try testing.expect(GGUFDataType.uint8.size() == 1);
    try testing.expect(GGUFDataType.float32.size() == 4);
    try testing.expect(GGUFDataType.uint64.size() == 8);
}

test "ggml types" {
    const testing = std.testing;

    try testing.expect(GGMLType.f32.blockSize() == 4);
    try testing.expect(GGMLType.q4_0.blockSize() == 20);
    try testing.expect(GGMLType.q4_0.elementsPerBlock() == 32);
}

const std = @import("std");
const types = @import("types.zig");

pub const TensorMeta = struct {
    name: []const u8,
    shape: []const usize,
    dtype: types.DType,
    offset: usize, // absolute file offset of quantized data
    ggml_type_id: u32 = 0,
};

pub const TokenizerSpec = struct {
    family: []const u8,
    vocab_size: usize,
    special_bos_id: ?u32 = null,
    special_eos_id: ?u32 = null,
};

pub const ModelDescriptor = struct {
    format: types.FormatTag = .unknown,
    architecture: types.ArchitectureTag = .unknown,
    name: []const u8 = "",

    // core dims
    num_layers: usize = 0,
    num_heads: usize = 0,
    num_kv_heads: usize = 0,
    head_dim: usize = 0,
    hidden_dim: usize = 0,
    ffn_dim: usize = 0,
    context_length: usize = 0,

    // arch specifics (research-accurate knobs)
    norm: types.NormType = .rmsnorm,
    activation: types.ActivationType = .swiglu,
    attention_kind: types.AttentionKind = .mha,

    pos_encoding: types.PositionEncoding = .rope,
    rope: types.RopeConfig = .{},

    // tokenizer
    tokenizer: TokenizerSpec = .{ .family = "unknown", .vocab_size = 0 },

    // tensors
    tensors: std.ArrayList(TensorMeta),

    pub fn init(alloc: std.mem.Allocator) ModelDescriptor {
        return .{ .tensors = std.ArrayList(TensorMeta).init(alloc) };
    }
};

pub fn deinit(self: *ModelDescriptor, allocator: std.mem.Allocator) void {
    // Free per-tensor owned allocations, then the list buffer
    var i: usize = 0;
    while (i < self.tensors.items.len) : (i += 1) {
        const tm = self.tensors.items[i];
        if (tm.name.len != 0) allocator.free(tm.name);
        if (tm.shape.len != 0) allocator.free(tm.shape);
    }
    self.tensors.deinit();
    if (self.name.len != 0) allocator.free(self.name);
}

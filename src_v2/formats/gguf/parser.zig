const std = @import("std");
const regs = @import("../../core/registries.zig");
const ir = @import("../../core/ir.zig");
const types = @import("../../core/types.zig");

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGUF_VERSION_SUPPORTED: u32 = 3; // v3

const GGUFDataType = enum(u32) {
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
    bfloat16 = 13,
    // Quant types exist hereafter but we don't need to materialize them for parsing metadata
    _,
};

fn readU32(r: anytype) !u32 {
    return try r.readIntLittle(u32);
}
fn readU64(r: anytype) !u64 {
    return try r.readIntLittle(u64);
}
fn readI32(r: anytype) !i32 {
    return try r.readIntLittle(i32);
}
fn readF32(r: anytype) !f32 {
    return @bitCast(try r.readIntLittle(u32));
}

fn readString(r: anytype, alloc: std.mem.Allocator) ![]u8 {
    const len = try readU64(r);
    const buf = try alloc.alloc(u8, len);
    _ = try r.readAll(buf);
    return buf;
}

fn skipValue(r: anytype, typ: GGUFDataType, alloc: std.mem.Allocator) !void {
    switch (typ) {
        .uint8, .int8 => {
            _ = try r.readIntLittle(u8);
        },
        .uint16, .int16, .bfloat16 => {
            _ = try r.readIntLittle(u16);
        },
        .uint32, .int32, .float32 => {
            _ = try r.readIntLittle(u32);
        },
        .uint64, .int64, .float64 => {
            _ = try r.readIntLittle(u64);
        },
        .bool => {
            _ = try r.readIntLittle(u8);
        },
        .string => {
            const len = try readU64(r);
            // skip len bytes
            var to_skip: u64 = len;
            var buf: [256]u8 = undefined;
            while (to_skip > 0) {
                const chunk: usize = if (to_skip > buf.len) buf.len else @intCast(to_skip);
                _ = try r.readAll(buf[0..chunk]);
                to_skip -= chunk;
            }
        },
        .array => {
            const elem_type_raw = try readU32(r);
            const elem_type: GGUFDataType = @enumFromInt(elem_type_raw);
            const count = try readU64(r);
            var i: u64 = 0;
            while (i < count) : (i += 1) try skipValue(r, elem_type, alloc);
        },
        _ => {
            // best-effort skip 4 bytes
            _ = try r.readIntLittle(u32);
        },
    }
}

fn supports(path: []const u8, head: []const u8) bool {
    _ = path;
    return std.mem.indexOf(u8, head, "GGUF") != null;
}

fn parse(allocator: std.mem.Allocator, path: []const u8) anyerror!regs.ParseResult {
    var file = try std.fs.cwd().openFile(path, .{});
    errdefer file.close();
    var r = file.reader();

    const magic = try readU32(r);
    if (magic != GGUF_MAGIC) return types.Error.InvalidFormat;
    const version = try readU32(r);
    // Tolerate GGUF versions >= 1 for metadata read
    if (version < 1) return types.Error.Unsupported;
    const tensor_count = try readU64(r);
    const metadata_kv_count = try readU64(r);

    // Accumulators
    var md = ir.ModelDescriptor.init(allocator);
    md.format = .gguf;
    var arch_tag: types.ArchitectureTag = .unknown;
    var model_name: []const u8 = "gguf-model";
    var vocab_size: usize = 0;
    var num_layers: usize = 0;
    var num_heads: usize = 0;
    var num_kv_heads: usize = 0;
    var embed_dim: usize = 0;
    var ffn_dim: usize = 0;
    var context_len: usize = 0;

    // Parse metadata entries
    var kv_i: u64 = 0;
    while (kv_i < metadata_kv_count) : (kv_i += 1) {
        const key = try readString(r, allocator);
        defer allocator.free(key);
        const typ_raw = try readU32(r);
        const typ: GGUFDataType = @enumFromInt(typ_raw);

        // Branch on keys we care about; skip the rest without heavy allocations
        if (std.mem.eql(u8, key, "general.architecture")) {
            if (typ == .string) {
                const s = try readString(r, allocator);
                defer allocator.free(s);
                if (std.mem.eql(u8, s, "llama")) arch_tag = .llama else if (std.mem.startsWith(u8, s, "qwen")) arch_tag = .qwen else arch_tag = .unknown;
            } else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "general.name")) {
            if (typ == .string) {
                const s = try readString(r, allocator);
                // capture name (dupe into md at the end)
                model_name = try allocator.dupe(u8, s);
                allocator.free(s);
            } else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "llama.vocab_size") or std.mem.eql(u8, key, "qwen.vocab_size") or std.mem.eql(u8, key, "qwen2.vocab_size")) {
            if (typ == .uint32) vocab_size = @intCast(try readU32(r)) else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "llama.embedding_length") or std.mem.eql(u8, key, "qwen.embedding_length") or std.mem.eql(u8, key, "qwen2.embedding_length")) {
            if (typ == .uint32) embed_dim = @intCast(try readU32(r)) else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "llama.block_count") or std.mem.eql(u8, key, "qwen.block_count") or std.mem.eql(u8, key, "qwen2.block_count")) {
            if (typ == .uint32) num_layers = @intCast(try readU32(r)) else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "llama.attention.head_count") or std.mem.eql(u8, key, "qwen.attention.head_count") or std.mem.eql(u8, key, "qwen2.attention.head_count")) {
            if (typ == .uint32) num_heads = @intCast(try readU32(r)) else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "llama.attention.head_count_kv") or std.mem.eql(u8, key, "qwen.attention.head_count_kv") or std.mem.eql(u8, key, "qwen2.attention.head_count_kv")) {
            if (typ == .uint32) num_kv_heads = @intCast(try readU32(r)) else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "llama.feed_forward_length") or std.mem.eql(u8, key, "qwen.feed_forward_length") or std.mem.eql(u8, key, "qwen2.feed_forward_length")) {
            if (typ == .uint32) ffn_dim = @intCast(try readU32(r)) else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "llama.context_length") or std.mem.eql(u8, key, "qwen.context_length") or std.mem.eql(u8, key, "qwen2.context_length")) {
            if (typ == .uint32) context_len = @intCast(try readU32(r)) else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "llama.rope.freq_base") or std.mem.eql(u8, key, "qwen.rope.freq_base") or std.mem.eql(u8, key, "rope.freq_base")) {
            if (typ == .float32) {
                const base = try readF32(r);
                md.rope.theta = base;
            } else try skipValue(r, typ, allocator);
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.tokens") or std.mem.eql(u8, key, "tokenizer.ggml.merges")) {
            // Skip large arrays to keep parse light; tokenizer implementation will load on demand
            try skipValue(r, typ, allocator);
        } else {
            try skipValue(r, typ, allocator);
        }
    }

    // Read tensor infos
    var tensor_infos = try allocator.alloc(struct {
        name: []u8,
        n_dims: u32,
        dims: []u64,
        ggml_type: u32,
        offset: u64,
    }, tensor_count);
    defer {
        // free names and dims after we copy into IR metas
        for (tensor_infos) |*ti| {
            allocator.free(ti.name);
            allocator.free(ti.dims);
        }
        allocator.free(tensor_infos);
    }

    var idx: usize = 0;
    while (idx < tensor_infos.len) : (idx += 1) {
        var t = &tensor_infos[idx];
        t.name = try readString(r, allocator);
        t.n_dims = try readU32(r);
        t.dims = try allocator.alloc(u64, t.n_dims);
        var j: usize = 0;
        while (j < t.n_dims) : (j += 1) t.dims[j] = try readU64(r);
        t.ggml_type = try readU32(r);
        t.offset = try readU64(r);
    }

    // Compute data offset (aligned to 32 bytes)
    const current_pos = try file.getPos();
    const data_offset = std.mem.alignForward(u64, current_pos, 32);
    // store data_offset into name field for now? We'll return via a side-channel soon. For now, caller can recompute.

    // Populate ModelDescriptor
    md.architecture = arch_tag;
    md.name = model_name;
    md.num_layers = num_layers;
    md.num_heads = num_heads;
    md.num_kv_heads = num_kv_heads;
    md.hidden_dim = embed_dim;
    md.ffn_dim = ffn_dim;
    md.context_length = context_len;
    if (num_heads > 0 and embed_dim > 0) md.head_dim = embed_dim / num_heads;

    // Set research-accurate defaults per arch
    switch (arch_tag) {
        .llama => {
            md.norm = .rmsnorm;
            md.activation = .swiglu;
            md.pos_encoding = .rope;
            md.rope = .{ .variant = .llama, .theta = 10000.0, .alpha = 1.0 };
            if (num_kv_heads == 1 and num_heads > 1) md.attention_kind = .mqa else if (num_kv_heads > 1 and num_kv_heads != num_heads) md.attention_kind = .gqa else md.attention_kind = .mha;
            md.tokenizer = .{ .family = "gguf", .vocab_size = vocab_size, .special_bos_id = null, .special_eos_id = null };
        },
        .qwen => {
            md.norm = .rmsnorm;
            md.activation = .swiglu;
            md.pos_encoding = .rope;
            md.rope = .{ .variant = .llama, .theta = 10000.0, .alpha = 1.0 };
            if (num_kv_heads == 1 and num_heads > 1) md.attention_kind = .mqa else if (num_kv_heads > 1 and num_kv_heads != num_heads) md.attention_kind = .gqa else md.attention_kind = .mha;
            md.tokenizer = .{ .family = "gguf", .vocab_size = vocab_size, .special_bos_id = null, .special_eos_id = null };
        },
        else => {},
    }

    // Load TensorMeta list
    var i: usize = 0;
    while (i < tensor_infos.len) : (i += 1) {
        const ti = tensor_infos[i];
        var shape = try allocator.alloc(usize, ti.n_dims);
        var d: usize = 0;
        while (d < ti.n_dims) : (d += 1) shape[d] = @intCast(ti.dims[d]);
        // Store tensor offset relative to data section; WeightStore will add data_offset
        const rel_off: usize = @intCast(ti.offset);
        try md.tensors.append(.{ .name = try allocator.dupe(u8, ti.name), .shape = shape, .dtype = .f32, .offset = rel_off, .ggml_type_id = ti.ggml_type });
    }

    file.close();
    return .{ .md = md, .data_offset = @intCast(data_offset) };
}

pub fn register(alloc: std.mem.Allocator) !void {
    const fp: regs.FormatParser = .{ .supports = supports, .parse = parse };
    try regs.registerFormat("gguf", fp);
    _ = alloc; // currently unused
}

const std = @import("std");

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
    _,
};

fn readU32(r: anytype) !u32 {
    return try r.readIntLittle(u32);
}
fn readU64(r: anytype) !u64 {
    return try r.readIntLittle(u64);
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
            _ = try r.readIntLittle(u32);
        },
    }
}

pub const LoadedTokenizer = struct {
    tokens: [][]u8,
    merges: [][]u8,
    bos_id: ?u32 = null,
    eos_id: ?u32 = null,
    unk_id: ?u32 = null,

    // If non-null, all allocations live in this arena and must be freed by arena.deinit()
    arena_ptr: ?*std.heap.ArenaAllocator = null,
};

pub const LoadOptions = struct {
    use_arena: bool = false,
};

pub fn loadWithOptions(allocator: std.mem.Allocator, path: []const u8, opts: LoadOptions) !LoadedTokenizer {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var r = file.reader();

    const magic = try readU32(r);
    if (magic != GGUF_MAGIC) return error.InvalidFormat;
    const version = try readU32(r);
    _ = version; // tolerant: proceed to read metadata entries
    const tensor_count = try readU64(r);
    const metadata_kv_count = try readU64(r);
    _ = tensor_count;

    var arena_box: ?*std.heap.ArenaAllocator = null;
    var a: std.mem.Allocator = allocator;
    if (opts.use_arena) {
        arena_box = try allocator.create(std.heap.ArenaAllocator);
        errdefer allocator.destroy(arena_box.?);
        arena_box.?.* = std.heap.ArenaAllocator.init(allocator);
        a = arena_box.?.allocator();
    }

    var tokens_slices: ?[][]u8 = null;
    var merges_slices: ?[][]u8 = null;
    var unk_id: ?u32 = null;

    var bos_id: ?u32 = null;
    var eos_id: ?u32 = null;

    var kv_i: u64 = 0;
    while (kv_i < metadata_kv_count) : (kv_i += 1) {
        const key = try readString(r, a);
        defer a.free(key);
        const typ_raw = try readU32(r);
        const typ: GGUFDataType = @enumFromInt(typ_raw);

        if (std.mem.eql(u8, key, "tokenizer.ggml.tokens") and typ == .array) {
            const elem_type_raw = try readU32(r);
            const elem_type: GGUFDataType = @enumFromInt(elem_type_raw);
            if (elem_type != .string) return error.UnexpectedTokenizerFormat;
            const count = try readU64(r);
            var arr = try a.alloc([]u8, count);
            var filled: usize = 0;
            errdefer {
                if (arena_box == null) {
                    var j: usize = 0;
                    while (j < filled) : (j += 1) a.free(arr[j]);
                    a.free(arr);
                }
            }
            var i: u64 = 0;
            while (i < count) : (i += 1) {
                arr[i] = try readString(r, a);
                filled += 1;
            }
            tokens_slices = arr;
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.merges") and typ == .array) {
            const elem_type_raw = try readU32(r);
            const elem_type: GGUFDataType = @enumFromInt(elem_type_raw);
            if (elem_type != .string) return error.UnexpectedTokenizerFormat;
            const count = try readU64(r);
            var arr = try a.alloc([]u8, count);
            var filled: usize = 0;
            errdefer {
                if (arena_box == null) {
                    var j: usize = 0;
                    while (j < filled) : (j += 1) a.free(arr[j]);
                    a.free(arr);
                }
            }
            var i: u64 = 0;
            while (i < count) : (i += 1) {
                arr[i] = try readString(r, a);
                filled += 1;
            }
            merges_slices = arr;
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.bos_token_id") and typ == .uint32) {
            bos_id = try r.readIntLittle(u32);
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.eos_token_id") and typ == .uint32) {
            eos_id = try r.readIntLittle(u32);
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.unknown_token_id") and typ == .uint32) {
            unk_id = try r.readIntLittle(u32);
        } else {
            try skipValue(r, typ, a);
        }
    }

    return .{ .tokens = tokens_slices orelse &[_][]u8{}, .merges = merges_slices orelse &[_][]u8{}, .bos_id = bos_id, .eos_id = eos_id, .unk_id = unk_id, .arena_ptr = arena_box };
}

pub fn load(allocator: std.mem.Allocator, path: []const u8) !LoadedTokenizer {
    return loadWithOptions(allocator, path, .{ .use_arena = false });
}

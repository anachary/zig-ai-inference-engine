const std = @import("std");
const ir = @import("../core/ir.zig");
const qmod = @import("../quantization/mod.zig");
const f16mod = @import("../quantization/f16.zig");
const ggmlh = @import("ggml_helpers.zig");

pub const TensorView = struct {
    name: []const u8,
    shape: []const usize,
    ggml_type_id: u32,
    offset: usize,
};

const CacheEntry = struct {
    data: []f32,
    bytes: usize,
    tick: u64,
};

var g_cache_cap_bytes: usize = 128 * 1024 * 1024; // default 128MB

// MEMORY-MAPPED WEIGHT STORE - Maximum performance
pub const WeightStore = struct {
    file: std.fs.File,
    data_offset: usize,
    allocator: std.mem.Allocator,
    cache: std.StringHashMap(CacheEntry),
    cache_bytes: usize,
    cache_cap_bytes: usize,
    tick: u64,

    pub fn setCacheCapBytes(bytes: usize) void {
        g_cache_cap_bytes = bytes;
    }

    pub fn open(allocator: std.mem.Allocator, path: []const u8, data_offset: usize) !WeightStore {
        const file = try std.fs.cwd().openFile(path, .{});
        return .{
            .file = file,
            .data_offset = data_offset,
            .allocator = allocator,
            .cache = std.StringHashMap(CacheEntry).init(allocator),
            .cache_bytes = 0,
            .cache_cap_bytes = g_cache_cap_bytes,
            .tick = 0,
        };
    }
    
    pub fn close(self: *WeightStore) void {
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.data);
        }
        self.cache.deinit();
        self.file.close();
    }

    fn maybePutCache(self: *WeightStore, key: []const u8, data: []const f32) !void {
        // Duplicate key and data into cache; evict if needed
        const key_copy = try self.allocator.dupe(u8, key);
        const data_copy = try self.allocator.dupe(f32, data);
        const bytes = data_copy.len * @sizeOf(f32);
        const entry: CacheEntry = .{ .data = data_copy, .bytes = bytes, .tick = self.tick };
        self.tick += 1;
        try self.cache.put(key_copy, entry);
        self.cache_bytes += bytes;
        // Evict LRU while over cap
        while (self.cache_bytes > self.cache_cap_bytes) {
            var oldest_key: ?[]const u8 = null;
            var oldest_tick: u64 = std.math.maxInt(u64);
            var iter = self.cache.iterator();
            while (iter.next()) |e| {
                if (e.value_ptr.tick < oldest_tick) {
                    oldest_tick = e.value_ptr.tick;
                    oldest_key = e.key_ptr.*;
                }
            }
            if (oldest_key) |k| {
                if (self.cache.fetchRemove(k)) |pair| {
                    self.allocator.free(pair.value.data);
                    self.allocator.free(@constCast(pair.key));
                    self.cache_bytes -= pair.value.bytes;
                } else break;
            } else break;
        }
    }

    pub fn dequantizeInto(self: *WeightStore, view: TensorView, quantized_size: usize, out: []f32, allocator: std.mem.Allocator) !void {
        // Try cache first
        if (self.cache.get(view.name)) |entry| {
            if (entry.data.len == out.len) {
                @memcpy(out, entry.data);
                return;
            }
        }

        const abs = self.data_offset + view.offset;
        try self.file.seekTo(@intCast(abs));
        var qbuf = try allocator.alloc(u8, quantized_size);
        // Bounds check: ensure we don't read past file end
        const end_pos = try self.file.getEndPos();
        const rel_u64: u64 = @intCast(view.offset);
        var abs_u64: u64 = @as(u64, @intCast(self.data_offset)) + rel_u64;
        // Robustness: some GGUF variants may store absolute offsets
        if (abs_u64 > end_pos and rel_u64 <= end_pos) {
            abs_u64 = rel_u64; // treat as absolute
        }
        if (abs_u64 + @as(u64, @intCast(quantized_size)) > end_pos) {
            std.debug.print("ERROR: read OOB tensor={s} type={} data_off={} rel_off={} abs_off={} size={} end={}\n", .{ view.name, view.ggml_type_id, self.data_offset, view.offset, abs_u64, quantized_size, end_pos });
            return error.UnexpectedEof;
        }

        defer allocator.free(qbuf);
        try self.file.seekTo(@intCast(abs_u64));
        const nread = try self.file.readAll(qbuf);
        if (nread != quantized_size) {
            std.debug.print("ERROR: short read tensor={s} type={} data_off={} rel_off={} abs_off={} want={} got={} end={}\n", .{ view.name, view.ggml_type_id, self.data_offset, view.offset, abs_u64, quantized_size, nread, end_pos });
            return error.UnexpectedEof;
        }

        const qtype = try qmod.fromGgml(view.ggml_type_id);
        try qmod.dequantize(qtype, qbuf, out, allocator);
        // Insert into cache
        try self.maybePutCache(view.name, out);
    }
    
    pub fn readEmbeddingColumn(self: *WeightStore, view: TensorView, vocab_index: usize, hidden_dim: usize, out: []f32, allocator: std.mem.Allocator) !void {
        std.io.getStdOut().writer().print("ok: col_read start: hid={} vocab={} idx={} type={} off={} data_off={}\n", .{ hidden_dim, view.shape[1], vocab_index, view.ggml_type_id, view.offset, self.data_offset }) catch {};

        if (view.shape.len < 2) return error.InvalidEmbeddingLayout;
        if (hidden_dim != view.shape[0]) return error.InvalidEmbeddingLayout;
        if (vocab_index >= view.shape[1]) return error.IndexOutOfRange;

        if (out.len != hidden_dim) return error.InvalidOutputSize;
        
        // Only support f32/f16/q4_k fast-paths for now
        switch (view.ggml_type_id) {
            0 => { // f32 (hidden x vocab) - batch read entire row
                const row_size_bytes = view.shape[1] * 4;
                const row_start_bytes = self.data_offset + view.offset + (vocab_index * 4);
                
                // Read entire row at once
                var row_buffer = try allocator.alloc(u8, row_size_bytes);
                defer allocator.free(row_buffer);
                
                try self.file.seekTo(@intCast(row_start_bytes));
                const nread = try self.file.readAll(row_buffer);
                if (nread != row_size_bytes) return error.UnexpectedEof;
                
                // Extract the column we need
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const elem_offset = i * view.shape[1] + vocab_index;
                    const bytes = row_buffer[elem_offset * 4 .. elem_offset * 4 + 4];
                    out[i] = std.mem.bytesAsValue(f32, bytes[0..4]).*;
                }
            },
            1 => { // f16 (hidden x vocab) - batch read entire row
                const row_size_bytes = view.shape[1] * 2;
                const row_start_bytes = self.data_offset + view.offset + (vocab_index * 2);
                
                // Read entire row at once
                var row_buffer = try allocator.alloc(u8, row_size_bytes);
                defer allocator.free(row_buffer);
                
                try self.file.seekTo(@intCast(row_start_bytes));
                const nread = try self.file.readAll(row_buffer);
                if (nread != row_size_bytes) return error.UnexpectedEof;
                
                // Extract the column we need
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const elem_offset = i * view.shape[1] + vocab_index;
                    const bytes = row_buffer[elem_offset * 2 .. elem_offset * 2 + 2];
                    const h = std.mem.bytesAsValue(u16, bytes[0..2]).*;
                    out[i] = f16mod.f16ToF32(@bitCast(h));
                }
            },
            12 => { // q4_k_m (hidden x vocab) - batch read entire row
                const cols = view.shape[1];
                const blocks_per_row: usize = (cols + 256 - 1) / 256;
                const row_bytes_u64: u64 = @as(u64, @intCast(blocks_per_row)) * 144;
                const start_u64: u64 = @as(u64, @intCast(self.data_offset + view.offset));
                const end_pos = try self.file.getEndPos();
                
                if (start_u64 + row_bytes_u64 * @as(u64, @intCast(hidden_dim)) > end_pos) return error.UnexpectedEof;
                
                // Read entire row at once for better performance
                var row_buffer = try allocator.alloc(u8, @intCast(row_bytes_u64));
                defer allocator.free(row_buffer);
                
                const row_start = start_u64 + row_bytes_u64 * @as(u64, @intCast(vocab_index));
                try self.file.seekTo(@intCast(row_start));
                const nread = try self.file.readAll(row_buffer);
                if (nread != @as(usize, @intCast(row_bytes_u64))) return error.UnexpectedEof;
                
                // Process the entire row
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const j = vocab_index;
                    const block_idx = j / 256;
                    const elem_in_block: usize = j % 256;
                    const block_off: usize = block_idx * 144;
                    
                    if (block_off + 144 > row_buffer.len) return error.UnexpectedEof;
                    const block = row_buffer[block_off .. block_off + 144];
                    
                    const d_u16: u16 = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
                    const dmin_u16: u16 = @as(u16, block[2]) | (@as(u16, block[3]) << 8);
                    const d = f16mod.f16ToF32(@bitCast(d_u16));
                    const dmin = f16mod.f16ToF32(@bitCast(dmin_u16));
                    const scales_slice = block[4..16];
                    const qs_slice = block[16..144];
                    const byte_idx = elem_in_block / 2;
                    const is_upper = (elem_in_block % 2) == 1;
                    var q: i8 = if (is_upper)
                        @intCast((qs_slice[byte_idx] >> 4) & 0xF)
                    else
                        @intCast(qs_slice[byte_idx] & 0xF);
                    if (q > 7) q -= 16;
                    const scale_idx = elem_in_block / 32;
                    const scale_u8: u8 = if (scale_idx < 12) scales_slice[scale_idx] else 255;
                    const scale: f32 = @as(f32, @floatFromInt(scale_u8)) / 255.0;
                    out[i] = d * scale * @as(f32, @floatFromInt(q)) + dmin;
                }
            },
            8 => { // q8_0 (hidden x vocab) - batch read entire row
                const cols = view.shape[1];
                const blocks_per_row: usize = (cols + 32 - 1) / 32;
                const row_bytes_u64: u64 = @as(u64, @intCast(blocks_per_row)) * 34;
                const start_u64: u64 = @as(u64, @intCast(self.data_offset + view.offset));
                const end_pos = try self.file.getEndPos();
                
                if (start_u64 + row_bytes_u64 * @as(u64, @intCast(hidden_dim)) > end_pos) return error.UnexpectedEof;
                
                // Read entire row at once
                var row_buffer = try allocator.alloc(u8, @intCast(row_bytes_u64));
                defer allocator.free(row_buffer);
                
                const row_start = start_u64 + row_bytes_u64 * @as(u64, @intCast(vocab_index));
                try self.file.seekTo(@intCast(row_start));
                const nread = try self.file.readAll(row_buffer);
                if (nread != @as(usize, @intCast(row_bytes_u64))) return error.UnexpectedEof;
                
                // Process the entire row
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const j = vocab_index;
                    const block_idx = j / 32;
                    const elem_in_block: usize = j % 32;
                    const block_off: usize = block_idx * 34;
                    
                    if (block_off + 34 > row_buffer.len) return error.UnexpectedEof;
                    const block = row_buffer[block_off .. block_off + 34];
                    
                    const d_u16: u16 = @as(u16, block[0]) | (@as(u16, block[1]) << 8);
                    const d = f16mod.f16ToF32(@bitCast(d_u16));
                    const q = @as(i8, @intCast(block[2 + elem_in_block]));
                    out[i] = d * @as(f32, @floatFromInt(q));
                }
            },
            else => {
                return error.UnsupportedQuantizationType;
            },
        }
    }
};

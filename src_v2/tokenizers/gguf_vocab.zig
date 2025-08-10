const std = @import("std");
const regs = @import("../core/registries.zig");
const gguf_tok = @import("../formats/gguf/tokenizer_loader.zig");

var g_loaded: ?gguf_tok.LoadedTokenizer = null;
var g_model_path: ?[]const u8 = null;
var g_vocab: ?std.StringHashMap(u32) = null; // piece -> id
var g_merges: ?std.StringHashMap(u32) = null; // "A B" -> rank
var g_tok_use_arena: bool = false;

pub fn setModelPath(path: []const u8) void {
    g_model_path = path;
}

pub fn setTokenizerArenaMode(enable: bool) void {
    g_tok_use_arena = enable;
}

fn ensure_loaded(allocator: std.mem.Allocator) !void {
    if (g_loaded == null) {
        const path = g_model_path orelse return error.ModelPathNotSet;
        const use_arena = g_tok_use_arena;
        g_loaded = try gguf_tok.loadWithOptions(allocator, path, .{ .use_arena = use_arena });
        // build vocab map
        var map = std.StringHashMap(u32).init(allocator);
        const loaded = g_loaded.?;
        var i: u32 = 0;
        while (i < loaded.tokens.len) : (i += 1) {
            try map.put(loaded.tokens[i], i);
        }
        g_vocab = map;
        // build merges rank map
        var m = std.StringHashMap(u32).init(allocator);
        var rank: u32 = 0;
        while (rank < loaded.merges.len) : (rank += 1) {
            try m.put(loaded.merges[rank], rank);
        }
        g_merges = m;
    }
}

pub fn getBosId() ?u32 {
    if (g_loaded) |lt| return lt.bos_id;
    return null;
}

pub fn getEosId() ?u32 {
    if (g_loaded) |lt| return lt.eos_id;
    return null;
}

pub fn getUnkId() ?u32 {
    if (g_loaded) |lt| return lt.unk_id;
    return null;
}

fn normalize_with_sentencepiece_space(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    // Replace ASCII spaces with U+2581 (UTF-8: E2 96 81)
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        const b = input[i];
        if (b == ' ') {
            try out.appendSlice(&[_]u8{ 0xE2, 0x96, 0x81 });
        } else {
            try out.append(b);
        }
    }
    return out.toOwnedSlice();
}

fn greedy_longest_match_tokenize(allocator: std.mem.Allocator, text: []const u8) ![]u32 {
    const vocab = g_vocab orelse return error.VocabNotReady;
    var ids = std.ArrayList(u32).init(allocator);
    errdefer ids.deinit();
    var i: usize = 0;
    while (i < text.len) {
        var matched = false;
        var j: usize = text.len;
        while (j > i) : (j -= 1) {
            const slice = text[i..j];
            if (vocab.get(slice)) |id| {
                try ids.append(id);
                i = j;
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Fallback: emit single byte if present in vocab; else error
            const one = text[i .. i + 1];
            if (vocab.get(one)) |id| {
                try ids.append(id);
                i += 1;
            } else {
                return error.UnknownTokenPiece;
            }
        }
    }
    return ids.toOwnedSlice();
}

fn bpe_apply(allocator: std.mem.Allocator, word: []const u8) ![]u8 {
    // Start from byte-level symbols and merge pairs by rank
    const merges = g_merges orelse return allocator.dupe(u8, word);
    var symbols = std.ArrayList([]const u8).init(allocator);
    errdefer symbols.deinit();
    // Seed with single-byte slices
    var i: usize = 0;
    while (i < word.len) : (i += 1) try symbols.append(word[i .. i + 1]);

    var changed = true;
    while (changed) {
        changed = false;
        if (symbols.items.len < 2) break;
        var best_pair_rank: ?u32 = null;
        var best_pair_idx: usize = 0;
        var k: usize = 0;
        while (k + 1 < symbols.items.len) : (k += 1) {
            var pair_buf = std.ArrayList(u8).init(allocator);
            defer pair_buf.deinit();
            try pair_buf.appendSlice(symbols.items[k]);
            try pair_buf.append(' ');
            try pair_buf.appendSlice(symbols.items[k + 1]);
            const key = pair_buf.items;
            if (merges.get(key)) |rank| {
                if (best_pair_rank == null or rank < best_pair_rank.?) {
                    best_pair_rank = rank;
                    best_pair_idx = k;
                }
            }
        }
        if (best_pair_rank) |_| {
            // Merge symbols[best_pair_idx] and symbols[best_pair_idx+1]
            var merged = std.ArrayList(u8).init(allocator);
            defer merged.deinit();
            const a = symbols.items[best_pair_idx];
            const b = symbols.items[best_pair_idx + 1];
            try merged.appendSlice(a);
            try merged.appendSlice(b);

            // Free 'b' if it was an allocated buffer (not a view into 'word') before removal
            const word_base = @intFromPtr(word.ptr);
            const word_end = word_base + word.len;
            const b_ptr = @intFromPtr(b.ptr);
            if (!(b_ptr >= word_base and b_ptr < word_end)) allocator.free(b);
            _ = symbols.orderedRemove(best_pair_idx + 1);

            // Free 'a' if it was an allocated buffer before overwriting
            const a_ptr = @intFromPtr(a.ptr);
            if (!(a_ptr >= word_base and a_ptr < word_end)) allocator.free(a);
            symbols.items[best_pair_idx] = try allocator.dupe(u8, merged.items);
            changed = true;
        }
    }

    // Join symbols to a single byte sequence
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (symbols.items) |s| try out.appendSlice(s);
    const result = try out.toOwnedSlice();

    // Free any symbol buffers we allocated (those not pointing into 'word')
    const base = @intFromPtr(word.ptr);
    const end = base + word.len;
    for (symbols.items) |s| {
        const p = @intFromPtr(s.ptr);
        if (!(p >= base and p < end)) allocator.free(s);
    }
    symbols.deinit();

    return result;
}

fn tokenize_word(allocator: std.mem.Allocator, word: []const u8) ![]u32 {
    const vocab = g_vocab orelse return error.VocabNotReady;
    const merged = try bpe_apply(allocator, word);
    defer allocator.free(merged);
    // Longest-match over merged sequence
    var ids = std.ArrayList(u32).init(allocator);
    errdefer ids.deinit();
    var i: usize = 0;
    while (i < merged.len) {
        var matched = false;
        var j: usize = merged.len;
        while (j > i) : (j -= 1) {
            const slice = merged[i..j];
            if (vocab.get(slice)) |id| {
                try ids.append(id);
                i = j;
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Fallback to single byte
            const one = merged[i .. i + 1];
            if (vocab.get(one)) |id| {
                try ids.append(id);
                i += 1;
            } else {
                if (getUnkId()) |unk| {
                    try ids.append(unk);
                    i += 1;
                } else return error.UnknownTokenPiece;
            }
        }
    }
    return ids.toOwnedSlice();
}

pub fn tokenize(allocator: std.mem.Allocator, input: []const u8) anyerror![]u32 {
    try ensure_loaded(allocator);
    const norm = try normalize_with_sentencepiece_space(allocator, input);
    defer allocator.free(norm);
    // Split on U+2581 boundaries to words
    var ids = std.ArrayList(u32).init(allocator);
    errdefer ids.deinit();
    var i: usize = 0;
    while (i < norm.len) {
        // Words in SentencePiece typically include a leading U+2581 marker in the piece strings.
        if (i + 2 < norm.len and norm[i] == 0xE2 and norm[i + 1] == 0x96 and norm[i + 2] == 0x81) {
            const start = i; // include marker in the word slice
            i += 3;
            while (i < norm.len) {
                if (i + 2 < norm.len and norm[i] == 0xE2 and norm[i + 1] == 0x96 and norm[i + 2] == 0x81) break;
                i += 1;
            }
            const word = norm[start..i];
            const word_ids = tokenize_word(allocator, word) catch |err| blk: {
                if (err == error.UnknownTokenPiece) break :blk try greedy_longest_match_tokenize(allocator, word);
                return err;
            };
            defer allocator.free(word_ids);
            try ids.appendSlice(word_ids);
        } else {
            // No marker at current position (e.g., start-of-text). Treat until next marker as a word,
            // but prefix a U+2581 to match vocab pieces like "▁Hello".
            const start = i;
            while (i < norm.len and !(i + 2 < norm.len and norm[i] == 0xE2 and norm[i + 1] == 0x96 and norm[i + 2] == 0x81)) : (i += 1) {}
            const bare = norm[start..i];
            var buf = std.ArrayList(u8).init(allocator);
            defer buf.deinit();
            try buf.appendSlice(&[_]u8{ 0xE2, 0x96, 0x81 }); // prepend marker
            try buf.appendSlice(bare);
            const word_ids = tokenize_word(allocator, buf.items) catch |err| blk2: {
                if (err == error.UnknownTokenPiece) break :blk2 try greedy_longest_match_tokenize(allocator, buf.items);
                return err;
            };
            defer allocator.free(word_ids);
            try ids.appendSlice(word_ids);
        }
    }
    return ids.toOwnedSlice();
}

pub fn detokenize(allocator: std.mem.Allocator, tokens: []const u32) anyerror![]u8 {
    try ensure_loaded(allocator);
    const loaded = g_loaded.?;
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    for (tokens) |tid| {
        if (tid >= loaded.tokens.len) continue; // skip invalid ids
        const piece = loaded.tokens[tid];
        try buf.appendSlice(piece);
    }
    var out = try buf.toOwnedSlice();
    // Replace U+2581 with space for display, then return tightly-sized buffer
    var i: usize = 0;
    var w = i;
    while (i < out.len) : (i += 1) {
        if (i + 2 < out.len and out[i] == 0xE2 and out[i + 1] == 0x96 and out[i + 2] == 0x81) {
            out[w] = ' ';
            w += 1;
            i += 2;
        } else {
            out[w] = out[i];
            w += 1;
        }
    }
    const tight = try allocator.alloc(u8, w);
    std.mem.copy(u8, tight, out[0..w]);
    allocator.free(out);
    return tight;
}

pub fn deinit(allocator: std.mem.Allocator) void {
    if (g_vocab) |*m| {
        m.deinit();
        g_vocab = null;
    }
    if (g_merges) |*m| {
        m.deinit();
        g_merges = null;
    }
    if (g_loaded) |*lt| {
        if (lt.arena_ptr) |arena| {
            // All memory for tokens/merges came from the arena; free it in one shot
            arena.deinit();
            allocator.destroy(arena);
        } else {
            var i: usize = 0;
            while (i < lt.tokens.len) : (i += 1) allocator.free(lt.tokens[i]);
            allocator.free(lt.tokens);
            i = 0;
            while (i < lt.merges.len) : (i += 1) allocator.free(lt.merges[i]);
            allocator.free(lt.merges);
        }
        g_loaded = null;
    }
}

pub fn register() !void {
    const tok: regs.Tokenizer = .{ .tokenize = tokenize, .detokenize = detokenize };
    try regs.registerTokenizer("gguf", tok);
}

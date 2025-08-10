const std = @import("std");
const api = @import("src_v2");
const tok = api.tok;

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    tok.setModelPath("models/llama-2-7b-chat.gguf");
    // Prefer explicit API to toggle arena mode instead of env in Zig 0.11
    tok.setTokenizerArenaMode(true);
    defer tok.deinit(gpa);

    const samples = [_][]const u8{
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "naïve café résumé coöperate soufflé",
        "こんにちは世界 你好，世界 Привет мир",
        "Numbers: 1234567890 and punctuation !?.,;:—",
    };

    var iter: usize = 0;
    while (iter < 300) : (iter += 1) {
        const s = samples[iter % samples.len];
        const ids = tok.tokenize(gpa, s) catch |e| blk: {
            if (e == error.UnknownTokenPiece) {
                // Reduce to ASCII
                var filtered = std.ArrayList(u8).init(gpa);
                defer filtered.deinit();
                for (s) |ch| {
                    if (ch < 0x80) try filtered.append(ch) else try filtered.append(' ');
                }
                break :blk try tok.tokenize(gpa, filtered.items);
            }
            return e;
        };
        defer gpa.free(ids);
        const out = try tok.detokenize(gpa, ids);
        defer gpa.free(out);
        if (out.len == 0) return error.EmptyOutput;
    }

    std.debug.print("tokenizer_stress_test OK\n", .{});
}

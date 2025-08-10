const std = @import("std");
const api = @import("src_v2");
const tok = api.tok;

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    tok.setModelPath("models/llama-2-7b-chat.gguf");
    defer tok.deinit(gpa);

    const input = "Hello world";
    const ids = try tok.tokenize(gpa, input);
    defer gpa.free(ids);
    const out = try tok.detokenize(gpa, ids);
    defer gpa.free(out);

    // Basic: output is non-empty and contains "Hello" (after normalization round-trip)
    if (out.len == 0) return error.EmptyOutput;
    if (std.mem.indexOf(u8, out, "Hello") == null) return error.Mismatch;

    std.debug.print("tokenizer_test OK\n", .{});

    // Cleanup tokenizer globals to avoid allocator leaks in tests
    api.tok.deinit(gpa);
}

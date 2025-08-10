const std = @import("std");
const api = @import("src_v2");
const core = api.core;

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const model_path = "models/llama-2-7b-chat.gguf";
    var session = try core.loadModel(gpa, model_path);

    // Forward on a tiny prompt and assert shapes
    const prompt = "Hello";
    const ids = try api.tok.tokenize(gpa, prompt);
    defer gpa.free(ids);

    var logits = try gpa.alloc(f32, session.model.tokenizer.vocab_size);
    defer gpa.free(logits);
    try core.forward(&session, ids, logits);

    if (logits.len != session.model.tokenizer.vocab_size) return error.BadLogitLen;

    // Deterministic greedy: sample once with temperature=0
    var buf = std.ArrayList(u8).init(gpa);
    defer buf.deinit();
    try core.generateN(&session, prompt, .{ .temperature = 0.0, .top_k = 0, .top_p = 1.0, .seed = 0 }, 1, buf.writer());
    if (buf.items.len == 0) return error.NoOutput;

    std.debug.print("smoke_forward_test OK\n", .{});
}

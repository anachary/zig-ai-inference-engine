const std = @import("std");
const api = @import("src_v2");
const chat = @import("src_v2").chat;

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    var it = try std.process.ArgIterator.initWithAllocator(gpa);
    defer it.deinit();
    _ = it.next();
    const model_path = it.next() orelse "models/llama-2-7b-chat.gguf";

    // Configure tokenizer to use the same GGUF file and arena mode for fast teardown
    api.tok.setModelPath(model_path);
    api.tok.setTokenizerArenaMode(true);
    defer api.tok.deinit(gpa);

    var session = chat.ChatSession.init(gpa, model_path, .{}) catch |err| {
        // Print a friendly error so we can see the exact cause (e.g., missing weights)
        try std.io.getStdOut().writer().print("ERROR: init failed: {any}\n", .{err});
        return err;
    };
    defer session.deinit();

    var stdin = std.io.getStdIn().reader();
    var stdout = std.io.getStdOut().writer();

    try stdout.print("Model: {s}\nType your message and press Enter. Ctrl+C to exit.\n\n> ", .{model_path});

    var line = std.ArrayList(u8).init(gpa);
    defer line.deinit();

    while (true) {
        line.clearRetainingCapacity();
        if (stdin.readUntilDelimiterArrayList(&line, '\n', 16 * 1024)) |_| {} else |_| break;
        const user = line.items;
        if (user.len == 0) {
            try stdout.print("> ", .{});
            continue;
        }
        // Append user turn; include a default system prompt on very first turn
        try session.appendUser("You are a helpful assistant.", user);
        // Generate assistant reply streaming to stdout
        try session.generate(stdout);
        try stdout.print("\n\n> ", .{});
    }
}

const std = @import("std");
const tok = @import("../../tokenizers/gguf_vocab.zig");

pub const Role = enum { system, user, assistant };

pub const Message = struct {
    role: Role,
    content: []const u8,
};

fn append(s: *std.ArrayList(u8), text: []const u8) !void {
    try s.appendSlice(text);
}

/// Build a LLaMA‑2‑Chat formatted prompt text from optional system and a
/// sequence of messages. Follows the llama.cpp style:
///   First turn: <s>[INST] <<SYS>>{system}<</SYS>> {user} [/INST]{assistant}
///   Next:       [INST] {user} [/INST]{assistant}
/// The final prompt should end after [/INST] when awaiting a new assistant reply.
pub fn formatPrompt(allocator: std.mem.Allocator, system_opt: ?[]const u8, messages: []const Message) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    try append(&buf, "<s>");

    var i: usize = 0;
    var used_system = false;

    // If a system message is provided, apply it to the first user turn
    var system_text: ?[]const u8 = null;
    if (system_opt) |sys| system_text = sys;

    while (i < messages.len) {
        const msg = messages[i];
        switch (msg.role) {
            .system => {
                // Defer applying until we see the first user
                system_text = msg.content;
                i += 1;
            },
            .user => {
                // Start an instruction block
                try append(&buf, "[INST] ");
                if (!used_system) {
                    if (system_text) |sys| {
                        try append(&buf, "<<SYS>>\n");
                        try append(&buf, sys);
                        try append(&buf, "\n<</SYS>>\n\n");
                        used_system = true;
                    }
                }
                try append(&buf, msg.content);
                try append(&buf, " [/INST]");
                i += 1;
                // If next message is assistant, append its content immediately (completed turn)
                if (i < messages.len and messages[i].role == .assistant) {
                    try append(&buf, " ");
                    try append(&buf, messages[i].content);
                    i += 1;
                }
                try append(&buf, "\n");
            },
            .assistant => {
                // If assistant appears without preceding user, append raw
                try append(&buf, msg.content);
                try append(&buf, "\n");
                i += 1;
            },
        }
    }

    return buf.toOwnedSlice();
}

/// Convenience: tokenizes the formatted prompt with the GGUF tokenizer
pub fn buildTokens(allocator: std.mem.Allocator, system_opt: ?[]const u8, messages: []const Message) ![]u32 {
    const prompt = try formatPrompt(allocator, system_opt, messages);
    defer allocator.free(prompt);
    return try tok.tokenize(allocator, prompt);
}

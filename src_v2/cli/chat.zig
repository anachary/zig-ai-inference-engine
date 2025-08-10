const std = @import("std");
const session_mod = @import("../runtime/session.zig");

pub fn run(allocator: std.mem.Allocator, model_path: []const u8) !void {
    var session = try session_mod.load(allocator, model_path);
    defer _ = session; // placeholder

    std.debug.print("Chat CLI placeholder loaded model: {s}\n", .{model_path});
}


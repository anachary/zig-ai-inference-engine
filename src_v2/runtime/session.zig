const std = @import("std");
const api = @import("../core/api.zig");

pub fn load(allocator: std.mem.Allocator, path: []const u8) anyerror!api.RuntimeSession {
    return api.loadModel(allocator, path);
}


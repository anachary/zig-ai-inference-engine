const std = @import("std");

/// Utility functions
pub fn alignTo(value: usize, alignment: usize) usize {
    return (value + alignment - 1) & ~(alignment - 1);
}

/// File utilities
pub const file_utils = struct {
    pub fn getFileSize(path: []const u8) !u64 {
        const file_handle = try std.fs.cwd().openFile(path, .{});
        defer file_handle.close();
        return file_handle.getEndPos();
    }
};

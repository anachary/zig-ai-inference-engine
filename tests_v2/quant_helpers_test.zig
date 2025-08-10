const std = @import("std");
const ggmlh = @import("../src_v2/runtime/ggml_helpers.zig");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    // Simple edge cases for calcNumel
    try std.testing.expectError(error.InvalidShape, ggmlh.calcNumel(&[_]usize{ 0 }));
    try std.testing.expectEqual(@as(usize, 6), try ggmlh.calcNumel(&[_]usize{ 2, 3 }));

    // Check sizes/lengths for a few ggml types
    const shape_q4k = [_]usize{ 256 };
    const qbytes_q4k = try ggmlh.calcQuantizedSize(12, &shape_q4k); // q4_k
    try std.testing.expectEqual(@as(usize, 144), qbytes_q4k);
    const out_q4k = try ggmlh.calcDequantizedLen(12, &shape_q4k);
    try std.testing.expectEqual(@as(usize, 256), out_q4k);

    const shape_q8 = [_]usize{ 32 };
    const qbytes_q8 = try ggmlh.calcQuantizedSize(8, &shape_q8); // q8_0
    try std.testing.expectEqual(@as(usize, 34), qbytes_q8);
    const out_q8 = try ggmlh.calcDequantizedLen(8, &shape_q8);
    try std.testing.expectEqual(@as(usize, 32), out_q8);
}


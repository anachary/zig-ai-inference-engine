const std = @import("std");
const f16mod = @import("f16.zig");

pub const Q4_K_M_Block = extern struct {
    d: u16,
    dmin: u16,
    scales: [12]u8,
    qs: [128]u8,

    pub const BLOCK_SIZE = 144;
    pub const ELEMENTS_PER_BLOCK = 256;
};

pub fn dequantize(quantized_data: []const u8, output: []f32, allocator: std.mem.Allocator) !void {
    _ = allocator;
    const num_blocks = quantized_data.len / Q4_K_M_Block.BLOCK_SIZE;
    if (output.len != num_blocks * Q4_K_M_Block.ELEMENTS_PER_BLOCK) return error.InvalidOutputSize;

    for (0..num_blocks) |bi| {
        const off = bi * Q4_K_M_Block.BLOCK_SIZE;
        const block_data = quantized_data[off .. off + Q4_K_M_Block.BLOCK_SIZE];
        const block: *const Q4_K_M_Block = @ptrCast(@alignCast(block_data.ptr));
        const d = f16mod.f16ToF32(@bitCast(block.d));
        const dmin = f16mod.f16ToF32(@bitCast(block.dmin));
        const out_off = bi * Q4_K_M_Block.ELEMENTS_PER_BLOCK;
        for (0..Q4_K_M_Block.ELEMENTS_PER_BLOCK) |i| {
            const byte_idx = i / 2;
            const is_upper = (i % 2) == 1;
            var q: i8 = if (is_upper) @intCast((block.qs[byte_idx] >> 4) & 0xF) else @intCast(block.qs[byte_idx] & 0xF);
            if (q > 7) q -= 16;
            const scale_idx = i / 32;
            const scale = if (scale_idx < 12) @as(f32, @floatFromInt(block.scales[scale_idx])) / 255.0 else 1.0;
            output[out_off + i] = d * scale * @as(f32, @floatFromInt(q)) + dmin;
        }
    }
}

const std = @import("std");
const f16mod = @import("f16.zig");
pub const Q8_0_Block = extern struct {
    d: u16,
    qs: [32]i8,

    pub const BLOCK_SIZE = 34;
    pub const ELEMENTS_PER_BLOCK = 32;
};

pub fn dequantize(quantized_data: []const u8, output: []f32, allocator: std.mem.Allocator) !void {
    _ = allocator;
    const num_blocks = quantized_data.len / Q8_0_Block.BLOCK_SIZE;
    if (output.len != num_blocks * Q8_0_Block.ELEMENTS_PER_BLOCK) return error.InvalidOutputSize;

    const Vi8 = @Vector(32, i8);
    _ = Vi8;
    const Vf32 = @Vector(32, f32);

    for (0..num_blocks) |bi| {
        const off = bi * Q8_0_Block.BLOCK_SIZE;
        const block_data = quantized_data[off .. off + Q8_0_Block.BLOCK_SIZE];
        const block: *const Q8_0_Block = @ptrCast(@alignCast(block_data.ptr));
        const d = f16mod.f16ToF32(@bitCast(block.d));
        const out_off = bi * Q8_0_Block.ELEMENTS_PER_BLOCK;

        // SIMD path: convert 32 i8 -> 32 f32 and scale by d
        var qs_arr: [32]i8 = block.qs; // copy to a local array to avoid alignment issues
        var tmp_arr: [32]f32 = undefined;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            tmp_arr[i] = @floatFromInt(qs_arr[i]);
        }
        const vf: Vf32 = @bitCast(tmp_arr);
        const vd: Vf32 = @splat(d);
        const vo: Vf32 = vf * vd;
        var out_arr: [32]f32 = @bitCast(vo);
        @memcpy(output[out_off .. out_off + 32], &out_arr);
    }
}

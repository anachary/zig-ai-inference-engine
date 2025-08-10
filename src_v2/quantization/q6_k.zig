const std = @import("std");
const f16mod = @import("f16.zig");

// Q6_K block (K-family), following ggml/llama.cpp layout
// QK_K = 256 elements per block
// Bytes per block = sizeof(f16) + 16 (per-group scales) + 192 (packed 6-bit quants) = 210
// 6-bit values are stored packed, 256 * 6 / 8 = 192 bytes
pub const Q6_K_Block = extern struct {
    d: u16,            // fp16 global scale
    scales: [16]u8,    // 16 groups per block (256 / 16)
    qs: [192]u8,       // packed 6-bit values

    pub const ELEMENTS_PER_BLOCK = 256;
    pub const BLOCK_SIZE = 210;
};

inline fn extract6(qs: []const u8, idx: usize) u8 {
    // Extract the idx-th 6-bit value from packed stream
    const bit_index = idx * 6;
    const byte_index = bit_index >> 3; // /8
    const bit_offset: u3 = @intCast(bit_index & 7);
    // Read up to 3 bytes to cover the 6 bits starting at bit_offset
    const b0: u32 = qs[byte_index];
    const b1: u32 = if (byte_index + 1 < qs.len) qs[byte_index + 1] else 0;
    const b2: u32 = if (byte_index + 2 < qs.len) qs[byte_index + 2] else 0;
    const window: u32 = b0 | (b1 << 8) | (b2 << 16);
    const v: u32 = (window >> bit_offset) & 0x3F; // 6 bits
    return @intCast(v);
}

pub fn dequantize(quantized_data: []const u8, output: []f32, allocator: std.mem.Allocator) !void {
    _ = allocator;
    if (quantized_data.len % Q6_K_Block.BLOCK_SIZE != 0) return error.InvalidQuantizedSize;
    const num_blocks = quantized_data.len / Q6_K_Block.BLOCK_SIZE;
    if (output.len != num_blocks * Q6_K_Block.ELEMENTS_PER_BLOCK) return error.InvalidOutputSize;

    var bi: usize = 0;
    while (bi < num_blocks) : (bi += 1) {
        const off = bi * Q6_K_Block.BLOCK_SIZE;
        const block_bytes = quantized_data[off .. off + Q6_K_Block.BLOCK_SIZE];
        const block: *const Q6_K_Block = @ptrCast(@alignCast(block_bytes.ptr));

        const d = f16mod.f16ToF32(@bitCast(block.d));
        const out_off = bi * Q6_K_Block.ELEMENTS_PER_BLOCK;

        // 16 groups of 16 elements each
        var i: usize = 0;
        while (i < Q6_K_Block.ELEMENTS_PER_BLOCK) : (i += 1) {
            const v6: u8 = extract6(&block.qs, i);
            // Convert to signed in range [-32 .. 31]
            var q: i8 = @intCast(v6);
            if (q >= 32) q -= 64;

            const scale_idx: usize = i / 16; // 16 elems per group
            const s_u8: u8 = block.scales[scale_idx];
            const s: f32 = @as(f32, @floatFromInt(s_u8)) / 255.0; // simplified scale normalization

            output[out_off + i] = d * s * @as(f32, @floatFromInt(q));
        }
    }
}


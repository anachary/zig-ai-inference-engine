const std = @import("std");

pub fn f16ToF32(f16_val: u16) f32 {
    const sign = (f16_val >> 15) & 0x1;
    const exponent = (f16_val >> 10) & 0x1F;
    const mantissa = f16_val & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            return if (sign == 1) -0.0 else 0.0;
        } else {
            const f32_mantissa = @as(f32, @floatFromInt(mantissa)) / 1024.0;
            const result = f32_mantissa * std.math.pow(f32, 2.0, -14.0);
            return if (sign == 1) -result else result;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            return if (sign == 1) -std.math.inf(f32) else std.math.inf(f32);
        } else {
            return std.math.nan(f32);
        }
    } else {
        const f32_exponent = @as(i32, @intCast(exponent)) - 15 + 127;
        const f32_mantissa = mantissa << 13;
        const f32_bits = (@as(u32, sign) << 31) | (@as(u32, @intCast(f32_exponent)) << 23) | f32_mantissa;
        return @bitCast(f32_bits);
    }
}

pub fn dequantize(quantized_data: []const u8, output: []f32, allocator: std.mem.Allocator) !void {
    _ = allocator;
    if (quantized_data.len != output.len * 2) return error.InvalidF16DataSize;
    var i: usize = 0;
    while (i < output.len) : (i += 1) {
        const off = i * 2;
        const v: u16 = @as(u16, quantized_data[off]) | (@as(u16, quantized_data[off + 1]) << 8);
        output[i] = f16ToF32(v);
    }
}

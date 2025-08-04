const std = @import("std");

/// Q4_K_M quantization block structure
/// Based on llama.cpp implementation
pub const Q4_K_M_Block = extern struct {
    d: f16, // Delta (scale factor)
    dmin: f16, // Minimum delta
    scales: [12]u8, // 12 bytes of scales
    qs: [128]u8, // 128 bytes of quantized values (4-bit packed)

    const BLOCK_SIZE = 144; // Total block size in bytes
    const ELEMENTS_PER_BLOCK = 256; // Number of f32 values per block
};

/// Q8_0 quantization block structure
pub const Q8_0_Block = extern struct {
    d: f16, // Delta (scale factor)
    qs: [32]i8, // 32 bytes of quantized values

    const BLOCK_SIZE = 34;
    const ELEMENTS_PER_BLOCK = 32;
};

/// F16 to F32 conversion
pub fn f16ToF32(f16_val: u16) f32 {
    // IEEE 754 half precision to single precision conversion
    const sign = (f16_val >> 15) & 0x1;
    const exponent = (f16_val >> 10) & 0x1F;
    const mantissa = f16_val & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            return if (sign == 1) -0.0 else 0.0;
        } else {
            // Denormalized number
            const f32_mantissa = @as(f32, @floatFromInt(mantissa)) / 1024.0;
            const result = f32_mantissa * std.math.pow(f32, 2.0, -14.0);
            return if (sign == 1) -result else result;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            // Infinity
            return if (sign == 1) -std.math.inf(f32) else std.math.inf(f32);
        } else {
            // NaN
            return std.math.nan(f32);
        }
    } else {
        // Normalized number
        const f32_exponent = @as(i32, @intCast(exponent)) - 15 + 127;
        const f32_mantissa = mantissa << 13;
        const f32_bits = (@as(u32, sign) << 31) | (@as(u32, @intCast(f32_exponent)) << 23) | f32_mantissa;
        return @bitCast(f32_bits);
    }
}

/// Dequantize Q4_K_M format to F32
/// This is the critical function for using Qwen2 and Llama-2 models
pub fn dequantizeQ4_K_M(
    quantized_data: []const u8,
    output: []f32,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator; // Not needed for this implementation

    const num_blocks = quantized_data.len / Q4_K_M_Block.BLOCK_SIZE;
    if (output.len != num_blocks * Q4_K_M_Block.ELEMENTS_PER_BLOCK) {
        return error.InvalidOutputSize;
    }

    std.log.debug("Dequantizing Q4_K_M: {} blocks, {} elements", .{ num_blocks, output.len });

    for (0..num_blocks) |block_idx| {
        const block_offset = block_idx * Q4_K_M_Block.BLOCK_SIZE;
        const block_data = quantized_data[block_offset .. block_offset + Q4_K_M_Block.BLOCK_SIZE];

        // Parse block structure
        const block: *const Q4_K_M_Block = @ptrCast(@alignCast(block_data.ptr));

        // Convert F16 deltas to F32
        const d = f16ToF32(@bitCast(block.d));
        const dmin = f16ToF32(@bitCast(block.dmin));

        // Dequantize each element in the block
        const output_offset = block_idx * Q4_K_M_Block.ELEMENTS_PER_BLOCK;

        for (0..Q4_K_M_Block.ELEMENTS_PER_BLOCK) |i| {
            // Q4_K_M uses complex scale and quantization scheme
            // This is a simplified implementation - real Q4_K_M is more complex

            const byte_idx = i / 2;
            const is_upper_nibble = (i % 2) == 1;

            var quantized_val: i8 = undefined;
            if (is_upper_nibble) {
                quantized_val = @intCast((block.qs[byte_idx] >> 4) & 0xF);
            } else {
                quantized_val = @intCast(block.qs[byte_idx] & 0xF);
            }

            // Convert 4-bit unsigned to signed
            if (quantized_val > 7) {
                quantized_val -= 16;
            }

            // Apply scale (simplified - real Q4_K_M has per-group scales)
            const scale_idx = i / 32; // 8 groups of 32 elements
            const scale = if (scale_idx < 12) @as(f32, @floatFromInt(block.scales[scale_idx])) / 255.0 else 1.0;

            const dequantized = d * scale * @as(f32, @floatFromInt(quantized_val)) + dmin;
            output[output_offset + i] = dequantized;
        }
    }

    std.log.debug("Q4_K_M dequantization complete");
}

/// Dequantize Q8_0 format to F32
pub fn dequantizeQ8_0(
    quantized_data: []const u8,
    output: []f32,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;

    const num_blocks = quantized_data.len / Q8_0_Block.BLOCK_SIZE;
    if (output.len != num_blocks * Q8_0_Block.ELEMENTS_PER_BLOCK) {
        return error.InvalidOutputSize;
    }

    std.log.debug("Dequantizing Q8_0: {} blocks, {} elements", .{ num_blocks, output.len });

    for (0..num_blocks) |block_idx| {
        const block_offset = block_idx * Q8_0_Block.BLOCK_SIZE;
        const block_data = quantized_data[block_offset .. block_offset + Q8_0_Block.BLOCK_SIZE];

        const block: *const Q8_0_Block = @ptrCast(@alignCast(block_data.ptr));

        // Convert F16 delta to F32
        const d = f16ToF32(@bitCast(block.d));

        // Dequantize each element
        const output_offset = block_idx * Q8_0_Block.ELEMENTS_PER_BLOCK;

        for (0..Q8_0_Block.ELEMENTS_PER_BLOCK) |i| {
            const quantized_val = block.qs[i];
            output[output_offset + i] = d * @as(f32, @floatFromInt(quantized_val));
        }
    }

    std.log.debug("Q8_0 dequantization complete");
}

/// Dequantize F16 format to F32
pub fn dequantizeF16(
    quantized_data: []const u8,
    output: []f32,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;

    if (quantized_data.len % 2 != 0) {
        return error.InvalidF16DataSize;
    }

    const num_elements = quantized_data.len / 2;
    if (output.len != num_elements) {
        return error.InvalidOutputSize;
    }

    std.log.debug("Dequantizing F16: {} elements", .{num_elements});

    for (0..num_elements) |i| {
        const f16_bytes = quantized_data[i * 2 .. (i + 1) * 2];
        const f16_val = std.mem.readIntLittle(u16, f16_bytes[0..2]);
        output[i] = f16ToF32(f16_val);
    }

    std.log.debug("F16 dequantization complete");
}

/// Generic dequantization function that dispatches to specific implementations
pub fn dequantize(
    ggml_type: @import("mod.zig").GGMLType,
    quantized_data: []const u8,
    output: []f32,
    allocator: std.mem.Allocator,
) !void {
    switch (ggml_type) {
        .f32 => {
            // Already F32, just copy
            if (quantized_data.len != output.len * 4) {
                return error.InvalidF32DataSize;
            }
            const f32_data = std.mem.bytesAsSlice(f32, quantized_data);
            @memcpy(output, f32_data);
        },
        .f16 => try dequantizeF16(quantized_data, output, allocator),
        .q4_k => try dequantizeQ4_K_M(quantized_data, output, allocator),
        .q8_0 => try dequantizeQ8_0(quantized_data, output, allocator),
        else => {
            std.log.warn("Unsupported quantization type: {s}", .{@tagName(ggml_type)});
            return error.UnsupportedQuantizationType;
        },
    }
}

/// Calculate the number of F32 elements that will result from dequantizing
pub fn calculateDequantizedSize(ggml_type: @import("mod.zig").GGMLType, quantized_size: usize) usize {
    return switch (ggml_type) {
        .f32 => quantized_size / 4,
        .f16 => quantized_size / 2,
        .q4_k => (quantized_size / Q4_K_M_Block.BLOCK_SIZE) * Q4_K_M_Block.ELEMENTS_PER_BLOCK,
        .q8_0 => (quantized_size / Q8_0_Block.BLOCK_SIZE) * Q8_0_Block.ELEMENTS_PER_BLOCK,
        else => 0, // Unsupported
    };
}

test "f16 to f32 conversion" {
    const testing = std.testing;

    // Test zero
    try testing.expect(f16ToF32(0x0000) == 0.0);

    // Test one
    try testing.expect(f16ToF32(0x3C00) == 1.0);

    // Test negative one
    try testing.expect(f16ToF32(0xBC00) == -1.0);
}

test "q4_k_m block size" {
    const testing = std.testing;
    try testing.expect(@sizeOf(Q4_K_M_Block) == Q4_K_M_Block.BLOCK_SIZE);
}

test "q8_0 block size" {
    const testing = std.testing;
    try testing.expect(@sizeOf(Q8_0_Block) == Q8_0_Block.BLOCK_SIZE);
}

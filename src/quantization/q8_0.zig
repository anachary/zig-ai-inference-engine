const std = @import("std");
const f16_utils = @import("f16.zig");

/// Q8_0 quantization implementation
/// 8-bit quantization with per-block scaling
/// Q8_0 quantization block structure
pub const Q8_0_Block = extern struct {
    d: f16, // Delta (scale factor)
    qs: [32]i8, // 32 bytes of quantized values

    pub const BLOCK_SIZE = 34;
    pub const ELEMENTS_PER_BLOCK = 32;
};

/// Dequantize Q8_0 format to F32
pub fn dequantize(
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
        const d = f16_utils.f16ToF32(@bitCast(block.d));

        // Dequantize each element
        const output_offset = block_idx * Q8_0_Block.ELEMENTS_PER_BLOCK;

        for (0..Q8_0_Block.ELEMENTS_PER_BLOCK) |i| {
            const quantized_val = block.qs[i];
            output[output_offset + i] = d * @as(f32, @floatFromInt(quantized_val));
        }
    }

    std.log.debug("Q8_0 dequantization complete");
}

/// Calculate the number of F32 elements from Q8_0 data size
pub fn calculateOutputSize(quantized_size: usize) usize {
    const num_blocks = quantized_size / Q8_0_Block.BLOCK_SIZE;
    return num_blocks * Q8_0_Block.ELEMENTS_PER_BLOCK;
}

/// Calculate the required quantized data size for a given number of elements
pub fn calculateQuantizedSize(num_elements: usize) usize {
    const num_blocks = (num_elements + Q8_0_Block.ELEMENTS_PER_BLOCK - 1) / Q8_0_Block.ELEMENTS_PER_BLOCK;
    return num_blocks * Q8_0_Block.BLOCK_SIZE;
}

/// Quantize F32 array to Q8_0 format
pub fn quantize(
    input: []const f32,
    output: []u8,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;

    const num_blocks = (input.len + Q8_0_Block.ELEMENTS_PER_BLOCK - 1) / Q8_0_Block.ELEMENTS_PER_BLOCK;
    const required_size = num_blocks * Q8_0_Block.BLOCK_SIZE;

    if (output.len < required_size) {
        return error.InsufficientOutputSize;
    }

    std.log.debug("Quantizing to Q8_0: {} elements, {} blocks", .{ input.len, num_blocks });

    for (0..num_blocks) |block_idx| {
        const input_offset = block_idx * Q8_0_Block.ELEMENTS_PER_BLOCK;
        const block_offset = block_idx * Q8_0_Block.BLOCK_SIZE;

        const elements_in_block = @min(Q8_0_Block.ELEMENTS_PER_BLOCK, input.len - input_offset);
        const block_input = input[input_offset .. input_offset + elements_in_block];

        // Find the maximum absolute value for scaling
        var max_abs: f32 = 0.0;
        for (block_input) |val| {
            max_abs = @max(max_abs, @fabs(val));
        }

        // Calculate scale factor
        const scale = if (max_abs > 0.0) max_abs / 127.0 else 1.0;
        const inv_scale = 1.0 / scale;

        // Set up block
        const block_data = output[block_offset .. block_offset + Q8_0_Block.BLOCK_SIZE];
        const block: *Q8_0_Block = @ptrCast(@alignCast(block_data.ptr));

        block.d = @bitCast(f16_utils.f32ToF16(scale));

        // Quantize values
        for (0..Q8_0_Block.ELEMENTS_PER_BLOCK) |i| {
            if (i < elements_in_block) {
                const quantized = @round(block_input[i] * inv_scale);
                const clamped = @max(-127.0, @min(127.0, quantized));
                block.qs[i] = @intFromFloat(clamped);
            } else {
                block.qs[i] = 0;
            }
        }
    }

    std.log.debug("Q8_0 quantization complete");
}

/// Q8_0 dequantizer implementation
pub const Q8_0_Dequantizer = struct {
    config: @import("mod.zig").QuantizationConfig,

    pub fn init() Q8_0_Dequantizer {
        return Q8_0_Dequantizer{
            .config = @import("mod.zig").QuantizationConfig.init(.q8_0),
        };
    }

    pub fn dequantizeImpl(self: *Q8_0_Dequantizer, quantized_data: []const u8, output: []f32, allocator: std.mem.Allocator) !void {
        _ = self;
        return dequantize(quantized_data, output, allocator);
    }

    pub fn getConfig(self: *Q8_0_Dequantizer) @import("mod.zig").QuantizationConfig {
        return self.config;
    }

    pub fn deinit(self: *Q8_0_Dequantizer) void {
        _ = self;
        // Nothing to clean up
    }
};

test "q8_0 block size" {
    const testing = std.testing;
    try testing.expect(@sizeOf(Q8_0_Block) == Q8_0_Block.BLOCK_SIZE);
}

test "q8_0 size calculations" {
    const testing = std.testing;

    // Test output size calculation
    const quantized_size = Q8_0_Block.BLOCK_SIZE * 3; // 3 blocks
    const expected_output = Q8_0_Block.ELEMENTS_PER_BLOCK * 3; // 3 * 32 = 96
    try testing.expect(calculateOutputSize(quantized_size) == expected_output);

    // Test quantized size calculation
    const num_elements = Q8_0_Block.ELEMENTS_PER_BLOCK * 2 + 10; // 2.31 blocks
    const expected_quantized = Q8_0_Block.BLOCK_SIZE * 3; // Rounded up to 3 blocks
    try testing.expect(calculateQuantizedSize(num_elements) == expected_quantized);
}

test "q8_0 round trip" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test data
    const test_values = [_]f32{ 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1 };
    const padded_size = Q8_0_Block.ELEMENTS_PER_BLOCK; // Pad to full block

    var input: [padded_size]f32 = std.mem.zeroes([padded_size]f32);
    @memcpy(input[0..test_values.len], &test_values);

    // Quantize
    const quantized_size = calculateQuantizedSize(input.len);
    var quantized_data = try allocator.alloc(u8, quantized_size);
    defer allocator.free(quantized_data);

    try quantize(&input, quantized_data, allocator);

    // Dequantize
    var output: [padded_size]f32 = undefined;
    try dequantize(quantized_data, &output, allocator);

    // Check results (with tolerance for quantization error)
    const tolerance = 0.1; // Q8_0 has some quantization error
    for (test_values, output[0..test_values.len]) |expected, actual| {
        try testing.expect(@fabs(expected - actual) < tolerance);
    }
}

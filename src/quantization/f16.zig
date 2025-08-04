const std = @import("std");

/// F16 to F32 conversion utilities
/// IEEE 754 half precision to single precision conversion
/// Convert F16 (u16) to F32
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

/// Convert F32 to F16 (u16)
pub fn f32ToF16(f32_val: f32) u16 {
    const f32_bits: u32 = @bitCast(f32_val);
    const sign = (f32_bits >> 31) & 0x1;
    const exponent = (f32_bits >> 23) & 0xFF;
    const mantissa = f32_bits & 0x7FFFFF;

    // Handle special cases
    if (exponent == 0) {
        // Zero or denormalized
        return @intCast(sign << 15);
    } else if (exponent == 255) {
        // Infinity or NaN
        if (mantissa == 0) {
            // Infinity
            return @intCast((sign << 15) | 0x7C00);
        } else {
            // NaN
            return @intCast((sign << 15) | 0x7C00 | (mantissa >> 13));
        }
    } else {
        // Normalized number
        const f16_exponent = @as(i32, @intCast(exponent)) - 127 + 15;

        if (f16_exponent <= 0) {
            // Underflow to zero
            return @intCast(sign << 15);
        } else if (f16_exponent >= 31) {
            // Overflow to infinity
            return @intCast((sign << 15) | 0x7C00);
        } else {
            // Normal conversion
            const f16_mantissa = mantissa >> 13;
            return @intCast((sign << 15) | (@as(u32, @intCast(f16_exponent)) << 10) | f16_mantissa);
        }
    }
}

/// Dequantize F16 format to F32
pub fn dequantize(
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

/// Quantize F32 array to F16 format
pub fn quantize(
    input: []const f32,
    output: []u8,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;

    if (output.len != input.len * 2) {
        return error.InvalidOutputSize;
    }

    std.log.debug("Quantizing to F16: {} elements", .{input.len});

    for (input, 0..) |val, i| {
        const f16_val = f32ToF16(val);
        const f16_bytes = std.mem.asBytes(&f16_val);
        output[i * 2] = f16_bytes[0];
        output[i * 2 + 1] = f16_bytes[1];
    }

    std.log.debug("F16 quantization complete");
}

/// F16 dequantizer implementation
pub const F16Dequantizer = struct {
    config: @import("mod.zig").QuantizationConfig,

    pub fn init() F16Dequantizer {
        return F16Dequantizer{
            .config = @import("mod.zig").QuantizationConfig.init(.f16),
        };
    }

    pub fn dequantizeImpl(self: *F16Dequantizer, quantized_data: []const u8, output: []f32, allocator: std.mem.Allocator) !void {
        _ = self;
        return dequantize(quantized_data, output, allocator);
    }

    pub fn getConfig(self: *F16Dequantizer) @import("mod.zig").QuantizationConfig {
        return self.config;
    }

    pub fn deinit(self: *F16Dequantizer) void {
        _ = self;
        // Nothing to clean up
    }
};

test "f16 conversion" {
    const testing = std.testing;

    // Test zero
    try testing.expect(f16ToF32(0x0000) == 0.0);

    // Test one
    try testing.expect(f16ToF32(0x3C00) == 1.0);

    // Test negative one
    try testing.expect(f16ToF32(0xBC00) == -1.0);

    // Test round-trip conversion
    const original: f32 = 3.14159;
    const f16_val = f32ToF16(original);
    const converted = f16ToF32(f16_val);

    // F16 has limited precision, so we check within tolerance
    const tolerance = 0.001;
    try testing.expect(@fabs(original - converted) < tolerance);
}

test "f16 dequantization" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create test F16 data
    const test_values = [_]f32{ 1.0, -1.0, 0.5, -0.5, 2.0 };
    var f16_data: [10]u8 = undefined;

    for (test_values, 0..) |val, i| {
        const f16_val = f32ToF16(val);
        const f16_bytes = std.mem.asBytes(&f16_val);
        f16_data[i * 2] = f16_bytes[0];
        f16_data[i * 2 + 1] = f16_bytes[1];
    }

    // Dequantize
    var output: [5]f32 = undefined;
    try dequantize(&f16_data, &output, allocator);

    // Check results (with tolerance for F16 precision)
    const tolerance = 0.001;
    for (test_values, output) |expected, actual| {
        try testing.expect(@fabs(expected - actual) < tolerance);
    }
}

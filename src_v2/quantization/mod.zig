const std = @import("std");

pub const q4_k_m = @import("q4_k_m.zig");
pub const q8_0 = @import("q8_0.zig");
pub const q6_km = @import("q6_k.zig");
pub const f16m = @import("f16.zig");

pub const QuantizationType = enum {
    f32,
    f16,
    q4_0,
    q4_1,
    q5_0,
    q5_1,
    q8_0,
    q8_1,
    q2_k,
    q3_k,
    q4_k,
    q5_k,
    q6_k,
    q8_k,
    iq2_xxs,
    iq2_xs,
    iq3_xxs,
    iq1_s,
    iq4_nl,
    iq3_s,
    iq2_s,
    iq4_xs,

    pub fn blockSize(self: QuantizationType) usize {
        // Bytes per block (for scalar types: bytes per element)
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .q4_0, .q4_1 => 20,
            .q5_0, .q5_1 => 24,
            .q8_0, .q8_1 => 34,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 256,
            .iq2_xxs => 16,
            .iq2_xs => 20,
            .iq3_xxs => 26,
            .iq1_s => 16,
            .iq4_nl => 32,
            .iq3_s => 32,
            .iq2_s => 20,
            .iq4_xs => 34,
        };
    }

    pub fn elementsPerBlock(self: QuantizationType) usize {
        return switch (self) {
            .f32, .f16 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1 => 32,
            .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s => 256,
            .iq4_nl, .iq3_s, .iq2_s, .iq4_xs => 256,
        };
    }
};

pub fn fromGgml(ggml_type_id: u32) !QuantizationType {
    return switch (ggml_type_id) {
        0 => .f32,
        1 => .f16,
        2 => .q4_0,
        3 => .q4_1,
        6 => .q5_0,
        7 => .q5_1,
        8 => .q8_0,
        9 => .q8_1,
        10 => .q2_k,
        11 => .q3_k,
        12 => .q4_k,
        13 => .q5_k,
        14 => .q6_k,
        15 => .q8_k,
        16 => .iq2_xxs,
        17 => .iq2_xs,
        18 => .iq3_xxs,
        19 => .iq1_s,
        20 => .iq4_nl,
        21 => .iq3_s,
        22 => .iq2_s,
        23 => .iq4_xs,
        else => error.UnsupportedGGMLType,
    };
}

/// Calculate quantized byte size needed to store num_elements of the given type
pub fn calculateQuantizedSize(qtype: QuantizationType, num_elements: usize) usize {
    return switch (qtype) {
        .f32 => num_elements * 4,
        .f16 => num_elements * 2,
        else => blk: {
            const epb = qtype.elementsPerBlock();
            const blocks = (num_elements + epb - 1) / epb;
            break :blk blocks * qtype.blockSize();
        },
    };
}

pub fn dequantize(qtype: QuantizationType, quantized_data: []const u8, output: []f32, allocator: std.mem.Allocator) !void {
    return switch (qtype) {
        .f32 => blk: {
            if (quantized_data.len != output.len * 4) return error.InvalidF32DataSize;
            const f32_data = std.mem.bytesAsSlice(f32, quantized_data);
            @memcpy(output, f32_data);
            break :blk;
        },
        .f16 => f16m.dequantize(quantized_data, output, allocator),
        .q4_k => q4_k_m.dequantize(quantized_data, output, allocator),
        .q8_0 => q8_0.dequantize(quantized_data, output, allocator),
        .q6_k => q6_km.dequantize(quantized_data, output, allocator),
        else => error.UnsupportedQuantizationType,
    };
}

pub fn calculateDequantizedSize(qtype: QuantizationType, quantized_size: usize) usize {
    return switch (qtype) {
        .f32 => quantized_size / 4,
        .f16 => quantized_size / 2,
        else => (quantized_size / qtype.blockSize()) * qtype.elementsPerBlock(),
    };
}

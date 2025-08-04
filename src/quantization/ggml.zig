const std = @import("std");
const QuantizationType = @import("mod.zig").QuantizationType;

/// GGML quantization type compatibility layer
/// Maps GGML type IDs to our quantization types

pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    
    /// Convert GGML type to our QuantizationType
    pub fn toQuantizationType(self: GGMLType) QuantizationType {
        return switch (self) {
            .f32 => .f32,
            .f16 => .f16,
            .q4_0 => .q4_0,
            .q4_1 => .q4_1,
            .q5_0 => .q5_0,
            .q5_1 => .q5_1,
            .q8_0 => .q8_0,
            .q8_1 => .q8_1,
            .q2_k => .q2_k,
            .q3_k => .q3_k,
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            .q6_k => .q6_k,
            .q8_k => .q8_k,
            .iq2_xxs => .iq2_xxs,
            .iq2_xs => .iq2_xs,
            .iq3_xxs => .iq3_xxs,
            .iq1_s => .iq1_s,
            .iq4_nl => .iq4_nl,
            .iq3_s => .iq3_s,
            .iq2_s => .iq2_s,
            .iq4_xs => .iq4_xs,
        };
    }
    
    /// Get block size for this GGML type
    pub fn blockSize(self: GGMLType) usize {
        return self.toQuantizationType().blockSize();
    }
    
    /// Get elements per block for this GGML type
    pub fn elementsPerBlock(self: GGMLType) usize {
        return self.toQuantizationType().elementsPerBlock();
    }
    
    /// Get bits per weight for this GGML type
    pub fn bitsPerWeight(self: GGMLType) f32 {
        return self.toQuantizationType().bitsPerWeight();
    }
    
    /// Convert to our tensor DataType (for compatibility)
    pub fn toDataType(self: GGMLType) @import("../core/tensor.zig").DataType {
        return switch (self) {
            .f32 => .f32,
            .f16 => .f16,
            else => .f32, // All quantized types become f32 after dequantization
        };
    }
    
    /// Check if this type is supported for dequantization
    pub fn isSupported(self: GGMLType) bool {
        return switch (self) {
            .f32, .f16, .q4_k, .q8_0 => true,
            else => false,
        };
    }
    
    /// Get human-readable name
    pub fn name(self: GGMLType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .q4_0 => "Q4_0",
            .q4_1 => "Q4_1",
            .q5_0 => "Q5_0",
            .q5_1 => "Q5_1",
            .q8_0 => "Q8_0",
            .q8_1 => "Q8_1",
            .q2_k => "Q2_K",
            .q3_k => "Q3_K",
            .q4_k => "Q4_K",
            .q5_k => "Q5_K",
            .q6_k => "Q6_K",
            .q8_k => "Q8_K",
            .iq2_xxs => "IQ2_XXS",
            .iq2_xs => "IQ2_XS",
            .iq3_xxs => "IQ3_XXS",
            .iq1_s => "IQ1_S",
            .iq4_nl => "IQ4_NL",
            .iq3_s => "IQ3_S",
            .iq2_s => "IQ2_S",
            .iq4_xs => "IQ4_XS",
        };
    }
};

/// Convert from raw GGML type ID to GGMLType
pub fn fromTypeId(type_id: u32) !GGMLType {
    return switch (type_id) {
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
        else => {
            std.log.warn("Unknown GGML type ID: {}", .{type_id});
            return error.UnsupportedGGMLType;
        },
    };
}

/// Dequantize using GGML type
pub fn dequantize(
    ggml_type: GGMLType,
    quantized_data: []const u8,
    output: []f32,
    allocator: std.mem.Allocator,
) !void {
    const qtype = ggml_type.toQuantizationType();
    return @import("mod.zig").dequantize(qtype, quantized_data, output, allocator);
}

/// Calculate dequantized size using GGML type
pub fn calculateDequantizedSize(ggml_type: GGMLType, quantized_size: usize) usize {
    const qtype = ggml_type.toQuantizationType();
    return @import("mod.zig").calculateDequantizedSize(qtype, quantized_size);
}

test "ggml type conversion" {
    const testing = std.testing;
    
    // Test type ID conversion
    const f32_type = try fromTypeId(0);
    try testing.expect(f32_type == .f32);
    
    const q4_k_type = try fromTypeId(12);
    try testing.expect(q4_k_type == .q4_k);
    
    // Test quantization type conversion
    try testing.expect(GGMLType.f32.toQuantizationType() == .f32);
    try testing.expect(GGMLType.q4_k.toQuantizationType() == .q4_k);
    
    // Test block sizes
    try testing.expect(GGMLType.f32.blockSize() == 4);
    try testing.expect(GGMLType.f16.blockSize() == 2);
    try testing.expect(GGMLType.q4_k.blockSize() == 144);
    try testing.expect(GGMLType.q8_0.blockSize() == 34);
    
    // Test support status
    try testing.expect(GGMLType.f32.isSupported());
    try testing.expect(GGMLType.f16.isSupported());
    try testing.expect(GGMLType.q4_k.isSupported());
    try testing.expect(GGMLType.q8_0.isSupported());
    try testing.expect(!GGMLType.q4_0.isSupported()); // Not yet implemented
}

test "ggml names" {
    const testing = std.testing;
    
    try testing.expectEqualStrings("F32", GGMLType.f32.name());
    try testing.expectEqualStrings("F16", GGMLType.f16.name());
    try testing.expectEqualStrings("Q4_K", GGMLType.q4_k.name());
    try testing.expectEqualStrings("Q8_0", GGMLType.q8_0.name());
}

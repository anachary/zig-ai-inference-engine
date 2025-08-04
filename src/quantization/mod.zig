const std = @import("std");

/// Format-agnostic weight compression and quantization algorithms
/// Contains pure quantization logic independent of file formats

// Quantization implementations
pub const q4_k_m = @import("q4_k_m.zig");
pub const q8_0 = @import("q8_0.zig");
pub const @"f16" = @import("f16.zig");
pub const ggml = @import("ggml.zig");

// Quantization types and interfaces
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

    pub fn bitsPerWeight(self: QuantizationType) f32 {
        return switch (self) {
            .f32 => 32.0,
            .f16 => 16.0,
            .q4_0, .q4_1, .q4_k, .iq4_nl, .iq4_xs => 4.0,
            .q5_0, .q5_1, .q5_k => 5.0,
            .q8_0, .q8_1, .q8_k => 8.0,
            .q2_k, .iq2_xxs, .iq2_xs, .iq2_s => 2.0,
            .q3_k, .iq3_xxs, .iq3_s => 3.0,
            .q6_k => 6.0,
            .iq1_s => 1.0,
        };
    }
};

pub const QuantizationConfig = struct {
    type: QuantizationType,
    block_size: usize,
    elements_per_block: usize,

    pub fn init(qtype: QuantizationType) QuantizationConfig {
        return QuantizationConfig{
            .type = qtype,
            .block_size = qtype.blockSize(),
            .elements_per_block = qtype.elementsPerBlock(),
        };
    }
};

/// Generic dequantization interface
pub const Dequantizer = struct {
    const Self = @This();

    dequantize_fn: *const fn (self: *anyopaque, quantized_data: []const u8, output: []f32, allocator: std.mem.Allocator) anyerror!void,
    get_config_fn: *const fn (self: *anyopaque) QuantizationConfig,
    deinit_fn: *const fn (self: *anyopaque, allocator: std.mem.Allocator) void,
    impl: *anyopaque,

    pub fn dequantize(self: *Self, quantized_data: []const u8, output: []f32, allocator: std.mem.Allocator) !void {
        return self.dequantize_fn(self.impl, quantized_data, output, allocator);
    }

    pub fn getConfig(self: *Self) QuantizationConfig {
        return self.get_config_fn(self.impl);
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.deinit_fn(self.impl, allocator);
    }
};

/// Create a dequantizer from any implementation
pub fn createDequantizer(allocator: std.mem.Allocator, implementation: anytype) !Dequantizer {
    const T = @TypeOf(implementation);
    const impl_ptr = try allocator.create(T);
    impl_ptr.* = implementation;

    const vtable = struct {
        fn dequantize(impl: *anyopaque, quantized_data: []const u8, output: []f32, alloc: std.mem.Allocator) anyerror!void {
            const self: *T = @ptrCast(@alignCast(impl));
            return self.dequantize(quantized_data, output, alloc);
        }

        fn getConfig(impl: *anyopaque) QuantizationConfig {
            const self: *T = @ptrCast(@alignCast(impl));
            return self.getConfig();
        }

        fn deinit(impl: *anyopaque, alloc: std.mem.Allocator) void {
            const self: *T = @ptrCast(@alignCast(impl));
            self.deinit();
            alloc.destroy(self);
        }
    };

    return Dequantizer{
        .dequantize_fn = vtable.dequantize,
        .get_config_fn = vtable.getConfig,
        .deinit_fn = vtable.deinit,
        .impl = impl_ptr,
    };
}

/// Generic dequantization function that dispatches to specific implementations
pub fn dequantize(
    qtype: QuantizationType,
    quantized_data: []const u8,
    output: []f32,
    allocator: std.mem.Allocator,
) !void {
    switch (qtype) {
        .f32 => {
            if (quantized_data.len != output.len * 4) {
                return error.InvalidF32DataSize;
            }
            const f32_data = std.mem.bytesAsSlice(f32, quantized_data);
            @memcpy(output, f32_data);
        },
        .f16 => try @"f16".dequantize(quantized_data, output, allocator),
        .q4_k => try q4_k_m.dequantize(quantized_data, output, allocator),
        .q8_0 => try q8_0.dequantize(quantized_data, output, allocator),
        else => {
            std.log.warn("Unsupported quantization type: {s}", .{@tagName(qtype)});
            return error.UnsupportedQuantizationType;
        },
    }
}

/// Calculate the number of F32 elements that will result from dequantizing
pub fn calculateDequantizedSize(qtype: QuantizationType, quantized_size: usize) usize {
    return switch (qtype) {
        .f32 => quantized_size / 4,
        .f16 => quantized_size / 2,
        .q4_k => (quantized_size / qtype.blockSize()) * qtype.elementsPerBlock(),
        .q8_0 => (quantized_size / qtype.blockSize()) * qtype.elementsPerBlock(),
        else => 0, // Unsupported
    };
}

test "quantization types" {
    const testing = std.testing;

    // Test block sizes
    try testing.expect(QuantizationType.f32.blockSize() == 4);
    try testing.expect(QuantizationType.f16.blockSize() == 2);
    try testing.expect(QuantizationType.q4_k.blockSize() == 144);
    try testing.expect(QuantizationType.q8_0.blockSize() == 34);

    // Test elements per block
    try testing.expect(QuantizationType.f32.elementsPerBlock() == 1);
    try testing.expect(QuantizationType.q4_k.elementsPerBlock() == 256);
    try testing.expect(QuantizationType.q8_0.elementsPerBlock() == 32);

    // Test bits per weight
    try testing.expect(QuantizationType.f32.bitsPerWeight() == 32.0);
    try testing.expect(QuantizationType.f16.bitsPerWeight() == 16.0);
    try testing.expect(QuantizationType.q4_k.bitsPerWeight() == 4.0);
    try testing.expect(QuantizationType.q8_0.bitsPerWeight() == 8.0);
}

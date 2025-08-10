const std = @import("std");
const qmod = @import("../quantization/mod.zig");

pub fn calcNumel(shape: []const usize) !usize {
    var total: usize = 1;
    for (shape) |d| {
        if (d == 0) return error.InvalidShape;
        const max_before_mul = std.math.maxInt(usize) / d;
        if (total > max_before_mul) return error.InvalidShape;
        total *= d;
    }
    return total;
}

pub fn mapGgmlToQuantType(ggml_type_id: u32) !qmod.QuantizationType {
    return qmod.fromGgml(ggml_type_id);
}

pub fn calcQuantizedSize(ggml_type_id: u32, shape: []const usize) !usize {
    const n = try calcNumel(shape);
    const qtype = try qmod.fromGgml(ggml_type_id);
    const epb = qtype.elementsPerBlock();
    const bsz = qtype.blockSize();
    // For block-quantized types, GGUF stores rows independently rounded to block boundary
    if (epb > 1 and shape.len >= 2) {
        // GGML stores ne0 as columns (fastest-changing), ne1 as rows
        const cols = shape[0];
        const rows = shape[1];
        const blocks_per_row = (cols + epb - 1) / epb;
        return rows * blocks_per_row * bsz;
    }
    return qmod.calculateQuantizedSize(qtype, n);
}

pub fn calcDequantizedLen(ggml_type_id: u32, shape: []const usize) !usize {
    const n = try calcNumel(shape);
    const qtype = try qmod.fromGgml(ggml_type_id);
    return switch (qtype) {
        .f32, .f16 => n,
        else => blk: {
            const epb = qtype.elementsPerBlock();
            const blocks = (n + epb - 1) / epb;
            break :blk blocks * epb;
        },
    };
}

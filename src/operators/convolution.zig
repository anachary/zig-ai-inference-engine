const std = @import("std");
const mod = @import("mod.zig");

const Operator = mod.Operator;
const Tensor = mod.Tensor;
const OpContext = mod.OpContext;

/// Convolution operator (placeholder)
pub const ConvolutionOp = struct {
    pub fn init() ConvolutionOp {
        return ConvolutionOp{};
    }
    
    pub fn deinit(self: *ConvolutionOp) void {
        _ = self;
    }
};

pub fn createConvolutionOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    _ = allocator;
    _ = params;
    return error.NotImplemented;
}

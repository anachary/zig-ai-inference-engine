const std = @import("std");
const mod = @import("mod.zig");

const Operator = mod.Operator;
const Tensor = mod.Tensor;
const OpContext = mod.OpContext;

/// Pooling operator (placeholder)
pub const PoolingOp = struct {
    pub fn init() PoolingOp {
        return PoolingOp{};
    }
    
    pub fn deinit(self: *PoolingOp) void {
        _ = self;
    }
};

pub fn createPoolingOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    _ = allocator;
    _ = params;
    return error.NotImplemented;
}

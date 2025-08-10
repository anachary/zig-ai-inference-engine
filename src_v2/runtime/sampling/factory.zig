const std = @import("std");
const regs = @import("../../core/registries.zig");
const types = @import("../../core/types.zig");

pub const Pipeline = struct {
    transforms: []const regs.LogitTransform,
    selector: regs.Selector,
};

pub fn buildPipeline(allocator: std.mem.Allocator, params: types.SamplingParams) !Pipeline {
    var list = std.ArrayList(regs.LogitTransform).init(allocator);
    defer list.deinit();

    // penalties would go here first (todo)

    if (params.temperature > 0 and !std.math.approxEqAbs(f32, params.temperature, 1.0, 1e-6)) {
        if (regs.getLogitTransform("temperature")) |t| try list.append(t);
    }
    if (params.top_k > 0) {
        if (regs.getLogitTransform("top-k")) |t| try list.append(t);
    }
    if (params.top_p < 0.9999) {
        if (regs.getLogitTransform("top-p")) |t| try list.append(t);
    }

    const selector = blk: {
        if (params.temperature <= 0.1 and params.top_k == 0 and params.top_p >= 0.9999) {
            break :blk regs.getSelector("greedy") orelse return error.MissingRegistry;
        } else {
            break :blk regs.getSelector("multinomial") orelse return error.MissingRegistry;
        }
    };

    const transforms = try allocator.alloc(regs.LogitTransform, list.items.len);
    std.mem.copy(regs.LogitTransform, transforms, list.items);

    return .{ .transforms = transforms, .selector = selector };
}


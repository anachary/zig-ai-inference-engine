const std = @import("std");
const regs = @import("../../core/registries.zig");

fn greedy_select(logits: []const f32, ctx: regs.SamplerCtx) u32 {
    _ = ctx;
    var best: u32 = 0;
    var best_val: f32 = -3.4e38;
    var i: u32 = 0;
    while (i < logits.len) : (i += 1) {
        if (logits[i] > best_val) {
            best = i;
            best_val = logits[i];
        }
    }
    return best;
}

fn multinomial_select(logits: []const f32, ctx: regs.SamplerCtx) u32 {
    // logits expected to be post-softmax probs for multinomial
    const rng_ptr = ctx.rng orelse return greedy_select(logits, ctx);
    var r = rng_ptr.*;
    var cum: f32 = 0;
    const t = r.float(f32);
    var i: usize = 0;
    while (i < logits.len) : (i += 1) {
        cum += logits[i];
        if (t <= cum) return @intCast(i);
    }
    return @intCast(logits.len - 1);
}

pub fn registerAll() !void {
    try regs.registerSelector("greedy", .{ .select = greedy_select });
    try regs.registerSelector("multinomial", .{ .select = multinomial_select });
}

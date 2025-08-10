const std = @import("std");
const regs = @import("../core/registries.zig");
const types = @import("../core/types.zig");

fn sample_greedy(logits: []const f32, params: types.SamplingParams) u32 {
    _ = params;
    var best: u32 = 0;
    var best_val: f32 = -3.4e38;
    var i: u32 = 0;
    while (i < logits.len) : (i += 1) {
        if (logits[i] > best_val) { best = i; best_val = logits[i]; }
    }
    return best;
}

pub fn registerGreedy() !void {
    const s: regs.Sampler = .{ .sample = sample_greedy };
    try regs.registerSampler("greedy", s);
}


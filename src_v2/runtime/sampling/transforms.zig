const std = @import("std");
const regs = @import("../../core/registries.zig");
const types = @import("../../core/types.zig");

fn apply_repetition_penalty(logits: []f32, ctx: regs.SamplerCtx) void {
    const rp = ctx.params.repetition_penalty;
    if (std.math.approxEqAbs(f32, rp, 1.0, 1e-6) or ctx.recent.len == 0) return;
    var i: usize = 0;
    // Build a tiny set of touched ids (naive O(n^2) acceptable for MVP)
    while (i < ctx.recent.len) : (i += 1) {
        const id = ctx.recent[i];
        if (id < logits.len) {
            if (logits[id] > 0) logits[id] /= rp else logits[id] *= rp;
        }
    }
}

fn apply_presence_frequency_penalties(logits: []f32, ctx: regs.SamplerCtx) void {
    if ((ctx.params.presence_penalty <= 0 and ctx.params.frequency_penalty <= 0) or ctx.recent.len == 0) return;
    var counts = std.AutoHashMap(u32, u32).init(std.heap.page_allocator);
    defer counts.deinit();
    var i: usize = 0;
    while (i < ctx.recent.len) : (i += 1) {
        const id = ctx.recent[i];
        const gop = counts.getOrPut(id) catch continue;
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }
    var it = counts.iterator();
    const neg_inf: f32 = -3.4e38;
    while (it.next()) |e| {
        const id = e.key_ptr.*;
        if (id >= logits.len) continue;
        const c = @as(f32, @floatFromInt(e.value_ptr.*));
        // Presence: push down if seen at all
        const pres = if (c > 0) ctx.params.presence_penalty else 0;
        // Frequency: push down proportional to count
        const freq = ctx.params.frequency_penalty * c;
        var v = logits[id] - (pres + freq);
        if (v < neg_inf) v = neg_inf;
        logits[id] = v;
    }
}

fn apply_temperature(logits: []f32, ctx: regs.SamplerCtx) void {
    const t = ctx.params.temperature;
    if (t <= 0.0 or std.math.approxEqAbs(f32, t, 1.0, 1e-6)) return;
    var i: usize = 0;
    while (i < logits.len) : (i += 1) logits[i] /= t;
}

fn apply_top_k(logits: []f32, ctx: regs.SamplerCtx) void {
    const k = ctx.params.top_k;
    if (k == 0 or k >= logits.len) return;
    // Find kth threshold (naive): copy, partial sort desc, set others to -inf
    var allocator = std.heap.page_allocator;
    var tmp = allocator.alloc(f32, logits.len) catch return;
    defer allocator.free(tmp);
    std.mem.copy(f32, tmp, logits);
    std.sort.block(f32, tmp, {}, struct {
        fn lessThan(_: void, a: f32, b: f32) bool {
            return a > b;
        }
    }.lessThan);
    const thresh = tmp[k - 1];
    var i: usize = 0;
    const neg_inf: f32 = -3.4e38;
    while (i < logits.len) : (i += 1) {
        if (logits[i] < thresh) logits[i] = neg_inf;
    }
}

fn apply_top_p(logits: []f32, ctx: regs.SamplerCtx) void {
    const p = ctx.params.top_p;
    if (p >= 0.9999) return;
    // Softmax to probs, sort desc, keep smallest prefix with cumulative>=p, mask others
    var allocator = std.heap.page_allocator;
    var idx = allocator.alloc(usize, logits.len) catch return;
    var probs = allocator.alloc(f32, logits.len) catch {
        allocator.free(idx);
        return;
    };
    defer {
        allocator.free(idx);
        allocator.free(probs);
    }
    var i: usize = 0;
    var maxv: f32 = -3.4e38;
    while (i < logits.len) : (i += 1) {
        if (logits[i] > maxv) maxv = logits[i];
    }
    var sum: f32 = 0;
    i = 0;
    while (i < logits.len) : (i += 1) {
        const e = @exp(logits[i] - maxv);
        probs[i] = e;
        sum += e;
        idx[i] = i;
    }
    i = 0;
    while (i < logits.len) : (i += 1) probs[i] /= sum;
    const Ctx = struct { probs: []f32 };
    std.sort.block(usize, idx, Ctx{ .probs = probs }, struct {
        fn lessThan(sort_ctx: Ctx, a: usize, b: usize) bool {
            return sort_ctx.probs[a] > sort_ctx.probs[b];
        }
    }.lessThan);
    var cum: f32 = 0;
    const neg_inf: f32 = -3.4e38;
    i = 0;
    while (i < idx.len) : (i += 1) {
        const id = idx[i];
        cum += probs[id];
        if (cum >= p) {
            // mask the rest
            var j: usize = i + 1;
            while (j < idx.len) : (j += 1) logits[idx[j]] = neg_inf;
            break;
        }
    }
}

fn apply_min_p(logits: []f32, ctx: regs.SamplerCtx) void {
    const min_p = ctx.params.min_p;
    if (min_p <= 0) return;
    // Typical/min-p: compute probs and keep tokens where prob >= min_p * max_prob
    var allocator = std.heap.page_allocator;
    var probs = allocator.alloc(f32, logits.len) catch return;
    defer allocator.free(probs);
    var i: usize = 0;
    var maxv: f32 = -3.4e38;
    while (i < logits.len) : (i += 1) {
        if (logits[i] > maxv) maxv = logits[i];
    }
    var sum: f32 = 0;
    i = 0;
    while (i < logits.len) : (i += 1) {
        const e = @exp(logits[i] - maxv);
        probs[i] = e;
        sum += e;
    }
    i = 0;
    while (i < logits.len) : (i += 1) probs[i] /= sum;
    var pmax: f32 = 0;
    i = 0;
    while (i < logits.len) : (i += 1) {
        if (probs[i] > pmax) pmax = probs[i];
    }
    const thr = pmax * min_p;
    const neg_inf: f32 = -3.4e38;
    i = 0;
    while (i < logits.len) : (i += 1) {
        if (probs[i] < thr) logits[i] = neg_inf;
    }
}

pub fn registerAll() !void {
    try regs.registerLogitTransform("repetition-penalty", .{ .apply = apply_repetition_penalty });
    try regs.registerLogitTransform("presence-frequency", .{ .apply = apply_presence_frequency_penalties });
    try regs.registerLogitTransform("temperature", .{ .apply = apply_temperature });
    try regs.registerLogitTransform("top-k", .{ .apply = apply_top_k });
    try regs.registerLogitTransform("top-p", .{ .apply = apply_top_p });
    try regs.registerLogitTransform("min-p", .{ .apply = apply_min_p });
}

const std = @import("std");

// Naive CPU causal attention kernel for one decode step t.
// Shapes:
// - num_heads, num_kv_heads, head_dim
// - q: [num_heads * head_dim]
// - k_cache: [(t+1) * num_kv_heads * head_dim]
// - v_cache: [(t+1) * num_kv_heads * head_dim]
// - out: [num_heads * head_dim]
// Mapping for GQA/MQA: each query head h uses kv_head = h % num_kv_heads.
pub fn attention_decode_once(num_heads: usize, num_kv_heads: usize, head_dim: usize, t: usize, q: []const f32, k_cache: []const f32, v_cache: []const f32, out: []f32) void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    // For each query head
    var h: usize = 0;
    while (h < num_heads) : (h += 1) {
        const kv = h % num_kv_heads;
        const q_off = h * head_dim;
        const out_off = h * head_dim;

        // Scores over time 0..t
        var max_score: f32 = -3.4e38;
        var scores = std.heap.page_allocator.alloc(f32, t + 1) catch return;
        defer std.heap.page_allocator.free(scores);
        var tt: usize = 0;
        while (tt <= t) : (tt += 1) {
            const k_off = (tt * num_kv_heads + kv) * head_dim;
            var dot: f32 = 0;
            var i: usize = 0;
            while (i < head_dim) : (i += 1) dot += q[q_off + i] * k_cache[k_off + i];
            scores[tt] = dot * scale;
            if (scores[tt] > max_score) max_score = scores[tt];
        }
        // Softmax (stable)
        var sum: f32 = 0;
        tt = 0;
        while (tt <= t) : (tt += 1) {
            scores[tt] = @exp(scores[tt] - max_score);
            sum += scores[tt];
        }
        if (sum == 0) sum = 1;
        tt = 0;
        while (tt <= t) : (tt += 1) scores[tt] /= sum;

        // Weighted sum of V
        var i: usize = 0;
        while (i < head_dim) : (i += 1) {
            var acc: f32 = 0;
            tt = 0;
            while (tt <= t) : (tt += 1) {
                const v_off = (tt * num_kv_heads + kv) * head_dim;
                acc += scores[tt] * v_cache[v_off + i];
            }
            out[out_off + i] = acc;
        }
    }
}


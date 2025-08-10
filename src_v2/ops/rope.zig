const std = @import("std");

// RoPE (Su et al., 2021) — LLaMA variant
// This function applies RoPE to a SINGLE head's Q and K vectors in-place.
// q, k: length = head_dim (one head)
// pos: sequence index (0-based)
// rope_theta: base theta (e.g., 10000.0 for LLaMA)
pub fn apply_rope_llama(q: []f32, k: []f32, head_dim: usize, pos: usize, rope_theta: f32) void {
    const D = head_dim;
    if (q.len < D or k.len < D) return;

    // For each pair (2i, 2i+1), rotate by angle dependent on i and pos
    var i: usize = 0;
    while (i + 1 < D) : (i += 2) {
        const half_idx = i / 2;
        const freq = std.math.pow(f32, rope_theta, -@as(f32, @floatFromInt(2 * half_idx)) / @as(f32, @floatFromInt(D)));
        const angle = @as(f32, @floatFromInt(pos)) * freq;
        const ca = @cos(angle);
        const sa = @sin(angle);

        const q0 = q[i];
        const q1 = q[i + 1];
        q[i] = q0 * ca - q1 * sa;
        q[i + 1] = q0 * sa + q1 * ca;

        const k0 = k[i];
        const k1 = k[i + 1];
        k[i] = k0 * ca - k1 * sa;
        k[i + 1] = k0 * sa + k1 * ca;
    }
}


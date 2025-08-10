const std = @import("std");

pub fn matmulF32(a: []const f32, b: []const f32, m: usize, k: usize, n: usize, out: []f32) void {
    // a: m x k (row-major), b: k x n (row-major), out: m x n
    // Optimized path when m == 1 (common in decoder): compute one row times B using SIMD over columns.
    if (m == 1) {
        // Use wider SIMD vectors for better performance
        const V = @Vector(16, f32); // AVX-512 width for better throughput
        var j: usize = 0;
        // process 16 columns at a time
        while (j + 16 <= n) : (j += 16) {
            var sum_vec: V = @splat(0.0);
            var p: usize = 0;
            // tight inner loop over k with unrolled SIMD
            @setRuntimeSafety(false);
            while (p + 4 <= k) : (p += 4) {
                const a0: V = @splat(a[p]);
                const a1: V = @splat(a[p + 1]);
                const a2: V = @splat(a[p + 2]);
                const a3: V = @splat(a[p + 3]);
                
                // load 16 contiguous from b rows p..p+3, columns j..j+15
                const b_vec0: V = b[p * n + j .. p * n + j + 16][0..16].*;
                const b_vec1: V = b[(p + 1) * n + j .. (p + 1) * n + j + 16][0..16].*;
                const b_vec2: V = b[(p + 2) * n + j .. (p + 2) * n + j + 16][0..16].*;
                const b_vec3: V = b[(p + 3) * n + j .. (p + 3) * n + j + 16][0..16].*;
                
                sum_vec += a0 * b_vec0 + a1 * b_vec1 + a2 * b_vec2 + a3 * b_vec3;
            }
            // handle remaining k elements
            while (p < k) : (p += 1) {
                const a0: V = @splat(a[p]);
                const b_vec: V = b[p * n + j .. p * n + j + 16][0..16].*;
                sum_vec += a0 * b_vec;
            }
            // store result
            out[j .. j + 16][0..16].* = sum_vec;
        }
        // remainder columns with smaller vectors
        while (j + 8 <= n) : (j += 8) {
            const V8 = @Vector(8, f32);
            var sum_vec: V8 = @splat(0.0);
            var p: usize = 0;
            @setRuntimeSafety(false);
            while (p < k) : (p += 1) {
                const a0: V8 = @splat(a[p]);
                const b_vec: V8 = b[p * n + j .. p * n + j + 8][0..8].*;
                sum_vec += a0 * b_vec;
            }
            out[j .. j + 8][0..8].* = sum_vec;
        }
        // final remainder columns
        while (j < n) : (j += 1) {
            var sum: f32 = 0;
            var p2: usize = 0;
            while (p2 < k) : (p2 += 1) sum += a[p2] * b[p2 * n + j];
            out[j] = sum;
        }
        return;
    }

    // For m > 1, use blocking for cache efficiency
    const block_size = 64; // Cache-friendly block size
    var i: usize = 0;
    while (i < m) : (i += block_size) {
        const i_end = @min(i + block_size, m);
        var j: usize = 0;
        while (j < n) : (j += block_size) {
            const j_end = @min(j + block_size, n);
            var kk: usize = 0;
            while (kk < k) : (kk += block_size) {
                const k_end = @min(kk + block_size, k);
                
                // Block multiplication
                var ii = i;
                while (ii < i_end) : (ii += 1) {
                    var jj = j;
                    while (jj < j_end) : (jj += 1) {
                        var sum: f32 = 0;
                        var kkk = kk;
                        while (kkk < k_end) : (kkk += 1) {
                            sum += a[ii * k + kkk] * b[kkk * n + jj];
                        }
                        out[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

pub fn matmulF32_Bt(a: []const f32, b: []const f32, m: usize, k: usize, n: usize, out: []f32) void {
    // a: m x k, b: (n x k) row-major (i.e., B^T stored as n rows of length k)
    // Compute out = a * b^T -> m x n
    if (m == 1) {
        const V = @Vector(16, f32);
        var j: usize = 0;
        while (j + 16 <= n) : (j += 16) {
            var sum_vec: V = @splat(0.0);
            var p: usize = 0;
            @setRuntimeSafety(false);
            while (p + 4 <= k) : (p += 4) {
                const a0: V = @splat(a[p]);
                const a1: V = @splat(a[p + 1]);
                const a2: V = @splat(a[p + 2]);
                const a3: V = @splat(a[p + 3]);
                
                // gather 16 values: b[row=j..j+15][col=p..p+3] => strides of k
                var tmp0: [16]f32 = undefined;
                var tmp1: [16]f32 = undefined;
                var tmp2: [16]f32 = undefined;
                var tmp3: [16]f32 = undefined;
                
                inline for (0..16) |off| {
                    tmp0[off] = b[(j + off) * k + p];
                    tmp1[off] = b[(j + off) * k + p + 1];
                    tmp2[off] = b[(j + off) * k + p + 2];
                    tmp3[off] = b[(j + off) * k + p + 3];
                }
                
                const b_vec0: V = @bitCast(tmp0);
                const b_vec1: V = @bitCast(tmp1);
                const b_vec2: V = @bitCast(tmp2);
                const b_vec3: V = @bitCast(tmp3);
                
                sum_vec += a0 * b_vec0 + a1 * b_vec1 + a2 * b_vec2 + a3 * b_vec3;
            }
            while (p < k) : (p += 1) {
                const a0: V = @splat(a[p]);
                var tmp: [16]f32 = undefined;
                inline for (0..16) |off| {
                    tmp[off] = b[(j + off) * k + p];
                }
                const b_vec: V = @bitCast(tmp);
                sum_vec += a0 * b_vec;
            }
            out[j .. j + 16][0..16].* = sum_vec;
        }
        while (j + 8 <= n) : (j += 8) {
            const V8 = @Vector(8, f32);
            var sum_vec: V8 = @splat(0.0);
            var p: usize = 0;
            while (p < k) : (p += 1) {
                const a0: V8 = @splat(a[p]);
                var tmp: [8]f32 = undefined;
                inline for (0..8) |off| {
                    tmp[off] = b[(j + off) * k + p];
                }
                const b_vec: V8 = @bitCast(tmp);
                sum_vec += a0 * b_vec;
            }
            out[j .. j + 8][0..8].* = sum_vec;
        }
        while (j < n) : (j += 1) {
            var sum: f32 = 0;
            var p2: usize = 0;
            while (p2 < k) : (p2 += 1) sum += a[p2] * b[j * k + p2];
            out[j] = sum;
        }
        return;
    }

    // Fallback generic for m > 1 with blocking
    const block_size = 64;
    var i: usize = 0;
    while (i < m) : (i += block_size) {
        const i_end = @min(i + block_size, m);
        var j: usize = 0;
        while (j < n) : (j += block_size) {
            const j_end = @min(j + block_size, n);
            var kk: usize = 0;
            while (kk < k) : (kk += block_size) {
                const k_end = @min(kk + block_size, k);
                
                // Block multiplication
                var ii = i;
                while (ii < i_end) : (ii += 1) {
                    var jj = j;
                    while (jj < j_end) : (jj += 1) {
                        var sum: f32 = 0;
                        var kkk = kk;
                        while (kkk < k_end) : (kkk += 1) {
                            sum += a[ii * k + kkk] * b[jj * k + kkk];
                        }
                        out[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

// FUSED QKV PROJECTION - Maximum performance optimization
// Processes Q, K, V projections together to reduce memory bandwidth
pub fn fusedQKVProjection(
    input: []const f32,           // [1, hidden_dim]
    wq: []const f32,              // [hidden_dim, q_out]
    wk: []const f32,              // [hidden_dim, kv_out] 
    wv: []const f32,              // [hidden_dim, kv_out]
    q_out: []f32,                 // [1, q_out]
    k_out: []f32,                 // [1, kv_out]
    v_out: []f32,                 // [1, kv_out]
    hidden_dim: usize,
    q_out_dim: usize,
    kv_out_dim: usize,
) void {
    // Use wider SIMD vectors for maximum throughput
    const V = @Vector(16, f32);
    
    // Process Q projection with SIMD
    var j: usize = 0;
    while (j + 16 <= q_out_dim) : (j += 16) {
        var sum_vec: V = @splat(0.0);
        var p: usize = 0;
        @setRuntimeSafety(false);
        while (p + 4 <= hidden_dim) : (p += 4) {
            const a0: V = @splat(input[p]);
            const a1: V = @splat(input[p + 1]);
            const a2: V = @splat(input[p + 2]);
            const a3: V = @splat(input[p + 3]);
            
            const b_vec0: V = wq[p * q_out_dim + j .. p * q_out_dim + j + 16][0..16].*;
            const b_vec1: V = wq[(p + 1) * q_out_dim + j .. (p + 1) * q_out_dim + j + 16][0..16].*;
            const b_vec2: V = wq[(p + 2) * q_out_dim + j .. (p + 2) * q_out_dim + j + 16][0..16].*;
            const b_vec3: V = wq[(p + 3) * q_out_dim + j .. (p + 3) * q_out_dim + j + 16][0..16].*;
            
            sum_vec += a0 * b_vec0 + a1 * b_vec1 + a2 * b_vec2 + a3 * b_vec3;
        }
        while (p < hidden_dim) : (p += 1) {
            const a0: V = @splat(input[p]);
            const b_vec: V = wq[p * q_out_dim + j .. p * q_out_dim + j + 16][0..16].*;
            sum_vec += a0 * b_vec;
        }
        q_out[j .. j + 16][0..16].* = sum_vec;
    }
    
    // Process K projection with SIMD
    j = 0;
    while (j + 16 <= kv_out_dim) : (j += 16) {
        var sum_vec: V = @splat(0.0);
        var p: usize = 0;
        @setRuntimeSafety(false);
        while (p + 4 <= hidden_dim) : (p += 4) {
            const a0: V = @splat(input[p]);
            const a1: V = @splat(input[p + 1]);
            const a2: V = @splat(input[p + 2]);
            const a3: V = @splat(input[p + 3]);
            
            const b_vec0: V = wk[p * kv_out_dim + j .. p * kv_out_dim + j + 16][0..16].*;
            const b_vec1: V = wk[(p + 1) * kv_out_dim + j .. (p + 1) * kv_out_dim + j + 16][0..16].*;
            const b_vec2: V = wk[(p + 2) * kv_out_dim + j .. (p + 2) * kv_out_dim + j + 16][0..16].*;
            const b_vec3: V = wk[(p + 3) * kv_out_dim + j .. (p + 3) * kv_out_dim + j + 16][0..16].*;
            
            sum_vec += a0 * b_vec0 + a1 * b_vec1 + a2 * b_vec2 + a3 * b_vec3;
        }
        while (p < hidden_dim) : (p += 1) {
            const a0: V = @splat(input[p]);
            const b_vec: V = wk[p * kv_out_dim + j .. p * kv_out_dim + j + 16][0..16].*;
            sum_vec += a0 * b_vec;
        }
        k_out[j .. j + 16][0..16].* = sum_vec;
    }
    
    // Process V projection with SIMD
    j = 0;
    while (j + 16 <= kv_out_dim) : (j += 16) {
        var sum_vec: V = @splat(0.0);
        var p: usize = 0;
        @setRuntimeSafety(false);
        while (p + 4 <= hidden_dim) : (p += 4) {
            const a0: V = @splat(input[p]);
            const a1: V = @splat(input[p + 1]);
            const a2: V = @splat(input[p + 2]);
            const a3: V = @splat(input[p + 3]);
            
            const b_vec0: V = wv[p * kv_out_dim + j .. p * kv_out_dim + j + 16][0..16].*;
            const b_vec1: V = wv[(p + 1) * kv_out_dim + j .. (p + 1) * kv_out_dim + j + 16][0..16].*;
            const b_vec2: V = wv[(p + 2) * kv_out_dim + j .. (p + 2) * kv_out_dim + j + 16][0..16].*;
            const b_vec3: V = wv[(p + 3) * kv_out_dim + j .. (p + 3) * kv_out_dim + j + 16][0..16].*;
            
            sum_vec += a0 * b_vec0 + a1 * b_vec1 + a2 * b_vec2 + a3 * b_vec3;
        }
        while (p < hidden_dim) : (p += 1) {
            const a0: V = @splat(input[p]);
            const b_vec: V = wv[p * kv_out_dim + j .. p * kv_out_dim + j + 16][0..16].*;
            sum_vec += a0 * b_vec;
        }
        v_out[j .. j + 16][0..16].* = sum_vec;
    }
    
    // Handle remainder elements
    for (q_out[q_out_dim - (q_out_dim % 16)..]) |*q| {
        var sum: f32 = 0;
        for (0..hidden_dim) |p| {
            sum += input[p] * wq[p * q_out_dim + @intFromPtr(q) - @intFromPtr(&q_out[0])];
        }
        q.* = sum;
    }
    
    for (k_out[kv_out_dim - (kv_out_dim % 16)..]) |*k| {
        var sum: f32 = 0;
        for (0..hidden_dim) |p| {
            sum += input[p] * wk[p * kv_out_dim + @intFromPtr(k) - @intFromPtr(&k_out[0])];
        }
        k.* = sum;
    }
    
    for (v_out[kv_out_dim - (kv_out_dim % 16)..]) |*v| {
        var sum: f32 = 0;
        for (0..hidden_dim) |p| {
            sum += input[p] * wv[p * kv_out_dim + @intFromPtr(v) - @intFromPtr(&v_out[0])];
        }
        v.* = sum;
    }
}

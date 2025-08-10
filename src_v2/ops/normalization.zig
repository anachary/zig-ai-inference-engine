pub fn rmsnorm(x: []f32, eps: f32) void {
    var sumsq: f32 = 0;
    for (x) |v| sumsq += v * v;
    const mean = sumsq / @as(f32, @floatFromInt(x.len));
    const scale = 1.0 / @sqrt(mean + eps);
    var i: usize = 0;
    while (i < x.len) : (i += 1) x[i] *= scale;
}


const std = @import("std");

/// Layer normalization: normalizes across the feature dimension
/// output = γ * (input - μ) / σ + β
/// where μ is mean, σ is standard deviation, γ is scale, β is bias
pub fn layerNorm(
    input: []const f32,
    output: []f32,
    gamma: []const f32,
    beta: []const f32,
    eps: f32,
) void {
    std.debug.assert(input.len == output.len);
    std.debug.assert(input.len == gamma.len);
    std.debug.assert(input.len == beta.len);
    
    if (input.len == 0) return;
    
    // Compute mean
    var sum: f32 = 0.0;
    for (input) |x| {
        sum += x;
    }
    const mean = sum / @as(f32, @floatFromInt(input.len));
    
    // Compute variance
    var var_sum: f32 = 0.0;
    for (input) |x| {
        const diff = x - mean;
        var_sum += diff * diff;
    }
    const variance = var_sum / @as(f32, @floatFromInt(input.len));
    const std_dev = @sqrt(variance + eps);
    
    // Apply normalization
    for (input, output, gamma, beta) |x, *y, g, b| {
        y.* = g * (x - mean) / std_dev + b;
    }
}

/// Layer normalization without learnable parameters (γ=1, β=0)
pub fn layerNormSimple(input: []const f32, output: []f32, eps: f32) void {
    std.debug.assert(input.len == output.len);
    
    if (input.len == 0) return;
    
    // Compute mean
    var sum: f32 = 0.0;
    for (input) |x| {
        sum += x;
    }
    const mean = sum / @as(f32, @floatFromInt(input.len));
    
    // Compute variance
    var var_sum: f32 = 0.0;
    for (input) |x| {
        const diff = x - mean;
        var_sum += diff * diff;
    }
    const variance = var_sum / @as(f32, @floatFromInt(input.len));
    const std_dev = @sqrt(variance + eps);
    
    // Apply normalization
    for (input, output) |x, *y| {
        y.* = (x - mean) / std_dev;
    }
}

/// RMS (Root Mean Square) normalization
/// Used in some transformer variants like LLaMA
/// output = input / sqrt(mean(input²) + eps) * gamma
pub fn rmsNorm(
    input: []const f32,
    output: []f32,
    gamma: []const f32,
    eps: f32,
) void {
    std.debug.assert(input.len == output.len);
    std.debug.assert(input.len == gamma.len);
    
    if (input.len == 0) return;
    
    // Compute mean of squares
    var sum_squares: f32 = 0.0;
    for (input) |x| {
        sum_squares += x * x;
    }
    const mean_squares = sum_squares / @as(f32, @floatFromInt(input.len));
    const rms = @sqrt(mean_squares + eps);
    
    // Apply RMS normalization
    for (input, output, gamma) |x, *y, g| {
        y.* = (x / rms) * g;
    }
}

/// RMS normalization without learnable parameters
pub fn rmsNormSimple(input: []const f32, output: []f32, eps: f32) void {
    std.debug.assert(input.len == output.len);
    
    if (input.len == 0) return;
    
    // Compute mean of squares
    var sum_squares: f32 = 0.0;
    for (input) |x| {
        sum_squares += x * x;
    }
    const mean_squares = sum_squares / @as(f32, @floatFromInt(input.len));
    const rms = @sqrt(mean_squares + eps);
    
    // Apply RMS normalization
    for (input, output) |x, *y| {
        y.* = x / rms;
    }
}

/// Batch normalization: normalizes across the batch dimension
/// Typically used during training, but can be used in inference with pre-computed statistics
pub fn batchNorm(
    input: []const f32,
    output: []f32,
    gamma: []const f32,
    beta: []const f32,
    running_mean: []const f32,
    running_var: []const f32,
    eps: f32,
) void {
    std.debug.assert(input.len == output.len);
    std.debug.assert(input.len == gamma.len);
    std.debug.assert(input.len == beta.len);
    std.debug.assert(input.len == running_mean.len);
    std.debug.assert(input.len == running_var.len);
    
    // Apply batch normalization using pre-computed statistics
    for (input, output, gamma, beta, running_mean, running_var) |x, *y, g, b, mean, variance| {
        const std_dev = @sqrt(variance + eps);
        y.* = g * (x - mean) / std_dev + b;
    }
}

/// Group normalization: divides channels into groups and normalizes within each group
pub fn groupNorm(
    input: []const f32,
    output: []f32,
    gamma: []const f32,
    beta: []const f32,
    num_groups: usize,
    channels_per_group: usize,
    eps: f32,
) void {
    std.debug.assert(input.len == output.len);
    std.debug.assert(input.len == gamma.len);
    std.debug.assert(input.len == beta.len);
    std.debug.assert(input.len == num_groups * channels_per_group);
    
    for (0..num_groups) |group| {
        const start_idx = group * channels_per_group;
        const end_idx = start_idx + channels_per_group;
        
        const group_input = input[start_idx..end_idx];
        const group_output = output[start_idx..end_idx];
        const group_gamma = gamma[start_idx..end_idx];
        const group_beta = beta[start_idx..end_idx];
        
        layerNorm(group_input, group_output, group_gamma, group_beta, eps);
    }
}

/// Instance normalization: normalizes each sample independently
pub fn instanceNorm(
    input: []const f32,
    output: []f32,
    gamma: []const f32,
    beta: []const f32,
    eps: f32,
) void {
    // Instance norm is the same as layer norm for 1D case
    layerNorm(input, output, gamma, beta, eps);
}

/// Compute statistics for normalization layers
pub const NormStats = struct {
    mean: f32,
    variance: f32,
    std_dev: f32,
    
    pub fn compute(input: []const f32, eps: f32) NormStats {
        if (input.len == 0) {
            return NormStats{ .mean = 0.0, .variance = 0.0, .std_dev = @sqrt(eps) };
        }
        
        // Compute mean
        var sum: f32 = 0.0;
        for (input) |x| {
            sum += x;
        }
        const mean = sum / @as(f32, @floatFromInt(input.len));
        
        // Compute variance
        var var_sum: f32 = 0.0;
        for (input) |x| {
            const diff = x - mean;
            var_sum += diff * diff;
        }
        const variance = var_sum / @as(f32, @floatFromInt(input.len));
        const std_dev = @sqrt(variance + eps);
        
        return NormStats{
            .mean = mean,
            .variance = variance,
            .std_dev = std_dev,
        };
    }
};

/// Normalization layer types
pub const NormType = enum {
    layer_norm,
    rms_norm,
    batch_norm,
    group_norm,
    instance_norm,
};

test "layer normalization" {
    const testing = std.testing;
    
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const gamma = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const beta = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    
    layerNorm(&input, &output, &gamma, &beta, 1e-5);
    
    // Check that output has approximately zero mean
    var sum: f32 = 0.0;
    for (output) |val| {
        sum += val;
    }
    const mean = sum / @as(f32, @floatFromInt(output.len));
    try testing.expectApproxEqRel(mean, 0.0, 1e-6);
    
    // Check that output has approximately unit variance
    var var_sum: f32 = 0.0;
    for (output) |val| {
        var_sum += val * val;
    }
    const variance = var_sum / @as(f32, @floatFromInt(output.len));
    try testing.expectApproxEqRel(variance, 1.0, 1e-6);
}

test "rms normalization" {
    const testing = std.testing;
    
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const gamma = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    
    rmsNorm(&input, &output, &gamma, 1e-5);
    
    // Check that RMS of output is approximately 1
    var sum_squares: f32 = 0.0;
    for (output) |val| {
        sum_squares += val * val;
    }
    const rms = @sqrt(sum_squares / @as(f32, @floatFromInt(output.len)));
    try testing.expectApproxEqRel(rms, 1.0, 1e-6);
}

test "norm stats computation" {
    const testing = std.testing;
    
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const stats = NormStats.compute(&input, 1e-5);
    
    try testing.expectApproxEqRel(stats.mean, 2.5, 1e-6);
    try testing.expectApproxEqRel(stats.variance, 1.25, 1e-6);
}

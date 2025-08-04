const std = @import("std");
const simd = @import("simd.zig");

/// Softmax activation function
/// Computes softmax(x) = exp(x_i) / sum(exp(x_j)) for all j
pub fn softmax(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);
    
    if (input.len == 0) return;
    
    // Find maximum for numerical stability
    const max_val = std.mem.max(f32, input);
    
    // Compute exp(x_i - max) and sum
    var sum: f32 = 0.0;
    for (input, output) |x, *y| {
        const exp_val = @exp(x - max_val);
        y.* = exp_val;
        sum += exp_val;
    }
    
    // Normalize by sum
    const inv_sum = 1.0 / sum;
    for (output) |*y| {
        y.* *= inv_sum;
    }
}

/// Softmax with temperature scaling
pub fn softmaxWithTemperature(input: []const f32, output: []f32, temperature: f32) void {
    std.debug.assert(input.len == output.len);
    std.debug.assert(temperature > 0.0);
    
    if (input.len == 0) return;
    
    // Scale by temperature and find maximum
    var max_val: f32 = -std.math.inf(f32);
    for (input) |x| {
        const scaled = x / temperature;
        max_val = @max(max_val, scaled);
    }
    
    // Compute exp((x_i / T) - max) and sum
    var sum: f32 = 0.0;
    for (input, output) |x, *y| {
        const exp_val = @exp((x / temperature) - max_val);
        y.* = exp_val;
        sum += exp_val;
    }
    
    // Normalize by sum
    const inv_sum = 1.0 / sum;
    for (output) |*y| {
        y.* *= inv_sum;
    }
}

/// ReLU activation function: f(x) = max(0, x)
pub fn relu(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);
    
    for (input, output) |x, *y| {
        y.* = @max(0.0, x);
    }
}

/// GELU activation function: f(x) = x * Φ(x)
/// where Φ(x) is the cumulative distribution function of the standard normal distribution
/// Approximation: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
pub fn gelu(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);
    
    const sqrt_2_over_pi: f32 = @sqrt(2.0 / std.math.pi);
    const coeff: f32 = 0.044715;
    
    for (input, output) |x, *y| {
        const x_cubed = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        const tanh_val = std.math.tanh(inner);
        y.* = 0.5 * x * (1.0 + tanh_val);
    }
}

/// SiLU (Swish) activation function: f(x) = x * sigmoid(x)
pub fn silu(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);
    
    for (input, output) |x, *y| {
        const sigmoid_val = 1.0 / (1.0 + @exp(-x));
        y.* = x * sigmoid_val;
    }
}

/// Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
pub fn sigmoid(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);
    
    for (input, output) |x, *y| {
        y.* = 1.0 / (1.0 + @exp(-x));
    }
}

/// Tanh activation function: f(x) = tanh(x)
pub fn tanh_activation(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);
    
    for (input, output) |x, *y| {
        y.* = std.math.tanh(x);
    }
}

/// Leaky ReLU activation function: f(x) = max(αx, x)
pub fn leakyRelu(input: []const f32, output: []f32, alpha: f32) void {
    std.debug.assert(input.len == output.len);
    
    for (input, output) |x, *y| {
        y.* = if (x > 0.0) x else alpha * x;
    }
}

/// ELU activation function: f(x) = x if x > 0, α(exp(x) - 1) if x ≤ 0
pub fn elu(input: []const f32, output: []f32, alpha: f32) void {
    std.debug.assert(input.len == output.len);
    
    for (input, output) |x, *y| {
        y.* = if (x > 0.0) x else alpha * (@exp(x) - 1.0);
    }
}

/// Apply activation function in-place
pub fn applyInPlace(activation_type: ActivationType, data: []f32) void {
    switch (activation_type) {
        .relu => relu(data, data),
        .gelu => gelu(data, data),
        .silu => silu(data, data),
        .sigmoid => sigmoid(data, data),
        .tanh => tanh_activation(data, data),
        .leaky_relu => |alpha| leakyRelu(data, data, alpha),
        .elu => |alpha| elu(data, data, alpha),
    }
}

/// Activation function types
pub const ActivationType = union(enum) {
    relu,
    gelu,
    silu,
    sigmoid,
    tanh,
    leaky_relu: f32, // alpha parameter
    elu: f32, // alpha parameter
};

/// Log-softmax for numerical stability in loss computation
pub fn logSoftmax(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);
    
    if (input.len == 0) return;
    
    // Find maximum for numerical stability
    const max_val = std.mem.max(f32, input);
    
    // Compute log-sum-exp
    var log_sum_exp: f32 = 0.0;
    for (input) |x| {
        log_sum_exp += @exp(x - max_val);
    }
    log_sum_exp = @log(log_sum_exp) + max_val;
    
    // Compute log-softmax
    for (input, output) |x, *y| {
        y.* = x - log_sum_exp;
    }
}

/// Compute cross-entropy loss between predictions and targets
pub fn crossEntropyLoss(predictions: []const f32, targets: []const f32) f32 {
    std.debug.assert(predictions.len == targets.len);
    
    var loss: f32 = 0.0;
    for (predictions, targets) |pred, target| {
        // Clamp prediction to avoid log(0)
        const clamped_pred = @max(pred, 1e-15);
        loss -= target * @log(clamped_pred);
    }
    
    return loss;
}

test "softmax basic" {
    const testing = std.testing;
    
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };
    
    softmax(&input, &output);
    
    // Check that probabilities sum to 1
    var sum: f32 = 0.0;
    for (output) |val| {
        sum += val;
    }
    try testing.expectApproxEqRel(sum, 1.0, 1e-6);
    
    // Check that output is in ascending order (since input is ascending)
    try testing.expect(output[0] < output[1]);
    try testing.expect(output[1] < output[2]);
}

test "relu activation" {
    const testing = std.testing;
    
    const input = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    
    relu(&input, &output);
    
    try testing.expect(output[0] == 0.0);
    try testing.expect(output[1] == 0.0);
    try testing.expect(output[2] == 0.0);
    try testing.expect(output[3] == 1.0);
    try testing.expect(output[4] == 2.0);
}

test "gelu activation" {
    const testing = std.testing;
    
    const input = [_]f32{ -1.0, 0.0, 1.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };
    
    gelu(&input, &output);
    
    // GELU(0) should be approximately 0
    try testing.expectApproxEqRel(output[1], 0.0, 1e-6);
    
    // GELU should be approximately identity for large positive values
    // GELU(1) ≈ 0.841
    try testing.expectApproxEqRel(output[2], 0.841, 1e-2);
}

test "sigmoid activation" {
    const testing = std.testing;
    
    const input = [_]f32{ -1000.0, 0.0, 1000.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };
    
    sigmoid(&input, &output);
    
    try testing.expectApproxEqRel(output[0], 0.0, 1e-6);
    try testing.expectApproxEqRel(output[1], 0.5, 1e-6);
    try testing.expectApproxEqRel(output[2], 1.0, 1e-6);
}

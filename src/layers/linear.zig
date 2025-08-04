const std = @import("std");
const matrix = @import("../math/matrix.zig");

const Matrix = matrix.Matrix;

/// Linear (fully connected) layer: y = xW + b
pub const Linear = struct {
    input_size: usize,
    output_size: usize,
    weight: Matrix,     // [input_size, output_size]
    bias: ?[]f32,       // [output_size] - optional
    allocator: std.mem.Allocator,
    
    pub fn init(
        allocator: std.mem.Allocator,
        input_size: usize,
        output_size: usize,
        use_bias: bool,
    ) !Linear {
        var weight = try Matrix.init(allocator, input_size, output_size);
        
        // Xavier/Glorot initialization
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = rng.random();
        const xavier_std = @sqrt(2.0 / @as(f32, @floatFromInt(input_size + output_size)));
        
        for (weight.data) |*val| {
            val.* = random.floatNorm(f32) * xavier_std;
        }
        
        var bias: ?[]f32 = null;
        if (use_bias) {
            bias = try allocator.alloc(f32, output_size);
            @memset(bias.?, 0.0); // Initialize bias to zero
        }
        
        return Linear{
            .input_size = input_size,
            .output_size = output_size,
            .weight = weight,
            .bias = bias,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Linear) void {
        self.weight.deinit();
        if (self.bias) |bias| {
            self.allocator.free(bias);
        }
    }
    
    /// Forward pass: output = input * weight + bias
    pub fn forward(self: *Linear, input: Matrix, output: *Matrix) !void {
        const batch_size = input.rows;
        std.debug.assert(input.cols == self.input_size);
        std.debug.assert(output.rows == batch_size and output.cols == self.output_size);
        
        // Matrix multiplication: output = input * weight
        try matrix.matmul(input, self.weight, output);
        
        // Add bias if present
        if (self.bias) |bias| {
            for (0..batch_size) |i| {
                const output_row = output.getRow(i);
                for (output_row, bias) |*out_val, bias_val| {
                    out_val.* += bias_val;
                }
            }
        }
    }
    
    /// Forward pass for single vector input
    pub fn forwardVector(self: *Linear, input: []const f32, output: []f32) !void {
        std.debug.assert(input.len == self.input_size);
        std.debug.assert(output.len == self.output_size);
        
        // Matrix-vector multiplication
        try matrix.matvec(self.weight, input, output);
        
        // Add bias if present
        if (self.bias) |bias| {
            for (output, bias) |*out_val, bias_val| {
                out_val.* += bias_val;
            }
        }
    }
    
    /// Load weights from external data
    pub fn loadWeights(self: *Linear, weight_data: []const f32, bias_data: ?[]const f32) !void {
        if (weight_data.len != self.weight.data.len) {
            return error.WeightSizeMismatch;
        }
        
        @memcpy(self.weight.data, weight_data);
        
        if (bias_data) |bias| {
            if (self.bias == null) {
                return error.BiasNotExpected;
            }
            if (bias.len != self.bias.?.len) {
                return error.BiasSizeMismatch;
            }
            @memcpy(self.bias.?, bias);
        }
    }
    
    /// Get parameter count
    pub fn getParameterCount(self: *Linear) usize {
        var count = self.weight.data.len;
        if (self.bias) |bias| {
            count += bias.len;
        }
        return count;
    }
};

/// Multi-layer perceptron (stack of linear layers)
pub const MLP = struct {
    layers: []Linear,
    activations: []ActivationType,
    allocator: std.mem.Allocator,
    
    pub const ActivationType = enum {
        none,
        relu,
        gelu,
        silu,
        tanh,
        sigmoid,
    };
    
    pub fn init(
        allocator: std.mem.Allocator,
        layer_sizes: []const usize,
        activations: []const ActivationType,
        use_bias: bool,
    ) !MLP {
        std.debug.assert(layer_sizes.len >= 2);
        std.debug.assert(activations.len == layer_sizes.len - 1);
        
        const num_layers = layer_sizes.len - 1;
        var layers = try allocator.alloc(Linear, num_layers);
        
        for (0..num_layers) |i| {
            layers[i] = try Linear.init(allocator, layer_sizes[i], layer_sizes[i + 1], use_bias);
        }
        
        const activation_copy = try allocator.dupe(ActivationType, activations);
        
        return MLP{
            .layers = layers,
            .activations = activation_copy,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MLP) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.activations);
    }
    
    /// Forward pass through all layers
    pub fn forward(self: *MLP, input: Matrix, output: *Matrix, temp_buffers: []Matrix) !void {
        std.debug.assert(temp_buffers.len >= self.layers.len - 1);
        
        var current_input = input;
        
        for (self.layers, self.activations, 0..) |*layer, activation, i| {
            var current_output = if (i == self.layers.len - 1) output else &temp_buffers[i];
            
            try layer.forward(current_input, current_output);
            
            // Apply activation function
            if (activation != .none) {
                applyActivation(activation, current_output.data);
            }
            
            current_input = current_output.*;
        }
    }
    
    fn applyActivation(activation: ActivationType, data: []f32) void {
        switch (activation) {
            .none => {},
            .relu => {
                for (data) |*val| {
                    val.* = @max(0.0, val.*);
                }
            },
            .gelu => {
                const sqrt_2_over_pi: f32 = @sqrt(2.0 / std.math.pi);
                const coeff: f32 = 0.044715;
                
                for (data) |*val| {
                    const x = val.*;
                    const x_cubed = x * x * x;
                    const inner = sqrt_2_over_pi * (x + coeff * x_cubed);
                    const tanh_val = std.math.tanh(inner);
                    val.* = 0.5 * x * (1.0 + tanh_val);
                }
            },
            .silu => {
                for (data) |*val| {
                    const x = val.*;
                    const sigmoid_val = 1.0 / (1.0 + @exp(-x));
                    val.* = x * sigmoid_val;
                }
            },
            .tanh => {
                for (data) |*val| {
                    val.* = std.math.tanh(val.*);
                }
            },
            .sigmoid => {
                for (data) |*val| {
                    val.* = 1.0 / (1.0 + @exp(-val.*));
                }
            },
        }
    }
};

test "linear layer forward pass" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var linear = try Linear.init(allocator, 3, 2, true);
    defer linear.deinit();
    
    // Set known weights for testing
    linear.weight.set(0, 0, 1.0); linear.weight.set(0, 1, 2.0);
    linear.weight.set(1, 0, 3.0); linear.weight.set(1, 1, 4.0);
    linear.weight.set(2, 0, 5.0); linear.weight.set(2, 1, 6.0);
    
    if (linear.bias) |bias| {
        bias[0] = 0.1;
        bias[1] = 0.2;
    }
    
    var input = try Matrix.init(allocator, 1, 3);
    defer input.deinit();
    input.set(0, 0, 1.0);
    input.set(0, 1, 2.0);
    input.set(0, 2, 3.0);
    
    var output = try Matrix.init(allocator, 1, 2);
    defer output.deinit();
    
    try linear.forward(input, &output);
    
    // Expected: [1*1 + 2*3 + 3*5 + 0.1, 1*2 + 2*4 + 3*6 + 0.2] = [22.1, 28.2]
    try testing.expectApproxEqRel(output.get(0, 0), 22.1, 1e-6);
    try testing.expectApproxEqRel(output.get(0, 1), 28.2, 1e-6);
}

test "linear layer vector forward" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var linear = try Linear.init(allocator, 2, 3, false);
    defer linear.deinit();
    
    // Set identity-like weights
    linear.weight.set(0, 0, 1.0); linear.weight.set(0, 1, 0.0); linear.weight.set(0, 2, 0.0);
    linear.weight.set(1, 0, 0.0); linear.weight.set(1, 1, 1.0); linear.weight.set(1, 2, 1.0);
    
    const input = [_]f32{ 2.0, 3.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };
    
    try linear.forwardVector(&input, &output);
    
    try testing.expect(output[0] == 2.0);
    try testing.expect(output[1] == 3.0);
    try testing.expect(output[2] == 3.0);
}

test "mlp creation and parameter count" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const layer_sizes = [_]usize{ 4, 8, 2 };
    const activations = [_]MLP.ActivationType{ .relu, .none };
    
    var mlp = try MLP.init(allocator, &layer_sizes, &activations, true);
    defer mlp.deinit();
    
    try testing.expect(mlp.layers.len == 2);
    try testing.expect(mlp.layers[0].input_size == 4);
    try testing.expect(mlp.layers[0].output_size == 8);
    try testing.expect(mlp.layers[1].input_size == 8);
    try testing.expect(mlp.layers[1].output_size == 2);
}

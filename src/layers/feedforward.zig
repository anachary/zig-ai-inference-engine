const std = @import("std");
const matrix = @import("../math/matrix.zig");
const activations = @import("../math/activations.zig");
const linear = @import("linear.zig");

const Matrix = matrix.Matrix;
const Linear = linear.Linear;

/// Feed-forward network used in transformer blocks
/// Typically: Linear -> Activation -> Linear
pub const FeedForward = struct {
    d_model: usize,
    d_ff: usize,
    activation_type: ActivationType,
    
    // Layers
    linear1: Linear,    // [d_model, d_ff]
    linear2: Linear,    // [d_ff, d_model]
    
    // Optional gating (for GLU variants)
    gate_linear: ?Linear, // [d_model, d_ff]
    
    allocator: std.mem.Allocator,
    
    pub const ActivationType = enum {
        relu,
        gelu,
        silu,
        glu,      // Gated Linear Unit
        swiglu,   // SwiGLU (SiLU + GLU)
        geglu,    // GeGLU (GELU + GLU)
    };
    
    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        d_ff: usize,
        activation_type: ActivationType,
        use_bias: bool,
    ) !FeedForward {
        var linear1 = try Linear.init(allocator, d_model, d_ff, use_bias);
        var linear2 = try Linear.init(allocator, d_ff, d_model, use_bias);
        
        // For gated variants, we need an additional linear layer
        var gate_linear: ?Linear = null;
        if (isGatedActivation(activation_type)) {
            gate_linear = try Linear.init(allocator, d_model, d_ff, use_bias);
        }
        
        return FeedForward{
            .d_model = d_model,
            .d_ff = d_ff,
            .activation_type = activation_type,
            .linear1 = linear1,
            .linear2 = linear2,
            .gate_linear = gate_linear,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *FeedForward) void {
        self.linear1.deinit();
        self.linear2.deinit();
        if (self.gate_linear) |*gate| {
            gate.deinit();
        }
    }
    
    /// Forward pass through feed-forward network
    pub fn forward(self: *FeedForward, input: Matrix, output: *Matrix) !void {
        const batch_size = input.rows;
        std.debug.assert(input.cols == self.d_model);
        std.debug.assert(output.rows == batch_size and output.cols == self.d_model);
        
        // Intermediate buffer for first linear layer output
        var intermediate = try Matrix.init(self.allocator, batch_size, self.d_ff);
        defer intermediate.deinit();
        
        // First linear transformation
        try self.linear1.forward(input, &intermediate);
        
        // Apply activation (with optional gating)
        if (self.gate_linear) |*gate| {
            var gate_output = try Matrix.init(self.allocator, batch_size, self.d_ff);
            defer gate_output.deinit();
            
            try gate.forward(input, &gate_output);
            try self.applyGatedActivation(&intermediate, &gate_output);
        } else {
            try self.applyActivation(intermediate.data);
        }
        
        // Second linear transformation
        try self.linear2.forward(intermediate, output);
    }
    
    fn applyActivation(self: *FeedForward, data: []f32) !void {
        switch (self.activation_type) {
            .relu => activations.relu(data, data),
            .gelu => activations.gelu(data, data),
            .silu => activations.silu(data, data),
            else => return error.UnsupportedActivation,
        }
    }
    
    fn applyGatedActivation(self: *FeedForward, main: *Matrix, gate: *Matrix) !void {
        std.debug.assert(main.rows == gate.rows and main.cols == gate.cols);
        
        switch (self.activation_type) {
            .glu => {
                // GLU: main ⊙ sigmoid(gate)
                activations.sigmoid(gate.data, gate.data);
                try matrix.hadamard(main.*, gate.*, main);
            },
            .swiglu => {
                // SwiGLU: main ⊙ SiLU(gate)
                activations.silu(gate.data, gate.data);
                try matrix.hadamard(main.*, gate.*, main);
            },
            .geglu => {
                // GeGLU: main ⊙ GELU(gate)
                activations.gelu(gate.data, gate.data);
                try matrix.hadamard(main.*, gate.*, main);
            },
            else => return error.UnsupportedGatedActivation,
        }
    }
    
    fn isGatedActivation(activation_type: ActivationType) bool {
        return switch (activation_type) {
            .glu, .swiglu, .geglu => true,
            else => false,
        };
    }
    
    /// Load weights from external data
    pub fn loadWeights(
        self: *FeedForward,
        linear1_weights: []const f32,
        linear1_bias: ?[]const f32,
        linear2_weights: []const f32,
        linear2_bias: ?[]const f32,
        gate_weights: ?[]const f32,
        gate_bias: ?[]const f32,
    ) !void {
        try self.linear1.loadWeights(linear1_weights, linear1_bias);
        try self.linear2.loadWeights(linear2_weights, linear2_bias);
        
        if (self.gate_linear) |*gate| {
            if (gate_weights == null) {
                return error.GateWeightsRequired;
            }
            try gate.loadWeights(gate_weights.?, gate_bias);
        }
    }
    
    /// Get total parameter count
    pub fn getParameterCount(self: *FeedForward) usize {
        var count = self.linear1.getParameterCount() + self.linear2.getParameterCount();
        if (self.gate_linear) |*gate| {
            count += gate.getParameterCount();
        }
        return count;
    }
};

/// Mixture of Experts (MoE) feed-forward layer
pub const MixtureOfExperts = struct {
    num_experts: usize,
    top_k: usize,
    d_model: usize,
    d_ff: usize,
    
    experts: []FeedForward,
    gate: Linear, // Router network
    
    allocator: std.mem.Allocator,
    
    pub fn init(
        allocator: std.mem.Allocator,
        num_experts: usize,
        top_k: usize,
        d_model: usize,
        d_ff: usize,
        activation_type: FeedForward.ActivationType,
    ) !MixtureOfExperts {
        std.debug.assert(top_k <= num_experts);
        
        var experts = try allocator.alloc(FeedForward, num_experts);
        for (experts) |*expert| {
            expert.* = try FeedForward.init(allocator, d_model, d_ff, activation_type, true);
        }
        
        var gate = try Linear.init(allocator, d_model, num_experts, false);
        
        return MixtureOfExperts{
            .num_experts = num_experts,
            .top_k = top_k,
            .d_model = d_model,
            .d_ff = d_ff,
            .experts = experts,
            .gate = gate,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MixtureOfExperts) void {
        for (self.experts) |*expert| {
            expert.deinit();
        }
        self.allocator.free(self.experts);
        self.gate.deinit();
    }
    
    /// Forward pass with expert routing
    pub fn forward(self: *MixtureOfExperts, input: Matrix, output: *Matrix) !void {
        const batch_size = input.rows;
        std.debug.assert(input.cols == self.d_model);
        std.debug.assert(output.rows == batch_size and output.cols == self.d_model);
        
        // Compute gating scores
        var gate_scores = try Matrix.init(self.allocator, batch_size, self.num_experts);
        defer gate_scores.deinit();
        
        try self.gate.forward(input, &gate_scores);
        
        // Apply softmax to get probabilities
        for (0..batch_size) |i| {
            const scores_row = gate_scores.getRow(i);
            activations.softmax(scores_row, scores_row);
        }
        
        // Initialize output to zero
        output.fill(0.0);
        
        // For each sample in the batch
        for (0..batch_size) |i| {
            const input_row = input.getRow(i);
            const output_row = output.getRow(i);
            const scores_row = gate_scores.getRow(i);
            
            // Find top-k experts
            var expert_indices = try self.allocator.alloc(usize, self.num_experts);
            defer self.allocator.free(expert_indices);
            
            for (expert_indices, 0..) |*idx, j| {
                idx.* = j;
            }
            
            // Sort by scores (descending)
            std.sort.heap(usize, expert_indices, scores_row, struct {
                fn lessThan(context: []f32, a: usize, b: usize) bool {
                    return context[a] > context[b];
                }
            }.lessThan);
            
            // Compute weighted sum of top-k experts
            var total_weight: f32 = 0.0;
            for (expert_indices[0..self.top_k]) |expert_idx| {
                total_weight += scores_row[expert_idx];
            }
            
            for (expert_indices[0..self.top_k]) |expert_idx| {
                const weight = scores_row[expert_idx] / total_weight;
                
                // Create single-row matrices for expert computation
                var expert_input = try Matrix.init(self.allocator, 1, self.d_model);
                defer expert_input.deinit();
                var expert_output = try Matrix.init(self.allocator, 1, self.d_model);
                defer expert_output.deinit();
                
                @memcpy(expert_input.getRow(0), input_row);
                
                try self.experts[expert_idx].forward(expert_input, &expert_output);
                
                const expert_output_row = expert_output.getRow(0);
                for (output_row, expert_output_row) |*out_val, expert_val| {
                    out_val.* += weight * expert_val;
                }
            }
        }
    }
};

test "feedforward basic" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var ff = try FeedForward.init(allocator, 4, 8, .relu, true);
    defer ff.deinit();
    
    var input = try Matrix.initRandom(allocator, 2, 4, std.rand.DefaultPrng.init(42).random());
    defer input.deinit();
    
    var output = try Matrix.init(allocator, 2, 4);
    defer output.deinit();
    
    try ff.forward(input, &output);
    
    // Basic sanity check - output should not be all zeros
    var has_nonzero = false;
    for (output.data) |val| {
        if (val != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "feedforward gated" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var ff = try FeedForward.init(allocator, 4, 8, .swiglu, false);
    defer ff.deinit();
    
    try testing.expect(ff.gate_linear != null);
    
    var input = try Matrix.initRandom(allocator, 1, 4, std.rand.DefaultPrng.init(42).random());
    defer input.deinit();
    
    var output = try Matrix.init(allocator, 1, 4);
    defer output.deinit();
    
    try ff.forward(input, &output);
    
    // Should complete without error
    try testing.expect(true);
}

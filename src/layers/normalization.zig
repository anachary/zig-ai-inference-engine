const std = @import("std");
const matrix = @import("../math/matrix.zig");
const normalization_math = @import("../math/normalization.zig");

const Matrix = matrix.Matrix;

/// Layer normalization layer
pub const LayerNorm = struct {
    d_model: usize,
    gamma: []f32,  // Scale parameters
    beta: []f32,   // Bias parameters
    eps: f32,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, d_model: usize) !LayerNorm {
        var gamma = try allocator.alloc(f32, d_model);
        var beta = try allocator.alloc(f32, d_model);
        
        // Initialize gamma to 1, beta to 0
        for (gamma) |*g| g.* = 1.0;
        for (beta) |*b| b.* = 0.0;
        
        return LayerNorm{
            .d_model = d_model,
            .gamma = gamma,
            .beta = beta,
            .eps = 1e-5,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *LayerNorm) void {
        self.allocator.free(self.gamma);
        self.allocator.free(self.beta);
    }
    
    /// Forward pass
    pub fn forward(self: *LayerNorm, input: Matrix, output: *Matrix) !void {
        const batch_size = input.rows;
        std.debug.assert(input.cols == self.d_model);
        std.debug.assert(output.rows == batch_size and output.cols == self.d_model);
        
        // Apply layer normalization to each row
        for (0..batch_size) |i| {
            const input_row = input.getRow(i);
            const output_row = output.getRow(i);
            
            normalization_math.layerNorm(input_row, output_row, self.gamma, self.beta, self.eps);
        }
    }
    
    /// Load parameters from external data
    pub fn loadWeights(self: *LayerNorm, gamma_data: []const f32, beta_data: []const f32) !void {
        if (gamma_data.len != self.gamma.len or beta_data.len != self.beta.len) {
            return error.ParameterSizeMismatch;
        }
        
        @memcpy(self.gamma, gamma_data);
        @memcpy(self.beta, beta_data);
    }
    
    /// Get parameter count
    pub fn getParameterCount(self: *LayerNorm) usize {
        return self.gamma.len + self.beta.len;
    }
};

/// RMS normalization layer (used in LLaMA and other models)
pub const RMSNorm = struct {
    d_model: usize,
    gamma: []f32,  // Scale parameters
    eps: f32,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, d_model: usize) !RMSNorm {
        var gamma = try allocator.alloc(f32, d_model);
        
        // Initialize gamma to 1
        for (gamma) |*g| g.* = 1.0;
        
        return RMSNorm{
            .d_model = d_model,
            .gamma = gamma,
            .eps = 1e-6,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *RMSNorm) void {
        self.allocator.free(self.gamma);
    }
    
    /// Forward pass
    pub fn forward(self: *RMSNorm, input: Matrix, output: *Matrix) !void {
        const batch_size = input.rows;
        std.debug.assert(input.cols == self.d_model);
        std.debug.assert(output.rows == batch_size and output.cols == self.d_model);
        
        // Apply RMS normalization to each row
        for (0..batch_size) |i| {
            const input_row = input.getRow(i);
            const output_row = output.getRow(i);
            
            normalization_math.rmsNorm(input_row, output_row, self.gamma, self.eps);
        }
    }
    
    /// Load parameters from external data
    pub fn loadWeights(self: *RMSNorm, gamma_data: []const f32) !void {
        if (gamma_data.len != self.gamma.len) {
            return error.ParameterSizeMismatch;
        }
        
        @memcpy(self.gamma, gamma_data);
    }
    
    /// Get parameter count
    pub fn getParameterCount(self: *RMSNorm) usize {
        return self.gamma.len;
    }
};

/// Group normalization layer
pub const GroupNorm = struct {
    num_groups: usize,
    num_channels: usize,
    gamma: []f32,
    beta: []f32,
    eps: f32,
    allocator: std.mem.Allocator,
    
    pub fn init(
        allocator: std.mem.Allocator,
        num_groups: usize,
        num_channels: usize,
    ) !GroupNorm {
        std.debug.assert(num_channels % num_groups == 0);
        
        var gamma = try allocator.alloc(f32, num_channels);
        var beta = try allocator.alloc(f32, num_channels);
        
        // Initialize gamma to 1, beta to 0
        for (gamma) |*g| g.* = 1.0;
        for (beta) |*b| b.* = 0.0;
        
        return GroupNorm{
            .num_groups = num_groups,
            .num_channels = num_channels,
            .gamma = gamma,
            .beta = beta,
            .eps = 1e-5,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *GroupNorm) void {
        self.allocator.free(self.gamma);
        self.allocator.free(self.beta);
    }
    
    /// Forward pass
    pub fn forward(self: *GroupNorm, input: Matrix, output: *Matrix) !void {
        const batch_size = input.rows;
        std.debug.assert(input.cols == self.num_channels);
        std.debug.assert(output.rows == batch_size and output.cols == self.num_channels);
        
        const channels_per_group = self.num_channels / self.num_groups;
        
        // Apply group normalization to each row
        for (0..batch_size) |i| {
            const input_row = input.getRow(i);
            const output_row = output.getRow(i);
            
            normalization_math.groupNorm(
                input_row,
                output_row,
                self.gamma,
                self.beta,
                self.num_groups,
                channels_per_group,
                self.eps,
            );
        }
    }
    
    /// Get parameter count
    pub fn getParameterCount(self: *GroupNorm) usize {
        return self.gamma.len + self.beta.len;
    }
};

/// Batch normalization layer (for completeness, though rarely used in transformers)
pub const BatchNorm = struct {
    num_features: usize,
    gamma: []f32,
    beta: []f32,
    running_mean: []f32,
    running_var: []f32,
    eps: f32,
    momentum: f32,
    training: bool,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, num_features: usize) !BatchNorm {
        var gamma = try allocator.alloc(f32, num_features);
        var beta = try allocator.alloc(f32, num_features);
        var running_mean = try allocator.alloc(f32, num_features);
        var running_var = try allocator.alloc(f32, num_features);
        
        // Initialize parameters
        for (gamma) |*g| g.* = 1.0;
        for (beta) |*b| b.* = 0.0;
        for (running_mean) |*m| m.* = 0.0;
        for (running_var) |*v| v.* = 1.0;
        
        return BatchNorm{
            .num_features = num_features,
            .gamma = gamma,
            .beta = beta,
            .running_mean = running_mean,
            .running_var = running_var,
            .eps = 1e-5,
            .momentum = 0.1,
            .training = false,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *BatchNorm) void {
        self.allocator.free(self.gamma);
        self.allocator.free(self.beta);
        self.allocator.free(self.running_mean);
        self.allocator.free(self.running_var);
    }
    
    /// Forward pass (inference mode only)
    pub fn forward(self: *BatchNorm, input: Matrix, output: *Matrix) !void {
        const batch_size = input.rows;
        std.debug.assert(input.cols == self.num_features);
        std.debug.assert(output.rows == batch_size and output.cols == self.num_features);
        
        // Apply batch normalization to each row using running statistics
        for (0..batch_size) |i| {
            const input_row = input.getRow(i);
            const output_row = output.getRow(i);
            
            normalization_math.batchNorm(
                input_row,
                output_row,
                self.gamma,
                self.beta,
                self.running_mean,
                self.running_var,
                self.eps,
            );
        }
    }
    
    /// Get parameter count
    pub fn getParameterCount(self: *BatchNorm) usize {
        return self.gamma.len + self.beta.len + self.running_mean.len + self.running_var.len;
    }
};

test "layer norm creation and forward" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var layer_norm = try LayerNorm.init(allocator, 4);
    defer layer_norm.deinit();
    
    var input = try Matrix.init(allocator, 2, 4);
    defer input.deinit();
    var output = try Matrix.init(allocator, 2, 4);
    defer output.deinit();
    
    // Set some test values
    input.set(0, 0, 1.0); input.set(0, 1, 2.0); input.set(0, 2, 3.0); input.set(0, 3, 4.0);
    input.set(1, 0, 5.0); input.set(1, 1, 6.0); input.set(1, 2, 7.0); input.set(1, 3, 8.0);
    
    try layer_norm.forward(input, &output);
    
    // Check that each row is normalized (approximately zero mean, unit variance)
    for (0..2) |i| {
        const row = output.getRow(i);
        var sum: f32 = 0.0;
        for (row) |val| sum += val;
        const mean = sum / @as(f32, @floatFromInt(row.len));
        
        try testing.expectApproxEqRel(mean, 0.0, 1e-6);
    }
}

test "rms norm creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var rms_norm = try RMSNorm.init(allocator, 8);
    defer rms_norm.deinit();
    
    try testing.expect(rms_norm.d_model == 8);
    try testing.expect(rms_norm.gamma.len == 8);
}

test "group norm creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var group_norm = try GroupNorm.init(allocator, 4, 16);
    defer group_norm.deinit();
    
    try testing.expect(group_norm.num_groups == 4);
    try testing.expect(group_norm.num_channels == 16);
}

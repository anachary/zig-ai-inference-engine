const std = @import("std");
const mod = @import("mod.zig");
const normalization_math = @import("../math/normalization.zig");

const Operator = mod.Operator;
const Tensor = mod.Tensor;
const OpContext = mod.OpContext;

/// Normalization operator
pub const NormalizationOp = struct {
    norm_type: NormType,
    num_features: usize,
    eps: f32,
    
    // Parameters
    gamma: ?Tensor,
    beta: ?Tensor,
    
    pub const NormType = enum {
        layer_norm,
        rms_norm,
        batch_norm,
    };
    
    pub fn init(
        allocator: std.mem.Allocator,
        norm_type: NormType,
        num_features: usize,
        eps: f32,
    ) !NormalizationOp {
        var gamma: ?Tensor = null;
        var beta: ?Tensor = null;
        
        if (norm_type == .layer_norm or norm_type == .batch_norm) {
            const param_shape = [_]usize{num_features};
            gamma = try Tensor.init(allocator, &param_shape);
            beta = try Tensor.init(allocator, &param_shape);
            
            // Initialize gamma to 1, beta to 0
            for (gamma.?.data) |*g| g.* = 1.0;
            for (beta.?.data) |*b| b.* = 0.0;
        } else if (norm_type == .rms_norm) {
            const param_shape = [_]usize{num_features};
            gamma = try Tensor.init(allocator, &param_shape);
            
            // Initialize gamma to 1
            for (gamma.?.data) |*g| g.* = 1.0;
        }
        
        return NormalizationOp{
            .norm_type = norm_type,
            .num_features = num_features,
            .eps = eps,
            .gamma = gamma,
            .beta = beta,
        };
    }
    
    pub fn deinit(self: *NormalizationOp) void {
        if (self.gamma) |*gamma| {
            gamma.deinit();
        }
        if (self.beta) |*beta| {
            beta.deinit();
        }
    }
    
    pub fn forward(
        self: *NormalizationOp,
        inputs: []const Tensor,
        outputs: []Tensor,
        context: *OpContext,
    ) !void {
        _ = context;
        std.debug.assert(inputs.len == 1);
        std.debug.assert(outputs.len == 1);
        
        const input = inputs[0];
        var output = &outputs[0];
        
        switch (self.norm_type) {
            .layer_norm => try self.applyLayerNorm(input, output),
            .rms_norm => try self.applyRMSNorm(input, output),
            .batch_norm => return error.BatchNormNotImplemented,
        }
    }
    
    fn applyLayerNorm(self: *NormalizationOp, input: Tensor, output: *Tensor) !void {
        if (input.shape.len == 2) {
            const batch_size = input.shape[0];
            const feature_size = input.shape[1];
            
            std.debug.assert(feature_size == self.num_features);
            
            for (0..batch_size) |b| {
                // Extract row data
                var input_row = std.ArrayList(f32).init(std.heap.page_allocator);
                defer input_row.deinit();
                var output_row = std.ArrayList(f32).init(std.heap.page_allocator);
                defer output_row.deinit();
                
                try input_row.resize(feature_size);
                try output_row.resize(feature_size);
                
                for (0..feature_size) |f| {
                    input_row.items[f] = input.get(&[_]usize{ b, f });
                }
                
                // Apply layer normalization
                normalization_math.layerNorm(
                    input_row.items,
                    output_row.items,
                    self.gamma.?.data,
                    self.beta.?.data,
                    self.eps,
                );
                
                // Copy back to output tensor
                for (0..feature_size) |f| {
                    output.set(&[_]usize{ b, f }, output_row.items[f]);
                }
            }
        } else if (input.shape.len == 1) {
            // Single vector normalization
            normalization_math.layerNorm(
                input.data,
                output.data,
                self.gamma.?.data,
                self.beta.?.data,
                self.eps,
            );
        } else {
            return error.UnsupportedLayerNormShape;
        }
    }
    
    fn applyRMSNorm(self: *NormalizationOp, input: Tensor, output: *Tensor) !void {
        if (input.shape.len == 2) {
            const batch_size = input.shape[0];
            const feature_size = input.shape[1];
            
            std.debug.assert(feature_size == self.num_features);
            
            for (0..batch_size) |b| {
                // Extract row data
                var input_row = std.ArrayList(f32).init(std.heap.page_allocator);
                defer input_row.deinit();
                var output_row = std.ArrayList(f32).init(std.heap.page_allocator);
                defer output_row.deinit();
                
                try input_row.resize(feature_size);
                try output_row.resize(feature_size);
                
                for (0..feature_size) |f| {
                    input_row.items[f] = input.get(&[_]usize{ b, f });
                }
                
                // Apply RMS normalization
                normalization_math.rmsNorm(
                    input_row.items,
                    output_row.items,
                    self.gamma.?.data,
                    self.eps,
                );
                
                // Copy back to output tensor
                for (0..feature_size) |f| {
                    output.set(&[_]usize{ b, f }, output_row.items[f]);
                }
            }
        } else if (input.shape.len == 1) {
            // Single vector normalization
            normalization_math.rmsNorm(
                input.data,
                output.data,
                self.gamma.?.data,
                self.eps,
            );
        } else {
            return error.UnsupportedRMSNormShape;
        }
    }
    
    pub fn getOutputShape(self: *NormalizationOp, input_shapes: []const []const usize) []const usize {
        _ = self;
        std.debug.assert(input_shapes.len == 1);
        // Normalization preserves input shape
        return input_shapes[0];
    }
};

// VTable implementations
fn normalizationDeinit(impl: *anyopaque, allocator: std.mem.Allocator) void {
    _ = allocator;
    const norm_op: *NormalizationOp = @ptrCast(@alignCast(impl));
    norm_op.deinit();
}

fn normalizationForward(
    impl: *anyopaque,
    inputs: []const Tensor,
    outputs: []Tensor,
    context: *OpContext,
) anyerror!void {
    const norm_op: *NormalizationOp = @ptrCast(@alignCast(impl));
    return norm_op.forward(inputs, outputs, context);
}

fn normalizationGetOutputShape(impl: *anyopaque, input_shapes: []const []const usize) []const usize {
    const norm_op: *NormalizationOp = @ptrCast(@alignCast(impl));
    return norm_op.getOutputShape(input_shapes);
}

/// Create LayerNorm operator
pub fn createLayerNormOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    const num_features = @as(usize, @intCast(params.object.get("num_features").?.integer));
    const eps = @as(f32, @floatCast(params.object.get("eps").?.float));
    
    var norm_op = try allocator.create(NormalizationOp);
    norm_op.* = try NormalizationOp.init(allocator, .layer_norm, num_features, eps);
    
    const vtable = &Operator.VTable{
        .deinit = normalizationDeinit,
        .forward = normalizationForward,
        .getOutputShape = normalizationGetOutputShape,
    };
    
    return Operator.init("LayerNorm", .normalization, vtable, norm_op);
}

/// Create RMSNorm operator
pub fn createRMSNormOp(allocator: std.mem.Allocator, params: std.json.Value) !Operator {
    const num_features = @as(usize, @intCast(params.object.get("num_features").?.integer));
    const eps = @as(f32, @floatCast(params.object.get("eps").?.float));
    
    var norm_op = try allocator.create(NormalizationOp);
    norm_op.* = try NormalizationOp.init(allocator, .rms_norm, num_features, eps);
    
    const vtable = &Operator.VTable{
        .deinit = normalizationDeinit,
        .forward = normalizationForward,
        .getOutputShape = normalizationGetOutputShape,
    };
    
    return Operator.init("RMSNorm", .normalization, vtable, norm_op);
}

const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../core/tensor.zig").Tensor;
const DataType = @import("../core/tensor.zig").DataType;
const TensorError = @import("../core/tensor.zig").TensorError;

/// ONNX operator implementation for real neural network inference
pub const ONNXOperators = struct {
    allocator: Allocator,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
        };
    }
    
    /// Matrix multiplication (MatMul operator)
    /// Implements ONNX MatMul: https://onnx.ai/onnx/operators/onnx__MatMul.html
    pub fn matmul(self: *Self, a: *const Tensor, b: *const Tensor) !Tensor {
        // Validate input tensors
        if (a.data_type != b.data_type) return TensorError.UnsupportedDataType;
        if (a.data_type != .f32) return TensorError.UnsupportedDataType; // Only f32 for now
        
        // Get shapes
        const a_shape = a.shape;
        const b_shape = b.shape;
        
        // Validate matrix multiplication compatibility
        if (a_shape.len < 2 or b_shape.len < 2) return TensorError.InvalidShape;
        if (a_shape[a_shape.len - 1] != b_shape[b_shape.len - 2]) return TensorError.ShapeMismatch;
        
        // Calculate output shape
        var output_shape = std.ArrayList(usize).init(self.allocator);
        defer output_shape.deinit();
        
        // Broadcast batch dimensions
        const max_batch_dims = @max(a_shape.len - 2, b_shape.len - 2);
        for (0..max_batch_dims) |i| {
            const a_dim = if (i < a_shape.len - 2) a_shape[i] else 1;
            const b_dim = if (i < b_shape.len - 2) b_shape[i] else 1;
            try output_shape.append(@max(a_dim, b_dim));
        }
        
        // Add matrix dimensions
        try output_shape.append(a_shape[a_shape.len - 2]); // rows from A
        try output_shape.append(b_shape[b_shape.len - 1]);  // cols from B
        
        // Create output tensor
        var result = try Tensor.init(self.allocator, output_shape.items, a.data_type);
        
        // Perform matrix multiplication
        try self.matmulImpl(a, b, &result);
        
        return result;
    }
    
    /// Element-wise addition (Add operator)
    /// Implements ONNX Add: https://onnx.ai/onnx/operators/onnx__Add.html
    pub fn add(self: *Self, a: *const Tensor, b: *const Tensor) !Tensor {
        if (a.data_type != b.data_type) return TensorError.UnsupportedDataType;
        if (a.data_type != .f32) return TensorError.UnsupportedDataType;
        
        // For now, require same shape (broadcasting TODO)
        if (!std.mem.eql(usize, a.shape, b.shape)) return TensorError.ShapeMismatch;
        
        var result = try Tensor.init(self.allocator, a.shape, a.data_type);
        
        const a_data = a.getData(f32);
        const b_data = b.getData(f32);
        const result_data = result.getMutableData(f32);
        
        for (0..a.element_count) |i| {
            result_data[i] = a_data[i] + b_data[i];
        }
        
        return result;
    }
    
    /// Softmax activation (Softmax operator)
    /// Implements ONNX Softmax: https://onnx.ai/onnx/operators/onnx__Softmax.html
    pub fn softmax(self: *Self, input: *const Tensor, axis: i32) !Tensor {
        if (input.data_type != .f32) return TensorError.UnsupportedDataType;
        
        var result = try Tensor.init(self.allocator, input.shape, input.data_type);
        
        const input_data = input.getData(f32);
        const result_data = result.getMutableData(f32);
        
        // Normalize axis
        const normalized_axis = if (axis < 0) 
            @as(usize, @intCast(@as(i32, @intCast(input.shape.len)) + axis))
        else 
            @as(usize, @intCast(axis));
        
        if (normalized_axis >= input.shape.len) return TensorError.InvalidShape;
        
        // Calculate softmax along specified axis
        try self.softmaxImpl(input_data, result_data, input.shape, normalized_axis);
        
        return result;
    }
    
    /// Layer normalization (LayerNormalization operator)
    /// Implements ONNX LayerNormalization: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
    pub fn layerNorm(self: *Self, input: *const Tensor, scale: *const Tensor, bias: *const Tensor, epsilon: f32) !Tensor {
        if (input.data_type != .f32 or scale.data_type != .f32 or bias.data_type != .f32) {
            return TensorError.UnsupportedDataType;
        }
        
        var result = try Tensor.init(self.allocator, input.shape, input.data_type);
        
        const input_data = input.getData(f32);
        const scale_data = scale.getData(f32);
        const bias_data = bias.getData(f32);
        const result_data = result.getMutableData(f32);
        
        try self.layerNormImpl(input_data, scale_data, bias_data, result_data, input.shape, epsilon);
        
        return result;
    }
    
    /// Reshape tensor (Reshape operator)
    /// Implements ONNX Reshape: https://onnx.ai/onnx/operators/onnx__Reshape.html
    pub fn reshape(self: *Self, input: *const Tensor, new_shape: []const i64) !Tensor {
        // Convert i64 shape to usize and handle -1 (infer dimension)
        var output_shape = std.ArrayList(usize).init(self.allocator);
        defer output_shape.deinit();
        
        var infer_dim_index: ?usize = null;
        var known_size: usize = 1;
        
        for (new_shape, 0..) |dim, i| {
            if (dim == -1) {
                if (infer_dim_index != null) return TensorError.InvalidShape; // Only one -1 allowed
                infer_dim_index = i;
                try output_shape.append(0); // Placeholder
            } else if (dim <= 0) {
                return TensorError.InvalidShape;
            } else {
                const dim_size = @as(usize, @intCast(dim));
                try output_shape.append(dim_size);
                known_size *= dim_size;
            }
        }
        
        // Infer the unknown dimension
        if (infer_dim_index) |idx| {
            if (input.element_count % known_size != 0) return TensorError.InvalidShape;
            output_shape.items[idx] = input.element_count / known_size;
        }
        
        // Validate total element count
        var total_elements: usize = 1;
        for (output_shape.items) |dim| {
            total_elements *= dim;
        }
        if (total_elements != input.element_count) return TensorError.InvalidShape;
        
        // Create new tensor with same data but different shape
        var result = try Tensor.init(self.allocator, output_shape.items, input.data_type);
        
        // Copy data (reshape is just a view change)
        const input_bytes = input.getDataBytes();
        const result_bytes = result.getMutableDataBytes();
        @memcpy(result_bytes, input_bytes);
        
        return result;
    }
    
    /// Transpose tensor (Transpose operator)
    /// Implements ONNX Transpose: https://onnx.ai/onnx/operators/onnx__Transpose.html
    pub fn transpose(self: *Self, input: *const Tensor, perm: ?[]const usize) !Tensor {
        const input_shape = input.shape;
        const ndim = input_shape.len;
        
        // Default permutation is reverse order
        var default_perm = std.ArrayList(usize).init(self.allocator);
        defer default_perm.deinit();
        
        const permutation = if (perm) |p| p else blk: {
            for (0..ndim) |i| {
                try default_perm.append(ndim - 1 - i);
            }
            break :blk default_perm.items;
        };
        
        if (permutation.len != ndim) return TensorError.InvalidShape;
        
        // Calculate output shape
        var output_shape = std.ArrayList(usize).init(self.allocator);
        defer output_shape.deinit();
        
        for (permutation) |axis| {
            if (axis >= ndim) return TensorError.InvalidShape;
            try output_shape.append(input_shape[axis]);
        }
        
        var result = try Tensor.init(self.allocator, output_shape.items, input.data_type);
        
        // Perform transpose
        try self.transposeImpl(input, &result, permutation);
        
        return result;
    }
    
    // Implementation helpers
    
    fn matmulImpl(self: *Self, a: *const Tensor, b: *const Tensor, result: *Tensor) !void {
        _ = self;
        
        const a_data = a.getData(f32);
        const b_data = b.getData(f32);
        const result_data = result.getMutableData(f32);
        
        const a_shape = a.shape;
        const b_shape = b.shape;
        
        // Simple 2D matrix multiplication for now
        if (a_shape.len == 2 and b_shape.len == 2) {
            const M = a_shape[0];
            const K = a_shape[1];
            const N = b_shape[1];
            
            for (0..M) |i| {
                for (0..N) |j| {
                    var sum: f32 = 0.0;
                    for (0..K) |k| {
                        sum += a_data[i * K + k] * b_data[k * N + j];
                    }
                    result_data[i * N + j] = sum;
                }
            }
        } else {
            // TODO: Implement batched matrix multiplication
            return TensorError.UnsupportedDataType;
        }
    }
    
    fn softmaxImpl(self: *Self, input: []const f32, output: []f32, shape: []const usize, axis: usize) !void {
        _ = self;
        
        // Simple implementation for last axis
        if (axis == shape.len - 1) {
            const axis_size = shape[axis];
            const outer_size = input.len / axis_size;
            
            for (0..outer_size) |i| {
                const start = i * axis_size;
                const end = start + axis_size;
                
                // Find max for numerical stability
                var max_val = input[start];
                for (start + 1..end) |j| {
                    max_val = @max(max_val, input[j]);
                }
                
                // Compute exp and sum
                var sum: f32 = 0.0;
                for (start..end) |j| {
                    const exp_val = @exp(input[j] - max_val);
                    output[j] = exp_val;
                    sum += exp_val;
                }
                
                // Normalize
                for (start..end) |j| {
                    output[j] /= sum;
                }
            }
        } else {
            // TODO: Implement softmax for other axes
            return TensorError.UnsupportedDataType;
        }
    }
    
    fn layerNormImpl(self: *Self, input: []const f32, scale: []const f32, bias: []const f32, output: []f32, shape: []const usize, epsilon: f32) !void {
        _ = self;
        
        // Simple implementation for last dimension normalization
        const last_dim = shape[shape.len - 1];
        const outer_size = input.len / last_dim;
        
        for (0..outer_size) |i| {
            const start = i * last_dim;
            const end = start + last_dim;
            
            // Calculate mean
            var mean: f32 = 0.0;
            for (start..end) |j| {
                mean += input[j];
            }
            mean /= @as(f32, @floatFromInt(last_dim));
            
            // Calculate variance
            var variance: f32 = 0.0;
            for (start..end) |j| {
                const diff = input[j] - mean;
                variance += diff * diff;
            }
            variance /= @as(f32, @floatFromInt(last_dim));
            
            // Normalize
            const std_dev = @sqrt(variance + epsilon);
            for (start..end) |j| {
                const idx = j - start;
                output[j] = (input[j] - mean) / std_dev * scale[idx] + bias[idx];
            }
        }
    }
    
    fn transposeImpl(self: *Self, input: *const Tensor, output: *Tensor, perm: []const usize) !void {
        _ = self;
        
        if (input.data_type != .f32) return TensorError.UnsupportedDataType;
        
        const input_data = input.getData(f32);
        const output_data = output.getMutableData(f32);
        
        // TODO: Implement efficient transpose
        // For now, simple element-by-element copy with index transformation
        _ = input_data;
        _ = output_data;
        _ = perm;
        
        return TensorError.UnsupportedDataType; // Placeholder
    }
};

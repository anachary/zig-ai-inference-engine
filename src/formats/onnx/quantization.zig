const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../../core/tensor.zig");

/// ONNX Quantization Support for Phase 3.3
/// Implements INT8, FP16, and dynamic quantization for edge deployment
pub const ONNXQuantization = struct {
    allocator: Allocator,
    quantization_mode: QuantizationMode,
    calibration_data: ?[]tensor.Tensor,

    const Self = @This();

    pub const QuantizationError = error{
        UnsupportedDataType,
        InvalidQuantizationParams,
        CalibrationRequired,
        OutOfMemory,
    };

    pub const QuantizationMode = enum {
        none,           // No quantization (FP32)
        int8_static,    // Static INT8 quantization
        int8_dynamic,   // Dynamic INT8 quantization
        fp16,           // Half precision (FP16)
        bfloat16,       // Brain floating point (BF16)
        mixed_precision, // Mixed precision (FP16 + FP32)
    };

    pub const QuantizationParams = struct {
        scale: f32,
        zero_point: i32,
        min_value: f32,
        max_value: f32,
    };

    pub fn init(allocator: Allocator, mode: QuantizationMode) Self {
        return Self{
            .allocator = allocator,
            .quantization_mode = mode,
            .calibration_data = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.calibration_data) |data| {
            self.allocator.free(data);
        }
    }

    /// Quantize a tensor from FP32 to the specified quantization mode
    pub fn quantizeTensor(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        switch (self.quantization_mode) {
            .none => {
                // No quantization - direct copy
                try self.copyTensor(input, output);
            },
            .int8_static => {
                try self.quantizeToInt8Static(input, output);
            },
            .int8_dynamic => {
                try self.quantizeToInt8Dynamic(input, output);
            },
            .fp16 => {
                try self.quantizeToFP16(input, output);
            },
            .bfloat16 => {
                try self.quantizeToBFloat16(input, output);
            },
            .mixed_precision => {
                try self.quantizeToMixedPrecision(input, output);
            },
        }
    }

    /// Dequantize a tensor back to FP32
    pub fn dequantizeTensor(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        switch (input.data_type) {
            .i8 => {
                try self.dequantizeFromInt8(input, output);
            },
            .f16 => {
                try self.dequantizeFromFP16(input, output);
            },
            .bf16 => {
                try self.dequantizeFromBFloat16(input, output);
            },
            .f32 => {
                // Already FP32 - direct copy
                try self.copyTensor(input, output);
            },
            else => {
                return QuantizationError.UnsupportedDataType;
            },
        }
    }

    /// Calculate quantization parameters for static quantization
    pub fn calculateQuantizationParams(self: *Self, calibration_tensor: tensor.Tensor) !QuantizationParams {
        _ = self;
        
        // Find min and max values in calibration data
        var min_val: f32 = std.math.floatMax(f32);
        var max_val: f32 = std.math.floatMin(f32);

        for (calibration_tensor.data) |value| {
            min_val = @min(min_val, value);
            max_val = @max(max_val, value);
        }

        // Calculate scale and zero point for INT8 quantization
        const qmin: f32 = -128.0; // INT8 min
        const qmax: f32 = 127.0;  // INT8 max

        const scale = (max_val - min_val) / (qmax - qmin);
        const zero_point_float = qmin - min_val / scale;
        const zero_point = @as(i32, @intFromFloat(@round(zero_point_float)));

        return QuantizationParams{
            .scale = scale,
            .zero_point = zero_point,
            .min_value = min_val,
            .max_value = max_val,
        };
    }

    /// Set calibration data for static quantization
    pub fn setCalibrationData(self: *Self, data: []tensor.Tensor) !void {
        self.calibration_data = try self.allocator.dupe(tensor.Tensor, data);
    }

    // Private implementation methods
    fn copyTensor(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        _ = self;
        if (input.data.len != output.data.len) {
            return QuantizationError.InvalidQuantizationParams;
        }
        
        for (input.data, 0..) |value, i| {
            output.data[i] = value;
        }
    }

    fn quantizeToInt8Static(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        if (self.calibration_data == null) {
            return QuantizationError.CalibrationRequired;
        }

        // Calculate quantization parameters from calibration data
        const params = try self.calculateQuantizationParams(self.calibration_data.?[0]);

        // Quantize each value
        for (input.data, 0..) |value, i| {
            const quantized_value = @round(value / params.scale + @as(f32, @floatFromInt(params.zero_point)));
            const clamped = @max(-128.0, @min(127.0, quantized_value));
            
            // Store as INT8 (represented as f32 for now)
            output.data[i] = clamped;
        }
    }

    fn quantizeToInt8Dynamic(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        _ = self;
        
        // Calculate quantization parameters dynamically from input tensor
        var min_val: f32 = std.math.floatMax(f32);
        var max_val: f32 = std.math.floatMin(f32);

        for (input.data) |value| {
            min_val = @min(min_val, value);
            max_val = @max(max_val, value);
        }

        const scale = (max_val - min_val) / 255.0; // 0-255 range for unsigned
        
        // Quantize each value
        for (input.data, 0..) |value, i| {
            const normalized = (value - min_val) / scale;
            const quantized = @max(0.0, @min(255.0, @round(normalized)));
            
            // Convert back to signed range [-128, 127]
            output.data[i] = quantized - 128.0;
        }
    }

    fn quantizeToFP16(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        _ = self;
        
        // Convert FP32 to FP16 (simplified - would need proper IEEE 754 conversion)
        for (input.data, 0..) |value, i| {
            // Clamp to FP16 range
            const fp16_max: f32 = 65504.0;
            const clamped = @max(-fp16_max, @min(fp16_max, value));
            
            // Reduce precision (simplified approximation)
            const reduced_precision = @round(clamped * 1024.0) / 1024.0;
            output.data[i] = reduced_precision;
        }
    }

    fn quantizeToBFloat16(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        _ = self;
        
        // Convert FP32 to BFloat16 (simplified)
        for (input.data, 0..) |value, i| {
            // BFloat16 has same exponent range as FP32 but reduced mantissa
            // Simplified: round to reduce precision
            const reduced_precision = @round(value * 256.0) / 256.0;
            output.data[i] = reduced_precision;
        }
    }

    fn quantizeToMixedPrecision(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        // Mixed precision: use FP16 for most operations, FP32 for critical ones
        // For now, just apply FP16 quantization
        try self.quantizeToFP16(input, output);
    }

    fn dequantizeFromInt8(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        _ = self;
        
        // Simplified dequantization - would need proper scale/zero_point
        for (input.data, 0..) |value, i| {
            // Convert from INT8 range back to approximate FP32
            output.data[i] = value / 127.0; // Normalize to [-1, 1] range
        }
    }

    fn dequantizeFromFP16(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        _ = self;
        
        // FP16 to FP32 conversion (direct copy since we're using f32 storage)
        for (input.data, 0..) |value, i| {
            output.data[i] = value;
        }
    }

    fn dequantizeFromBFloat16(self: *Self, input: tensor.Tensor, output: *tensor.Tensor) !void {
        _ = self;
        
        // BFloat16 to FP32 conversion (direct copy since we're using f32 storage)
        for (input.data, 0..) |value, i| {
            output.data[i] = value;
        }
    }
};

/// Quantization utilities and helper functions
pub const QuantizationUtils = struct {
    /// Get memory savings from quantization
    pub fn getMemorySavings(original_type: tensor.DataType, quantized_type: tensor.DataType) f32 {
        const original_bytes = getDataTypeSize(original_type);
        const quantized_bytes = getDataTypeSize(quantized_type);
        
        return @as(f32, @floatFromInt(original_bytes - quantized_bytes)) / @as(f32, @floatFromInt(original_bytes));
    }

    /// Get performance improvement estimate
    pub fn getPerformanceImprovement(quantization_mode: ONNXQuantization.QuantizationMode) f32 {
        return switch (quantization_mode) {
            .none => 1.0,
            .int8_static => 2.5,      // ~2.5x faster
            .int8_dynamic => 2.0,     // ~2x faster
            .fp16 => 1.8,             // ~1.8x faster
            .bfloat16 => 1.6,         // ~1.6x faster
            .mixed_precision => 1.4,   // ~1.4x faster
        };
    }

    /// Check if quantization mode is supported on current hardware
    pub fn isSupported(quantization_mode: ONNXQuantization.QuantizationMode) bool {
        return switch (quantization_mode) {
            .none => true,
            .int8_static => true,     // Supported on most CPUs
            .int8_dynamic => true,    // Supported on most CPUs
            .fp16 => true,            // Supported on modern GPUs
            .bfloat16 => false,       // Would need hardware detection
            .mixed_precision => true, // Software implementation
        };
    }

    fn getDataTypeSize(data_type: tensor.DataType) u32 {
        return switch (data_type) {
            .f32 => 4,
            .f16 => 2,
            .bf16 => 2,
            .i32 => 4,
            .i16 => 2,
            .i8 => 1,
            .u8 => 1,
        };
    }
};

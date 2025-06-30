const std = @import("std");
const Allocator = std.mem.Allocator;
const TensorInterface = @import("tensor.zig").TensorInterface;

/// Model format interface for different AI model formats
/// Supports ONNX, TensorFlow, PyTorch, and custom formats
pub const ModelInterface = struct {
    /// Model format types
    pub const Format = enum {
        onnx,
        tensorflow,
        pytorch,
        tflite,
        coreml,
        custom,
    };

    /// Model metadata
    pub const Metadata = struct {
        name: []const u8,
        version: []const u8,
        description: []const u8,
        author: []const u8,
        format: Format,
        ir_version: i64,
        opset_version: i64,
        producer_name: []const u8,
        producer_version: []const u8,
        domain: []const u8,
        model_size_bytes: usize,
        parameter_count: usize,
        input_count: usize,
        output_count: usize,
    };

    /// Model input/output specification
    pub const IOSpec = struct {
        name: []const u8,
        shape: []const i64, // -1 for dynamic dimensions
        dtype: TensorInterface.DataType,
        description: []const u8,
    };

    /// Model errors
    pub const ModelError = error{
        InvalidFormat,
        UnsupportedVersion,
        CorruptedModel,
        MissingGraph,
        InvalidNode,
        UnsupportedOperator,
        ParseError,
        OutOfMemory,
    };

    /// Model operations
    pub const Operations = struct {
        /// Parse model from bytes
        parseBytesFn: *const fn (ctx: *anyopaque, data: []const u8) ModelError!*anyopaque,
        
        /// Parse model from file
        parseFileFn: *const fn (ctx: *anyopaque, path: []const u8) ModelError!*anyopaque,
        
        /// Get model metadata
        getMetadataFn: *const fn (ctx: *anyopaque, model: *const anyopaque) Metadata,
        
        /// Get input specifications
        getInputsFn: *const fn (ctx: *anyopaque, model: *const anyopaque) []const IOSpec,
        
        /// Get output specifications
        getOutputsFn: *const fn (ctx: *anyopaque, model: *const anyopaque) []const IOSpec,
        
        /// Validate model
        validateFn: *const fn (ctx: *anyopaque, model: *const anyopaque) ModelError!void,
        
        /// Optimize model
        optimizeFn: *const fn (ctx: *anyopaque, model: *anyopaque) ModelError!void,
        
        /// Serialize model
        serializeFn: *const fn (ctx: *anyopaque, model: *const anyopaque, allocator: Allocator) ModelError![]u8,
        
        /// Free model
        freeFn: *const fn (ctx: *anyopaque, model: *anyopaque) void,
    };

    impl: Operations,
    ctx: *anyopaque,

    pub fn init(ctx: *anyopaque, impl: Operations) ModelInterface {
        return ModelInterface{
            .impl = impl,
            .ctx = ctx,
        };
    }

    pub fn parseBytes(self: *ModelInterface, data: []const u8) ModelError!*anyopaque {
        return self.impl.parseBytesFn(self.ctx, data);
    }

    pub fn parseFile(self: *ModelInterface, path: []const u8) ModelError!*anyopaque {
        return self.impl.parseFileFn(self.ctx, path);
    }

    pub fn getMetadata(self: *ModelInterface, model: *const anyopaque) Metadata {
        return self.impl.getMetadataFn(self.ctx, model);
    }

    pub fn getInputs(self: *ModelInterface, model: *const anyopaque) []const IOSpec {
        return self.impl.getInputsFn(self.ctx, model);
    }

    pub fn getOutputs(self: *ModelInterface, model: *const anyopaque) []const IOSpec {
        return self.impl.getOutputsFn(self.ctx, model);
    }

    pub fn validate(self: *ModelInterface, model: *const anyopaque) ModelError!void {
        return self.impl.validateFn(self.ctx, model);
    }

    pub fn optimize(self: *ModelInterface, model: *anyopaque) ModelError!void {
        return self.impl.optimizeFn(self.ctx, model);
    }

    pub fn serialize(self: *ModelInterface, model: *const anyopaque, allocator: Allocator) ModelError![]u8 {
        return self.impl.serializeFn(self.ctx, model, allocator);
    }

    pub fn free(self: *ModelInterface, model: *anyopaque) void {
        self.impl.freeFn(self.ctx, model);
    }
};

/// Inference engine interface
pub const InferenceInterface = struct {
    /// Inference configuration
    pub const Config = struct {
        max_batch_size: usize,
        max_sequence_length: usize,
        enable_profiling: bool,
        optimization_level: OptimizationLevel,
        precision: Precision,
        device_type: TensorInterface.Device,
    };

    /// Optimization levels
    pub const OptimizationLevel = enum {
        none,
        basic,
        aggressive,
        max,
    };

    /// Precision modes
    pub const Precision = enum {
        fp32,
        fp16,
        int8,
        mixed,
    };

    /// Inference statistics
    pub const Stats = struct {
        total_inferences: usize,
        average_latency_ms: f32,
        peak_memory_mb: usize,
        throughput_ops_per_sec: f32,
        cache_hit_ratio: f32,
    };

    /// Inference errors
    pub const InferenceError = error{
        ModelNotLoaded,
        InvalidInput,
        ShapeMismatch,
        DeviceError,
        OutOfMemory,
        ExecutionFailed,
        TimeoutError,
    };

    /// Inference operations
    pub const Operations = struct {
        /// Load model
        loadModelFn: *const fn (ctx: *anyopaque, model: *const anyopaque, config: Config) InferenceError!void,
        
        /// Unload model
        unloadModelFn: *const fn (ctx: *anyopaque) void,
        
        /// Run inference
        inferFn: *const fn (ctx: *anyopaque, inputs: []const TensorInterface) InferenceError![]TensorInterface,
        
        /// Run batch inference
        inferBatchFn: *const fn (ctx: *anyopaque, batch_inputs: []const []const TensorInterface) InferenceError![][]TensorInterface,
        
        /// Get inference statistics
        getStatsFn: *const fn (ctx: *anyopaque) Stats,
        
        /// Reset statistics
        resetStatsFn: *const fn (ctx: *anyopaque) void,
        
        /// Warm up model
        warmupFn: *const fn (ctx: *anyopaque) InferenceError!void,
        
        /// Set configuration
        setConfigFn: *const fn (ctx: *anyopaque, config: Config) InferenceError!void,
    };

    impl: Operations,
    ctx: *anyopaque,

    pub fn init(ctx: *anyopaque, impl: Operations) InferenceInterface {
        return InferenceInterface{
            .impl = impl,
            .ctx = ctx,
        };
    }

    pub fn loadModel(self: *InferenceInterface, model: *const anyopaque, config: Config) InferenceError!void {
        return self.impl.loadModelFn(self.ctx, model, config);
    }

    pub fn unloadModel(self: *InferenceInterface) void {
        self.impl.unloadModelFn(self.ctx);
    }

    pub fn infer(self: *InferenceInterface, inputs: []const TensorInterface) InferenceError![]TensorInterface {
        return self.impl.inferFn(self.ctx, inputs);
    }

    pub fn inferBatch(self: *InferenceInterface, batch_inputs: []const []const TensorInterface) InferenceError![][]TensorInterface {
        return self.impl.inferBatchFn(self.ctx, batch_inputs);
    }

    pub fn getStats(self: *InferenceInterface) Stats {
        return self.impl.getStatsFn(self.ctx);
    }

    pub fn resetStats(self: *InferenceInterface) void {
        self.impl.resetStatsFn(self.ctx);
    }

    pub fn warmup(self: *InferenceInterface) InferenceError!void {
        return self.impl.warmupFn(self.ctx);
    }

    pub fn setConfig(self: *InferenceInterface, config: Config) InferenceError!void {
        return self.impl.setConfigFn(self.ctx, config);
    }
};

/// Operator interface for neural network operations
pub const OperatorInterface = struct {
    /// Operator types
    pub const OpType = enum {
        // Arithmetic
        add, sub, mul, div, pow,
        // Matrix operations
        matmul, transpose, reshape,
        // Activations
        relu, sigmoid, tanh, softmax, gelu,
        // Convolution
        conv2d, conv3d, depthwise_conv2d,
        // Pooling
        max_pool, avg_pool, global_avg_pool,
        // Normalization
        batch_norm, layer_norm, instance_norm,
        // Reduction
        reduce_sum, reduce_mean, reduce_max, reduce_min,
        // Shape manipulation
        concat, split, slice, squeeze, unsqueeze,
        // Custom
        custom,
    };

    /// Operator attributes
    pub const Attributes = std.StringHashMap([]const u8);

    /// Operator operations
    pub const Operations = struct {
        /// Execute operator
        executeFn: *const fn (ctx: *anyopaque, op_type: OpType, inputs: []const TensorInterface, outputs: []TensorInterface, attributes: Attributes) InferenceInterface.InferenceError!void,
        
        /// Validate operator
        validateFn: *const fn (ctx: *anyopaque, op_type: OpType, input_shapes: []const []const usize, attributes: Attributes) InferenceInterface.InferenceError![][]usize,
        
        /// Get operator info
        getInfoFn: *const fn (ctx: *anyopaque, op_type: OpType) OpInfo,
    };

    /// Operator information
    pub const OpInfo = struct {
        name: []const u8,
        description: []const u8,
        min_inputs: usize,
        max_inputs: usize,
        min_outputs: usize,
        max_outputs: usize,
        supports_inplace: bool,
        supports_broadcasting: bool,
    };

    impl: Operations,
    ctx: *anyopaque,

    pub fn init(ctx: *anyopaque, impl: Operations) OperatorInterface {
        return OperatorInterface{
            .impl = impl,
            .ctx = ctx,
        };
    }

    pub fn execute(self: *OperatorInterface, op_type: OpType, inputs: []const TensorInterface, outputs: []TensorInterface, attributes: Attributes) InferenceInterface.InferenceError!void {
        return self.impl.executeFn(self.ctx, op_type, inputs, outputs, attributes);
    }

    pub fn validate(self: *OperatorInterface, op_type: OpType, input_shapes: []const []const usize, attributes: Attributes) InferenceInterface.InferenceError![][]usize {
        return self.impl.validateFn(self.ctx, op_type, input_shapes, attributes);
    }

    pub fn getInfo(self: *OperatorInterface, op_type: OpType) OpInfo {
        return self.impl.getInfoFn(self.ctx, op_type);
    }
};

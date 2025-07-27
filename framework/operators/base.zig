const std = @import("std");
const Allocator = std.mem.Allocator;
const interfaces = @import("../core/interfaces.zig");

const Tensor = interfaces.Tensor;
const Attributes = interfaces.Attributes;
const ExecutionContext = interfaces.ExecutionContext;
const FrameworkError = interfaces.FrameworkError;
const ShapeInference = interfaces.ShapeInference;
const OperatorValidation = interfaces.OperatorValidation;

/// Base operator interface that all operators must implement
pub const OperatorInterface = struct {
    /// Operator metadata
    pub const Metadata = struct {
        name: []const u8,
        version: []const u8,
        description: []const u8,
        domain: []const u8 = "ai.onnx",
        min_inputs: u32,
        max_inputs: u32,
        min_outputs: u32,
        max_outputs: u32,
        supports_inplace: bool = false,
        supports_broadcasting: bool = false,
        type_constraints: []const TypeConstraint = &.{},
    };

    /// Type constraint for operator inputs/outputs
    pub const TypeConstraint = struct {
        name: []const u8,
        allowed_types: []const Tensor.DataType,
        description: []const u8,
    };

    /// Function signatures for operator implementation
    pub const ComputeFn = fn (
        inputs: []const Tensor,
        outputs: []Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!void;

    pub const ValidateFn = fn (
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void;

    pub const InferShapesFn = fn (
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: Allocator,
    ) FrameworkError![][]usize;

    pub const OptimizeFn = fn (
        inputs: []const Tensor,
        attributes: *const Attributes,
        context: *ExecutionContext,
    ) FrameworkError!OptimizationHint;

    /// Optimization hints for the execution engine
    pub const OptimizationHint = struct {
        can_fuse: bool = false,
        preferred_memory_layout: MemoryLayout = .row_major,
        vectorization_factor: ?u32 = null,
        parallelization_strategy: ParallelizationStrategy = .none,
        memory_access_pattern: MemoryAccessPattern = .sequential,

        pub const MemoryLayout = enum {
            row_major,
            column_major,
            blocked,
            custom,
        };

        pub const ParallelizationStrategy = enum {
            none,
            data_parallel,
            task_parallel,
            pipeline_parallel,
        };

        pub const MemoryAccessPattern = enum {
            sequential,
            random,
            strided,
            gather_scatter,
        };
    };

    /// Complete operator definition
    pub const Definition = struct {
        metadata: Metadata,
        compute_fn: ComputeFn,
        validate_fn: ValidateFn,
        infer_shapes_fn: InferShapesFn,
        optimize_fn: ?OptimizeFn = null,
    };
};

/// Base class for implementing operators
pub fn BaseOperator(comptime OperatorImpl: type) type {
    return struct {
        const Self = @This();

        pub fn getDefinition() OperatorInterface.Definition {
            return OperatorInterface.Definition{
                .metadata = OperatorImpl.getMetadata(),
                .compute_fn = OperatorImpl.compute,
                .validate_fn = OperatorImpl.validate,
                .infer_shapes_fn = OperatorImpl.inferShapes,
                .optimize_fn = if (@hasDecl(OperatorImpl, "optimize")) OperatorImpl.optimize else null,
            };
        }

        /// Default validation implementation
        pub fn defaultValidate(
            input_shapes: []const []const usize,
            input_types: []const Tensor.DataType,
            attributes: *const Attributes,
        ) FrameworkError!void {
            _ = attributes;
            const metadata = OperatorImpl.getMetadata();

            // Check input count
            if (input_shapes.len < metadata.min_inputs or input_shapes.len > metadata.max_inputs) {
                return FrameworkError.InvalidInput;
            }

            // Check type constraints
            for (metadata.type_constraints) |constraint| {
                // This is a simplified type checking - real implementation would be more sophisticated
                for (input_types) |input_type| {
                    var type_allowed = false;
                    for (constraint.allowed_types) |allowed_type| {
                        if (input_type == allowed_type) {
                            type_allowed = true;
                            break;
                        }
                    }
                    if (!type_allowed) {
                        return FrameworkError.DataTypeMismatch;
                    }
                }
            }
        }

        /// Default shape inference implementation
        pub fn defaultInferShapes(
            input_shapes: []const []const usize,
            attributes: *const Attributes,
            allocator: Allocator,
        ) FrameworkError![][]usize {
            _ = attributes;
            
            // Default: output shape is same as first input shape
            if (input_shapes.len == 0) {
                return FrameworkError.InvalidInput;
            }

            const output_shapes = try allocator.alloc([]usize, 1);
            output_shapes[0] = try allocator.dupe(usize, input_shapes[0]);
            return output_shapes;
        }

        /// Helper function to check tensor compatibility
        pub fn checkTensorCompatibility(
            inputs: []const Tensor,
            expected_types: []const Tensor.DataType,
        ) FrameworkError!void {
            if (inputs.len != expected_types.len) {
                return FrameworkError.InvalidInput;
            }

            for (inputs, expected_types) |input, expected_type| {
                if (input.dtype != expected_type) {
                    return FrameworkError.DataTypeMismatch;
                }
            }
        }

        /// Helper function to check shape compatibility
        pub fn checkShapeCompatibility(
            shape1: []const usize,
            shape2: []const usize,
        ) bool {
            if (shape1.len != shape2.len) return false;
            
            for (shape1, shape2) |dim1, dim2| {
                if (dim1 != dim2) return false;
            }
            return true;
        }

        /// Helper function for broadcasting compatibility
        pub fn checkBroadcastCompatibility(
            shape1: []const usize,
            shape2: []const usize,
        ) bool {
            const max_dims = @max(shape1.len, shape2.len);
            
            var i: usize = 0;
            while (i < max_dims) : (i += 1) {
                const dim1 = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
                const dim2 = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;
                
                if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
                    return false;
                }
            }
            return true;
        }

        /// Helper function to calculate broadcast output shape
        pub fn calculateBroadcastShape(
            shape1: []const usize,
            shape2: []const usize,
            allocator: Allocator,
        ) ![]usize {
            const max_dims = @max(shape1.len, shape2.len);
            const output_shape = try allocator.alloc(usize, max_dims);
            
            var i: usize = 0;
            while (i < max_dims) : (i += 1) {
                const dim1 = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
                const dim2 = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;
                
                output_shape[max_dims - 1 - i] = @max(dim1, dim2);
            }
            
            return output_shape;
        }

        /// Helper function to validate attributes
        pub fn validateAttribute(
            attributes: *const Attributes,
            name: []const u8,
            expected_type: type,
            required: bool,
        ) FrameworkError!void {
            const value = attributes.get(name);
            
            if (value == null and required) {
                return FrameworkError.ValidationFailed;
            }
            
            if (value) |v| {
                const valid = switch (expected_type) {
                    i64 => switch (v) {
                        .int => true,
                        else => false,
                    },
                    f64 => switch (v) {
                        .float => true,
                        else => false,
                    },
                    []const u8 => switch (v) {
                        .string => true,
                        else => false,
                    },
                    else => false,
                };
                
                if (!valid) {
                    return FrameworkError.ValidationFailed;
                }
            }
        }
    };
}

/// Utility functions for operator implementations
pub const OperatorUtils = struct {
    /// Calculate total elements in a shape
    pub fn calculateTotalElements(shape: []const usize) usize {
        var total: usize = 1;
        for (shape) |dim| {
            total *= dim;
        }
        return total;
    }

    /// Get element size for a data type
    pub fn getElementSize(dtype: Tensor.DataType) usize {
        return switch (dtype) {
            .f32, .i32, .u32 => 4,
            .f16, .i16, .u16 => 2,
            .i8, .u8, .bool => 1,
            .f64, .i64, .u64 => 8,
        };
    }

    /// Check if two shapes are equal
    pub fn shapesEqual(shape1: []const usize, shape2: []const usize) bool {
        if (shape1.len != shape2.len) return false;
        
        for (shape1, shape2) |dim1, dim2| {
            if (dim1 != dim2) return false;
        }
        return true;
    }

    /// Copy tensor data
    pub fn copyTensorData(src: *const Tensor, dst: *Tensor) FrameworkError!void {
        if (src.data.len != dst.data.len) {
            return FrameworkError.ShapeMismatch;
        }
        
        if (src.dtype != dst.dtype) {
            return FrameworkError.DataTypeMismatch;
        }
        
        @memcpy(dst.data, src.data);
    }
};

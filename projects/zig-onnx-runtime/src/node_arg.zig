const std = @import("std");
const Allocator = std.mem.Allocator;
const ElementType = @import("ort_value.zig").ElementType;

/// ONNX Runtime NodeArg - Exact replica of onnxruntime.NodeArg
/// Represents a model input or output argument
pub const NodeArg = struct {
    allocator: Allocator,
    name: []const u8,
    type_info: TypeInfo,

    const Self = @This();

    /// Type information for a node argument
    pub const TypeInfo = struct {
        tensor_type: ?TensorTypeInfo = null,
        sequence_type: ?SequenceTypeInfo = null,
        map_type: ?MapTypeInfo = null,
        optional_type: ?OptionalTypeInfo = null,
        sparse_tensor_type: ?SparseTensorTypeInfo = null,

        /// Tensor type information
        pub const TensorTypeInfo = struct {
            element_type: ElementType,
            shape: []const i64, // -1 for dynamic dimensions

            pub fn deinit(self: *TensorTypeInfo, allocator: Allocator) void {
                // Free shape safely - check for valid pointer and length
                if (self.shape.len > 0) {
                    allocator.free(self.shape);
                    self.shape = &[_]i64{}; // Mark as freed
                }
            }

            pub fn clone(self: *const TensorTypeInfo, allocator: Allocator) !TensorTypeInfo {
                return TensorTypeInfo{
                    .element_type = self.element_type,
                    .shape = try allocator.dupe(i64, self.shape),
                };
            }
        };

        /// Sequence type information
        pub const SequenceTypeInfo = struct {
            element_type: ElementType,

            pub fn clone(self: *const SequenceTypeInfo, allocator: Allocator) !SequenceTypeInfo {
                _ = allocator;
                return SequenceTypeInfo{
                    .element_type = self.element_type,
                };
            }
        };

        /// Map type information
        pub const MapTypeInfo = struct {
            key_type: ElementType,
            value_type: ElementType,

            pub fn clone(self: *const MapTypeInfo, allocator: Allocator) !MapTypeInfo {
                _ = allocator;
                return MapTypeInfo{
                    .key_type = self.key_type,
                    .value_type = self.value_type,
                };
            }
        };

        /// Optional type information
        pub const OptionalTypeInfo = struct {
            element_type: ElementType,

            pub fn clone(self: *const OptionalTypeInfo, allocator: Allocator) !OptionalTypeInfo {
                _ = allocator;
                return OptionalTypeInfo{
                    .element_type = self.element_type,
                };
            }
        };

        /// Sparse tensor type information
        pub const SparseTensorTypeInfo = struct {
            element_type: ElementType,
            shape: []const i64,

            pub fn deinit(self: *SparseTensorTypeInfo, allocator: Allocator) void {
                allocator.free(self.shape);
            }

            pub fn clone(self: *const SparseTensorTypeInfo, allocator: Allocator) !SparseTensorTypeInfo {
                return SparseTensorTypeInfo{
                    .element_type = self.element_type,
                    .shape = try allocator.dupe(i64, self.shape),
                };
            }
        };

        /// Get string representation of type
        pub fn toString(self: *const TypeInfo) []const u8 {
            if (self.tensor_type) |tensor_type| {
                return tensor_type.element_type.toString();
            }
            if (self.sequence_type) |sequence_type| {
                return sequence_type.element_type.toString();
            }
            if (self.map_type) |map_type| {
                _ = map_type;
                return "map";
            }
            if (self.optional_type) |optional_type| {
                return optional_type.element_type.toString();
            }
            if (self.sparse_tensor_type) |sparse_tensor_type| {
                return sparse_tensor_type.element_type.toString();
            }
            return "unknown";
        }

        /// Check if type is tensor
        pub fn isTensor(self: *const TypeInfo) bool {
            return self.tensor_type != null;
        }

        /// Check if type is sequence
        pub fn isSequence(self: *const TypeInfo) bool {
            return self.sequence_type != null;
        }

        /// Check if type is map
        pub fn isMap(self: *const TypeInfo) bool {
            return self.map_type != null;
        }

        /// Check if type is optional
        pub fn isOptional(self: *const TypeInfo) bool {
            return self.optional_type != null;
        }

        /// Check if type is sparse tensor
        pub fn isSparseTensor(self: *const TypeInfo) bool {
            return self.sparse_tensor_type != null;
        }

        /// Get element type
        pub fn getElementType(self: *const TypeInfo) ?ElementType {
            if (self.tensor_type) |tensor_type| {
                return tensor_type.element_type;
            }
            if (self.sequence_type) |sequence_type| {
                return sequence_type.element_type;
            }
            if (self.optional_type) |optional_type| {
                return optional_type.element_type;
            }
            if (self.sparse_tensor_type) |sparse_tensor_type| {
                return sparse_tensor_type.element_type;
            }
            return null;
        }

        /// Get shape (for tensor types)
        pub fn getShape(self: *const TypeInfo) ?[]const i64 {
            if (self.tensor_type) |tensor_type| {
                return tensor_type.shape;
            }
            if (self.sparse_tensor_type) |sparse_tensor_type| {
                return sparse_tensor_type.shape;
            }
            return null;
        }

        /// Clone type info
        pub fn clone(self: *const TypeInfo, allocator: Allocator) !TypeInfo {
            var cloned = TypeInfo{};

            if (self.tensor_type) |tensor_type| {
                cloned.tensor_type = try tensor_type.clone(allocator);
            }
            if (self.sequence_type) |sequence_type| {
                cloned.sequence_type = try sequence_type.clone(allocator);
            }
            if (self.map_type) |map_type| {
                cloned.map_type = try map_type.clone(allocator);
            }
            if (self.optional_type) |optional_type| {
                cloned.optional_type = try optional_type.clone(allocator);
            }
            if (self.sparse_tensor_type) |sparse_tensor_type| {
                cloned.sparse_tensor_type = try sparse_tensor_type.clone(allocator);
            }

            return cloned;
        }

        /// Deinitialize type info
        pub fn deinit(self: *TypeInfo, allocator: Allocator) void {
            if (self.tensor_type) |*tensor_type| {
                tensor_type.deinit(allocator);
                self.tensor_type = null; // Mark as freed
            }
            if (self.sparse_tensor_type) |*sparse_tensor_type| {
                sparse_tensor_type.deinit(allocator);
                self.sparse_tensor_type = null; // Mark as freed
            }
        }
    };

    /// Initialize NodeArg
    pub fn init(allocator: Allocator, name: []const u8, type_info: TypeInfo) !Self {
        return Self{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .type_info = type_info,
        };
    }

    /// Create tensor NodeArg
    pub fn createTensor(
        allocator: Allocator,
        name: []const u8,
        element_type: ElementType,
        shape: []const i64,
    ) !Self {
        // Add safety checks to prevent segfaults
        if (shape.len == 0) {
            std.log.warn("Empty shape provided to createTensor, using default", .{});
        }

        // Safely duplicate the shape with error handling
        const duplicated_shape = allocator.dupe(i64, shape) catch |err| {
            std.log.err("Failed to duplicate shape in createTensor: {}", .{err});
            // Return a safe default instead of crashing
            const default_shape = allocator.alloc(i64, 1) catch {
                // If allocation fails completely, use a static default
                const static_default = &[_]i64{1};
                return Self{
                    .allocator = allocator,
                    .name = try allocator.dupe(u8, name),
                    .type_info = TypeInfo{
                        .tensor_type = TypeInfo.TensorTypeInfo{
                            .element_type = element_type,
                            .shape = static_default,
                        },
                    },
                };
            };
            default_shape[0] = 1; // Set the default dimension
            return Self{
                .allocator = allocator,
                .name = try allocator.dupe(u8, name),
                .type_info = TypeInfo{
                    .tensor_type = TypeInfo.TensorTypeInfo{
                        .element_type = element_type,
                        .shape = default_shape,
                    },
                },
            };
        };

        const tensor_type = TypeInfo.TensorTypeInfo{
            .element_type = element_type,
            .shape = duplicated_shape,
        };

        const type_info = TypeInfo{
            .tensor_type = tensor_type,
        };

        return Self.init(allocator, name, type_info);
    }

    /// Create sequence NodeArg
    pub fn createSequence(
        allocator: Allocator,
        name: []const u8,
        element_type: ElementType,
    ) !Self {
        const sequence_type = TypeInfo.SequenceTypeInfo{
            .element_type = element_type,
        };

        const type_info = TypeInfo{
            .sequence_type = sequence_type,
        };

        return Self.init(allocator, name, type_info);
    }

    /// Create map NodeArg
    pub fn createMap(
        allocator: Allocator,
        name: []const u8,
        key_type: ElementType,
        value_type: ElementType,
    ) !Self {
        const map_type = TypeInfo.MapTypeInfo{
            .key_type = key_type,
            .value_type = value_type,
        };

        const type_info = TypeInfo{
            .map_type = map_type,
        };

        return Self.init(allocator, name, type_info);
    }

    /// Get name
    /// Exact replica of onnxruntime.NodeArg.name
    pub fn getName(self: *const Self) []const u8 {
        return self.name;
    }

    /// Get type
    /// Exact replica of onnxruntime.NodeArg.type
    pub fn getType(self: *const Self) []const u8 {
        return self.type_info.toString();
    }

    /// Get shape (for tensor types)
    /// Exact replica of onnxruntime.NodeArg.shape
    pub fn getShape(self: *const Self) ?[]const i64 {
        return self.type_info.getShape();
    }

    /// Check if shape is dynamic
    pub fn hasDynamicShape(self: *const Self) bool {
        if (self.type_info.getShape()) |shape| {
            for (shape) |dim| {
                if (dim == -1) {
                    return true;
                }
            }
        }
        return false;
    }

    /// Get number of dimensions
    pub fn getRank(self: *const Self) ?usize {
        if (self.type_info.getShape()) |shape| {
            return shape.len;
        }
        return null;
    }

    /// Get element type
    pub fn getElementType(self: *const Self) ?ElementType {
        return self.type_info.getElementType();
    }

    /// Check if compatible with another NodeArg
    pub fn isCompatibleWith(self: *const Self, other: *const Self) bool {
        // Check element type compatibility
        const self_element_type = self.type_info.getElementType();
        const other_element_type = other.type_info.getElementType();

        if (self_element_type != other_element_type) {
            return false;
        }

        // Check shape compatibility (allowing dynamic dimensions)
        const self_shape = self.type_info.getShape();
        const other_shape = other.type_info.getShape();

        if (self_shape == null and other_shape == null) {
            return true;
        }

        if (self_shape == null or other_shape == null) {
            return false;
        }

        if (self_shape.?.len != other_shape.?.len) {
            return false;
        }

        for (self_shape.?, other_shape.?) |self_dim, other_dim| {
            if (self_dim != -1 and other_dim != -1 and self_dim != other_dim) {
                return false;
            }
        }

        return true;
    }

    /// Validate NodeArg
    pub fn validate(self: *const Self) !void {
        if (self.name.len == 0) {
            return error.EmptyNodeArgName;
        }

        if (self.type_info.getElementType()) |element_type| {
            if (!element_type.isSupported()) {
                return error.UnsupportedElementType;
            }
        }

        if (self.type_info.getShape()) |shape| {
            for (shape) |dim| {
                if (dim < -1 or dim == 0) {
                    return error.InvalidShapeDimension;
                }
            }
        }
    }

    /// Clone NodeArg
    pub fn clone(self: *const Self, allocator: Allocator) !NodeArg {
        const cloned_type_info = try self.type_info.clone(allocator);
        return NodeArg{
            .allocator = allocator,
            .name = try allocator.dupe(u8, self.name),
            .type_info = cloned_type_info,
        };
    }

    /// Get detailed string representation
    pub fn getDetailedString(self: *const Self, allocator: Allocator) ![]u8 {
        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();

        try result.writer().print("NodeArg(name='{s}', type='{s}'", .{ self.name, self.type_info.toString() });

        if (self.type_info.getShape()) |shape| {
            try result.appendSlice(", shape=[");
            for (shape, 0..) |dim, i| {
                if (i > 0) try result.appendSlice(", ");
                if (dim == -1) {
                    try result.appendSlice("?");
                } else {
                    try result.writer().print("{}", .{dim});
                }
            }
            try result.appendSlice("]");
        }

        try result.appendSlice(")");
        return result.toOwnedSlice();
    }

    /// Deinitialize NodeArg
    pub fn deinit(self: *Self) void {
        // Free name safely
        if (self.name.len > 0) {
            self.allocator.free(self.name);
        }

        // Free type info safely
        self.type_info.deinit(self.allocator);
    }
};

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Core tensor interface for the framework
pub const Tensor = struct {
    data: []u8,
    shape: []const usize,
    dtype: DataType,
    strides: []const usize,
    allocator: Allocator,

    pub const DataType = enum {
        f32,
        f16,
        i32,
        i16,
        i8,
        u8,
        bool,
        f64,
        i64,
        u32,
        u16,
        u64,
    };

    pub fn init(allocator: Allocator, shape: []const usize, dtype: DataType) !Tensor {
        const element_size = getElementSize(dtype);
        const total_elements = calculateTotalElements(shape);
        const data = try allocator.alloc(u8, total_elements * element_size);
        
        const owned_shape = try allocator.dupe(usize, shape);
        const strides = try calculateStrides(allocator, shape);
        
        return Tensor{
            .data = data,
            .shape = owned_shape,
            .dtype = dtype,
            .strides = strides,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
    }

    pub fn getData(self: *const Tensor, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.data);
    }

    pub fn getMutableData(self: *Tensor, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.data);
    }

    pub fn getElementCount(self: *const Tensor) usize {
        return calculateTotalElements(self.shape);
    }

    fn getElementSize(dtype: DataType) usize {
        return switch (dtype) {
            .f32, .i32, .u32 => 4,
            .f16, .i16, .u16 => 2,
            .i8, .u8, .bool => 1,
            .f64, .i64, .u64 => 8,
        };
    }

    fn calculateTotalElements(shape: []const usize) usize {
        var total: usize = 1;
        for (shape) |dim| {
            total *= dim;
        }
        return total;
    }

    fn calculateStrides(allocator: Allocator, shape: []const usize) ![]usize {
        const strides = try allocator.alloc(usize, shape.len);
        if (shape.len == 0) return strides;
        
        strides[shape.len - 1] = 1;
        var i = shape.len - 1;
        while (i > 0) {
            i -= 1;
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
};

/// Attributes for operators
pub const Attributes = struct {
    map: std.StringHashMap(AttributeValue),
    allocator: Allocator,

    pub const AttributeValue = union(enum) {
        int: i64,
        float: f64,
        string: []const u8,
        ints: []const i64,
        floats: []const f64,
        strings: []const []const u8,
        tensor: Tensor,
    };

    pub fn init(allocator: Allocator) Attributes {
        return Attributes{
            .map = std.StringHashMap(AttributeValue).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Attributes) void {
        self.map.deinit();
    }

    pub fn set(self: *Attributes, key: []const u8, value: AttributeValue) !void {
        try self.map.put(key, value);
    }

    pub fn get(self: *const Attributes, key: []const u8) ?AttributeValue {
        return self.map.get(key);
    }

    pub fn getInt(self: *const Attributes, key: []const u8, default: i64) i64 {
        if (self.get(key)) |value| {
            return switch (value) {
                .int => |i| i,
                else => default,
            };
        }
        return default;
    }

    pub fn getFloat(self: *const Attributes, key: []const u8, default: f64) f64 {
        if (self.get(key)) |value| {
            return switch (value) {
                .float => |f| f,
                else => default,
            };
        }
        return default;
    }

    pub fn getString(self: *const Attributes, key: []const u8, default: []const u8) []const u8 {
        if (self.get(key)) |value| {
            return switch (value) {
                .string => |s| s,
                else => default,
            };
        }
        return default;
    }
};

/// Execution context for operators
pub const ExecutionContext = struct {
    allocator: Allocator,
    device: Device,
    optimization_level: OptimizationLevel,
    profiling_enabled: bool,
    memory_pool: ?*MemoryPool,

    pub const Device = enum {
        cpu,
        gpu,
        auto,
    };

    pub const OptimizationLevel = enum {
        none,
        basic,
        aggressive,
    };

    pub const MemoryPool = struct {
        allocator: Allocator,
        
        pub fn alloc(self: *MemoryPool, size: usize) ![]u8 {
            return self.allocator.alloc(u8, size);
        }
        
        pub fn free(self: *MemoryPool, memory: []u8) void {
            self.allocator.free(memory);
        }
    };

    pub fn init(allocator: Allocator) ExecutionContext {
        return ExecutionContext{
            .allocator = allocator,
            .device = .auto,
            .optimization_level = .basic,
            .profiling_enabled = false,
            .memory_pool = null,
        };
    }

    pub fn createTensor(self: *ExecutionContext, shape: []const usize, dtype: Tensor.DataType) !Tensor {
        return Tensor.init(self.allocator, shape, dtype);
    }

    pub fn allocateMemory(self: *ExecutionContext, size: usize) ![]u8 {
        if (self.memory_pool) |pool| {
            return pool.alloc(size);
        }
        return self.allocator.alloc(u8, size);
    }

    pub fn freeMemory(self: *ExecutionContext, memory: []u8) void {
        if (self.memory_pool) |pool| {
            pool.free(memory);
        } else {
            self.allocator.free(memory);
        }
    }
};

/// Error types for the framework
pub const FrameworkError = error{
    InvalidInput,
    InvalidOutput,
    ShapeMismatch,
    DataTypeMismatch,
    UnsupportedOperation,
    MemoryAllocationFailed,
    DeviceNotAvailable,
    OperatorNotFound,
    ValidationFailed,
    ExecutionFailed,
    OutOfMemory,
};

/// Shape inference interface
pub const ShapeInference = struct {
    pub const ShapeInferenceFn = fn (
        input_shapes: []const []const usize,
        attributes: *const Attributes,
        allocator: Allocator,
    ) FrameworkError![][]usize;
};

/// Operator validation interface
pub const OperatorValidation = struct {
    pub const ValidationFn = fn (
        input_shapes: []const []const usize,
        input_types: []const Tensor.DataType,
        attributes: *const Attributes,
    ) FrameworkError!void;
};

/// Performance profiling interface
pub const Profiler = struct {
    start_time: i64,
    events: std.ArrayList(Event),
    allocator: Allocator,

    pub const Event = struct {
        name: []const u8,
        start_time: i64,
        end_time: i64,
        memory_used: usize,
    };

    pub fn init(allocator: Allocator) Profiler {
        return Profiler{
            .start_time = std.time.nanoTimestamp(),
            .events = std.ArrayList(Event).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.events.deinit();
    }

    pub fn startEvent(self: *Profiler, name: []const u8) !void {
        try self.events.append(Event{
            .name = name,
            .start_time = std.time.nanoTimestamp(),
            .end_time = 0,
            .memory_used = 0,
        });
    }

    pub fn endEvent(self: *Profiler, memory_used: usize) void {
        if (self.events.items.len > 0) {
            var event = &self.events.items[self.events.items.len - 1];
            event.end_time = std.time.nanoTimestamp();
            event.memory_used = memory_used;
        }
    }

    pub fn getReport(self: *const Profiler) []const Event {
        return self.events.items;
    }
};

/// Memory management interface
pub const MemoryManager = struct {
    allocator: Allocator,
    total_allocated: usize,
    peak_usage: usize,

    pub fn init(allocator: Allocator) MemoryManager {
        return MemoryManager{
            .allocator = allocator,
            .total_allocated = 0,
            .peak_usage = 0,
        };
    }

    pub fn allocate(self: *MemoryManager, size: usize) ![]u8 {
        const memory = try self.allocator.alloc(u8, size);
        self.total_allocated += size;
        if (self.total_allocated > self.peak_usage) {
            self.peak_usage = self.total_allocated;
        }
        return memory;
    }

    pub fn deallocate(self: *MemoryManager, memory: []u8) void {
        self.total_allocated -= memory.len;
        self.allocator.free(memory);
    }

    pub fn getCurrentUsage(self: *const MemoryManager) usize {
        return self.total_allocated;
    }

    pub fn getPeakUsage(self: *const MemoryManager) usize {
        return self.peak_usage;
    }
};

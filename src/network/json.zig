const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");

pub const JSONError = error{
    InvalidJSON,
    MissingField,
    InvalidType,
    OutOfMemory,
};

// Request/Response structures for API endpoints
pub const InferRequest = struct {
    inputs: []TensorData,
    model_id: ?[]const u8 = null,
    options: ?InferOptions = null,

    pub const TensorData = struct {
        name: []const u8,
        shape: []usize,
        data: []f32,
        dtype: []const u8 = "float32",
    };

    pub const InferOptions = struct {
        batch_size: ?u32 = null,
        timeout_ms: ?u32 = null,
        precision: ?[]const u8 = null,
    };
};

pub const InferResponse = struct {
    outputs: []TensorData,
    model_id: []const u8,
    inference_time_ms: f64,
    status: []const u8 = "success",

    pub const TensorData = struct {
        name: []const u8,
        shape: []usize,
        data: []f32,
        dtype: []const u8 = "float32",
    };
};

pub const BatchRequest = struct {
    requests: []InferRequest,
    batch_options: ?BatchOptions = null,

    pub const BatchOptions = struct {
        max_batch_size: ?u32 = null,
        timeout_ms: ?u32 = null,
        parallel: ?bool = null,
    };
};

pub const BatchResponse = struct {
    responses: []InferResponse,
    batch_time_ms: f64,
    status: []const u8 = "success",
};

pub const ModelInfo = struct {
    id: []const u8,
    name: []const u8,
    version: []const u8,
    input_specs: []TensorSpec,
    output_specs: []TensorSpec,
    loaded: bool,
    memory_usage_mb: f64,

    pub const TensorSpec = struct {
        name: []const u8,
        shape: []i32, // -1 for dynamic dimensions
        dtype: []const u8,
    };
};

pub const ModelsResponse = struct {
    models: []ModelInfo,
    total_count: u32,
    status: []const u8 = "success",
};

pub const LoadModelRequest = struct {
    path: []const u8,
    model_id: ?[]const u8 = null,
    options: ?LoadOptions = null,

    pub const LoadOptions = struct {
        cache: ?bool = null,
        optimize: ?bool = null,
        quantize: ?bool = null,
    };
};

pub const HealthResponse = struct {
    status: []const u8 = "healthy",
    version: []const u8,
    uptime_seconds: u64,
    memory_usage_mb: f64,
    models_loaded: u32,
    requests_processed: u64,
};

pub const StatsResponse = struct {
    engine: EngineStats,
    performance: PerformanceStats,
    memory: MemoryStats,
    status: []const u8 = "success",

    pub const EngineStats = struct {
        models_loaded: u32,
        operators_available: u32,
        tensors_pooled: u32,
        uptime_seconds: u64,
    };

    pub const PerformanceStats = struct {
        requests_processed: u64,
        avg_latency_ms: f64,
        throughput_rps: f64,
        errors_count: u64,
    };

    pub const MemoryStats = struct {
        current_usage_mb: f64,
        peak_usage_mb: f64,
        pool_hit_rate: f64,
        gpu_usage_mb: f64,
    };
};

pub const ErrorResponse = struct {
    @"error": []const u8,
    message: []const u8,
    code: u32,
    details: ?[]const u8 = null,
    status: []const u8 = "error",
};

// JSON parsing and serialization functions
pub const JSONProcessor = struct {
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn parseInferRequest(self: *Self, json_data: []const u8) !InferRequest {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_data, .{});
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return JSONError.InvalidJSON;

        // Parse inputs array
        const inputs_value = root.object.get("inputs") orelse return JSONError.MissingField;
        if (inputs_value != .array) return JSONError.InvalidType;

        var inputs = std.ArrayList(InferRequest.TensorData).init(self.allocator);
        defer inputs.deinit();

        for (inputs_value.array.items) |input_item| {
            if (input_item != .object) return JSONError.InvalidType;

            const name = input_item.object.get("name") orelse return JSONError.MissingField;
            const shape = input_item.object.get("shape") orelse return JSONError.MissingField;
            const data = input_item.object.get("data") orelse return JSONError.MissingField;

            if (name != .string or shape != .array or data != .array) return JSONError.InvalidType;

            // Parse shape
            var shape_list = std.ArrayList(usize).init(self.allocator);
            defer shape_list.deinit();
            for (shape.array.items) |dim| {
                if (dim != .integer) return JSONError.InvalidType;
                try shape_list.append(@intCast(dim.integer));
            }

            // Parse data
            var data_list = std.ArrayList(f32).init(self.allocator);
            defer data_list.deinit();
            for (data.array.items) |value| {
                const float_val = switch (value) {
                    .integer => @as(f32, @floatFromInt(value.integer)),
                    .float => @as(f32, @floatCast(value.float)),
                    else => return JSONError.InvalidType,
                };
                try data_list.append(float_val);
            }

            const dtype = if (input_item.object.get("dtype")) |dt|
                if (dt == .string) dt.string else "float32"
            else
                "float32";

            try inputs.append(.{
                .name = try self.allocator.dupe(u8, name.string),
                .shape = try shape_list.toOwnedSlice(),
                .data = try data_list.toOwnedSlice(),
                .dtype = try self.allocator.dupe(u8, dtype),
            });
        }

        // Parse optional fields
        const model_id = if (root.object.get("model_id")) |mid|
            if (mid == .string) try self.allocator.dupe(u8, mid.string) else null
        else
            null;

        return InferRequest{
            .inputs = try inputs.toOwnedSlice(),
            .model_id = model_id,
            .options = null, // TODO: Parse options
        };
    }

    pub fn serializeInferResponse(self: *Self, response: InferResponse) ![]u8 {
        var string = std.ArrayList(u8).init(self.allocator);
        defer string.deinit();

        try std.json.stringify(response, .{}, string.writer());
        return string.toOwnedSlice();
    }

    pub fn serializeErrorResponse(self: *Self, error_response: ErrorResponse) ![]u8 {
        var string = std.ArrayList(u8).init(self.allocator);
        defer string.deinit();

        try std.json.stringify(error_response, .{}, string.writer());
        return string.toOwnedSlice();
    }

    pub fn serializeHealthResponse(self: *Self, health: HealthResponse) ![]u8 {
        var string = std.ArrayList(u8).init(self.allocator);
        defer string.deinit();

        try std.json.stringify(health, .{}, string.writer());
        return string.toOwnedSlice();
    }

    pub fn serializeStatsResponse(self: *Self, stats: StatsResponse) ![]u8 {
        var string = std.ArrayList(u8).init(self.allocator);
        defer string.deinit();

        try std.json.stringify(stats, .{}, string.writer());
        return string.toOwnedSlice();
    }

    pub fn serializeModelsResponse(self: *Self, models: ModelsResponse) ![]u8 {
        var string = std.ArrayList(u8).init(self.allocator);
        defer string.deinit();

        try std.json.stringify(models, .{}, string.writer());
        return string.toOwnedSlice();
    }
};

// Utility functions for tensor conversion
pub fn tensorToJSON(allocator: Allocator, t: tensor.Tensor) !InferResponse.TensorData {
    const shape = try allocator.dupe(usize, t.shape);

    // Extract data based on tensor type
    var data = std.ArrayList(f32).init(allocator);
    defer data.deinit();

    const total_elements = t.numel();
    var i: usize = 0;
    while (i < total_elements) : (i += 1) {
        const value = switch (t.dtype) {
            .f32 => try t.get_f32_flat(i),
            .f16 => blk: {
                // For now, convert through f32 - TODO: implement proper f16 support
                const f32_val = try t.get_f32_flat(i);
                break :blk f32_val;
            },
            .i32, .i16, .i8, .u8 => blk: {
                // For now, convert through f32 - TODO: implement proper integer support
                const f32_val = try t.get_f32_flat(i);
                break :blk f32_val;
            },
        };
        try data.append(value);
    }

    return InferResponse.TensorData{
        .name = "output", // TODO: Get actual tensor name
        .shape = shape,
        .data = try data.toOwnedSlice(),
        .dtype = @tagName(t.dtype),
    };
}

pub fn jsonToTensor(allocator: Allocator, json_tensor: InferRequest.TensorData) !tensor.Tensor {
    var t = try tensor.Tensor.init(allocator, json_tensor.shape, .f32);

    // Copy data
    const total_elements = t.numel();
    if (json_tensor.data.len != total_elements) {
        return JSONError.InvalidType;
    }

    var i: usize = 0;
    while (i < total_elements) : (i += 1) {
        try t.set_f32_flat(i, json_tensor.data[i]);
    }

    return t;
}

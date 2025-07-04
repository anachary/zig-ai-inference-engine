const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;

// Import the main Zig AI platform
const ai_platform = @import("../lib.zig");

// Global allocator for C API
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Platform instance storage
var platform_instances = std.HashMap(*anyopaque, *ai_platform.Platform, std.hash_map.AutoContext(*anyopaque), std.hash_map.default_max_load_percentage).init(allocator);

/// C API Error codes
const CApiError = enum(c_int) {
    Success = 0,
    InvalidInput = -1,
    InitializationFailed = -2,
    ModelLoadFailed = -3,
    InferenceFailed = -4,
    BufferTooSmall = -5,
    PlatformNotFound = -6,
    OutOfMemory = -7,
    JsonParseError = -8,
};

/// IoT-specific configuration structure
const IoTConfig = struct {
    environment: ai_platform.Environment = .production,
    deployment_target: ai_platform.DeploymentTarget = .iot,
    enable_monitoring: bool = true,
    enable_logging: bool = false,
    enable_metrics: bool = false,
    enable_auto_scaling: bool = false,
    health_check_interval_ms: u64 = 60000,
    log_level: ai_platform.LogLevel = .err,
    max_memory_mb: ?u32 = 512,
    max_cpu_cores: ?u32 = 4,
    enable_gpu: bool = false,
    data_directory: []const u8 = "/tmp/zig-ai-data",
    log_directory: []const u8 = "/tmp/zig-ai-logs",
};

/// Initialize Zig AI Platform for IoT
/// @param config_json JSON configuration string
/// @return Platform handle or null on failure
export fn zig_ai_platform_init(config_json: [*:0]const u8) ?*anyopaque {
    const config_str = std.mem.span(config_json);
    
    // Parse JSON configuration
    var parsed = json.parseFromSlice(json.Value, allocator, config_str, .{}) catch |err| {
        std.log.err("Failed to parse config JSON: {}", .{err});
        return null;
    };
    defer parsed.deinit();
    
    // Convert JSON to IoT config
    const iot_config = parseIoTConfig(parsed.value) catch |err| {
        std.log.err("Failed to parse IoT config: {}", .{err});
        return null;
    };
    
    // Create platform config
    const platform_config = ai_platform.PlatformConfig{
        .environment = iot_config.environment,
        .deployment_target = iot_config.deployment_target,
        .enable_monitoring = iot_config.enable_monitoring,
        .enable_logging = iot_config.enable_logging,
        .enable_metrics = iot_config.enable_metrics,
        .enable_auto_scaling = iot_config.enable_auto_scaling,
        .health_check_interval_ms = iot_config.health_check_interval_ms,
        .log_level = iot_config.log_level,
        .max_memory_mb = iot_config.max_memory_mb,
        .max_cpu_cores = iot_config.max_cpu_cores,
        .enable_gpu = iot_config.enable_gpu,
        .data_directory = iot_config.data_directory,
        .log_directory = iot_config.log_directory,
    };
    
    // Initialize platform
    var platform = allocator.create(ai_platform.Platform) catch |err| {
        std.log.err("Failed to allocate platform: {}", .{err});
        return null;
    };
    
    platform.* = ai_platform.Platform.init(allocator, platform_config) catch |err| {
        std.log.err("Failed to initialize platform: {}", .{err});
        allocator.destroy(platform);
        return null;
    };
    
    // Start platform
    platform.start() catch |err| {
        std.log.err("Failed to start platform: {}", .{err});
        platform.deinit();
        allocator.destroy(platform);
        return null;
    };
    
    // Store platform instance
    const handle = @as(*anyopaque, @ptrCast(platform));
    platform_instances.put(handle, platform) catch |err| {
        std.log.err("Failed to store platform instance: {}", .{err});
        platform.stop();
        platform.deinit();
        allocator.destroy(platform);
        return null;
    };
    
    std.log.info("✅ Zig AI Platform initialized for IoT (handle: {})", .{@intFromPtr(handle)});
    return handle;
}

/// Deinitialize Zig AI Platform
/// @param handle Platform handle
export fn zig_ai_platform_deinit(handle: *anyopaque) void {
    if (platform_instances.get(handle)) |platform| {
        platform.stop();
        platform.deinit();
        allocator.destroy(platform);
        _ = platform_instances.remove(handle);
        std.log.info("🛑 Zig AI Platform deinitialized (handle: {})", .{@intFromPtr(handle)});
    } else {
        std.log.warn("⚠️ Attempted to deinitialize unknown platform handle: {}", .{@intFromPtr(handle)});
    }
}

/// Load a model for inference
/// @param handle Platform handle
/// @param model_path Path to the model file
/// @return Error code
export fn zig_ai_load_model(handle: *anyopaque, model_path: [*:0]const u8) c_int {
    const platform = platform_instances.get(handle) orelse {
        std.log.err("❌ Invalid platform handle for model loading");
        return @intFromEnum(CApiError.PlatformNotFound);
    };
    
    const path_str = std.mem.span(model_path);
    
    // Load model (simplified for IoT)
    platform.loadModel(path_str) catch |err| {
        std.log.err("❌ Failed to load model {s}: {}", .{ path_str, err });
        return @intFromEnum(CApiError.ModelLoadFailed);
    };
    
    std.log.info("✅ Model loaded: {s}", .{path_str});
    return @intFromEnum(CApiError.Success);
}

/// Perform inference
/// @param handle Platform handle
/// @param input_json JSON input data
/// @param output_buffer Buffer for output JSON
/// @param buffer_size Size of output buffer
/// @return Error code
export fn zig_ai_inference(
    handle: *anyopaque,
    input_json: [*:0]const u8,
    output_buffer: [*]u8,
    buffer_size: usize,
) c_int {
    const platform = platform_instances.get(handle) orelse {
        std.log.err("❌ Invalid platform handle for inference");
        return @intFromEnum(CApiError.PlatformNotFound);
    };
    
    const input_str = std.mem.span(input_json);
    
    // Parse input JSON
    var parsed_input = json.parseFromSlice(json.Value, allocator, input_str, .{}) catch |err| {
        std.log.err("❌ Failed to parse input JSON: {}", .{err});
        return @intFromEnum(CApiError.JsonParseError);
    };
    defer parsed_input.deinit();
    
    // Perform inference
    const result = platform.performInference(parsed_input.value) catch |err| {
        std.log.err("❌ Inference failed: {}", .{err});
        return @intFromEnum(CApiError.InferenceFailed);
    };
    defer allocator.free(result);
    
    // Check buffer size
    if (result.len >= buffer_size) {
        std.log.err("❌ Output buffer too small: need {}, have {}", .{ result.len + 1, buffer_size });
        return @intFromEnum(CApiError.BufferTooSmall);
    };
    
    // Copy result to output buffer
    @memcpy(output_buffer[0..result.len], result);
    output_buffer[result.len] = 0; // Null terminate
    
    return @intFromEnum(CApiError.Success);
}

/// Get platform status
/// @param handle Platform handle
/// @param status_buffer Buffer for status JSON
/// @param buffer_size Size of status buffer
/// @return Error code
export fn zig_ai_get_status(
    handle: *anyopaque,
    status_buffer: [*]u8,
    buffer_size: usize,
) c_int {
    const platform = platform_instances.get(handle) orelse {
        std.log.err("❌ Invalid platform handle for status");
        return @intFromEnum(CApiError.PlatformNotFound);
    };
    
    // Get platform status
    const status = platform.getStatus();
    
    // Create status JSON
    var status_json = std.ArrayList(u8).init(allocator);
    defer status_json.deinit();
    
    const writer = status_json.writer();
    json.stringify(status, .{}, writer) catch |err| {
        std.log.err("❌ Failed to serialize status: {}", .{err});
        return @intFromEnum(CApiError.JsonParseError);
    };
    
    // Check buffer size
    if (status_json.items.len >= buffer_size) {
        std.log.err("❌ Status buffer too small: need {}, have {}", .{ status_json.items.len + 1, buffer_size });
        return @intFromEnum(CApiError.BufferTooSmall);
    };
    
    // Copy status to buffer
    @memcpy(status_buffer[0..status_json.items.len], status_json.items);
    status_buffer[status_json.items.len] = 0; // Null terminate
    
    return @intFromEnum(CApiError.Success);
}

/// Parse IoT configuration from JSON
fn parseIoTConfig(json_value: json.Value) !IoTConfig {
    var config = IoTConfig{};
    
    if (json_value != .object) {
        return error.InvalidConfigFormat;
    }
    
    const obj = json_value.object;
    
    // Parse environment
    if (obj.get("environment")) |env_val| {
        if (env_val == .string) {
            if (std.mem.eql(u8, env_val.string, "development")) {
                config.environment = .development;
            } else if (std.mem.eql(u8, env_val.string, "production")) {
                config.environment = .production;
            }
        }
    }
    
    // Parse deployment target
    if (obj.get("deployment_target")) |target_val| {
        if (target_val == .string) {
            if (std.mem.eql(u8, target_val.string, "iot")) {
                config.deployment_target = .iot;
            }
        }
    }
    
    // Parse boolean flags
    if (obj.get("enable_monitoring")) |val| {
        if (val == .bool) config.enable_monitoring = val.bool;
    }
    if (obj.get("enable_logging")) |val| {
        if (val == .bool) config.enable_logging = val.bool;
    }
    if (obj.get("enable_metrics")) |val| {
        if (val == .bool) config.enable_metrics = val.bool;
    }
    if (obj.get("enable_auto_scaling")) |val| {
        if (val == .bool) config.enable_auto_scaling = val.bool;
    }
    if (obj.get("enable_gpu")) |val| {
        if (val == .bool) config.enable_gpu = val.bool;
    }
    
    // Parse numeric values
    if (obj.get("health_check_interval_ms")) |val| {
        if (val == .integer) config.health_check_interval_ms = @as(u64, @intCast(val.integer));
    }
    if (obj.get("max_memory_mb")) |val| {
        if (val == .integer) config.max_memory_mb = @as(u32, @intCast(val.integer));
    }
    if (obj.get("max_cpu_cores")) |val| {
        if (val == .integer) config.max_cpu_cores = @as(u32, @intCast(val.integer));
    }
    
    // Parse log level
    if (obj.get("log_level")) |val| {
        if (val == .string) {
            if (std.mem.eql(u8, val.string, "debug")) {
                config.log_level = .debug;
            } else if (std.mem.eql(u8, val.string, "info")) {
                config.log_level = .info;
            } else if (std.mem.eql(u8, val.string, "warn")) {
                config.log_level = .warn;
            } else if (std.mem.eql(u8, val.string, "error")) {
                config.log_level = .err;
            }
        }
    }
    
    return config;
}

/// Cleanup function called on library unload
export fn zig_ai_cleanup() void {
    // Cleanup all remaining platform instances
    var iterator = platform_instances.iterator();
    while (iterator.next()) |entry| {
        const platform = entry.value_ptr.*;
        platform.stop();
        platform.deinit();
        allocator.destroy(platform);
    }
    platform_instances.deinit();
    
    // Cleanup global allocator
    _ = gpa.deinit();
    
    std.log.info("🧹 Zig AI C API cleanup completed");
}

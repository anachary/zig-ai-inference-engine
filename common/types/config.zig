const std = @import("std");
const errors = @import("errors.zig");

/// Configuration types for the Zig AI ecosystem
/// Provides consistent configuration across all projects

/// Base configuration interface
pub const ConfigInterface = struct {
    /// Validate configuration
    validateFn: *const fn (ctx: *anyopaque) errors.ZigAIError!void,
    
    /// Get configuration as JSON
    toJsonFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) errors.ZigAIError![]u8,
    
    /// Load configuration from JSON
    fromJsonFn: *const fn (ctx: *anyopaque, json: []const u8) errors.ZigAIError!void,
    
    /// Clone configuration
    cloneFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) errors.ZigAIError!*anyopaque,
    
    /// Free configuration
    freeFn: *const fn (ctx: *anyopaque) void,

    ctx: *anyopaque,

    pub fn validate(self: *const ConfigInterface) errors.ZigAIError!void {
        return self.validateFn(self.ctx);
    }

    pub fn toJson(self: *const ConfigInterface, allocator: std.mem.Allocator) errors.ZigAIError![]u8 {
        return self.toJsonFn(self.ctx, allocator);
    }

    pub fn fromJson(self: *ConfigInterface, json: []const u8) errors.ZigAIError!void {
        return self.fromJsonFn(self.ctx, json);
    }

    pub fn clone(self: *const ConfigInterface, allocator: std.mem.Allocator) errors.ZigAIError!ConfigInterface {
        const new_ctx = try self.cloneFn(self.ctx, allocator);
        return ConfigInterface{
            .validateFn = self.validateFn,
            .toJsonFn = self.toJsonFn,
            .fromJsonFn = self.fromJsonFn,
            .cloneFn = self.cloneFn,
            .freeFn = self.freeFn,
            .ctx = new_ctx,
        };
    }

    pub fn free(self: *ConfigInterface) void {
        self.freeFn(self.ctx);
    }
};

/// Memory configuration
pub const MemoryConfig = struct {
    /// Maximum memory usage in MB
    max_memory_mb: u32 = 1024,
    
    /// Memory allocation strategy
    allocation_strategy: AllocationStrategy = .general_purpose,
    
    /// Tensor pool size
    tensor_pool_size: usize = 100,
    
    /// Enable memory tracking
    enable_tracking: bool = true,
    
    /// Memory alignment
    alignment: u8 = 16,
    
    /// Arena size for arena allocator
    arena_size_mb: u32 = 256,

    pub const AllocationStrategy = enum {
        general_purpose,
        arena,
        pool,
        stack,
        ring_buffer,
    };

    pub fn validate(self: *const MemoryConfig) errors.ZigAIError!void {
        if (self.max_memory_mb == 0) {
            return errors.CoreError.InvalidConfiguration;
        }
        if (self.tensor_pool_size == 0) {
            return errors.CoreError.InvalidConfiguration;
        }
        if (self.alignment == 0 or (self.alignment & (self.alignment - 1)) != 0) {
            return errors.CoreError.InvalidConfiguration;
        }
        if (self.arena_size_mb > self.max_memory_mb) {
            return errors.CoreError.InvalidConfiguration;
        }
    }

    pub fn getRecommendedForDevice(device_type: DeviceType, available_memory_mb: u32) MemoryConfig {
        return switch (device_type) {
            .iot => MemoryConfig{
                .max_memory_mb = @min(512, available_memory_mb / 2),
                .allocation_strategy = .arena,
                .tensor_pool_size = 20,
                .enable_tracking = false,
                .alignment = 8,
                .arena_size_mb = @min(128, available_memory_mb / 4),
            },
            .desktop => MemoryConfig{
                .max_memory_mb = @min(2048, available_memory_mb / 2),
                .allocation_strategy = .general_purpose,
                .tensor_pool_size = 100,
                .enable_tracking = true,
                .alignment = 16,
                .arena_size_mb = @min(512, available_memory_mb / 4),
            },
            .server => MemoryConfig{
                .max_memory_mb = @min(8192, available_memory_mb / 2),
                .allocation_strategy = .pool,
                .tensor_pool_size = 500,
                .enable_tracking = true,
                .alignment = 32,
                .arena_size_mb = @min(2048, available_memory_mb / 4),
            },
        };
    }
};

/// Compute configuration
pub const ComputeConfig = struct {
    /// Number of threads (null = auto-detect)
    num_threads: ?u32 = null,
    
    /// Device type preference
    device_type: DeviceType = .auto,
    
    /// Enable SIMD optimizations
    enable_simd: bool = true,
    
    /// Enable GPU acceleration
    enable_gpu: bool = true,
    
    /// GPU backend preference
    gpu_backend: GPUBackend = .auto,
    
    /// Optimization level
    optimization_level: OptimizationLevel = .balanced,

    pub const DeviceType = enum {
        auto,
        cpu,
        gpu,
        npu,
        iot,
        desktop,
        server,
    };

    pub const GPUBackend = enum {
        auto,
        cuda,
        vulkan,
        opencl,
        metal,
        directx,
    };

    pub const OptimizationLevel = enum {
        none,
        basic,
        balanced,
        aggressive,
        max,
    };

    pub fn validate(self: *const ComputeConfig) errors.ZigAIError!void {
        if (self.num_threads) |threads| {
            if (threads == 0) {
                return errors.CoreError.InvalidConfiguration;
            }
        }
    }

    pub fn getRecommendedThreads(self: *const ComputeConfig) u32 {
        if (self.num_threads) |threads| {
            return threads;
        }
        
        const cpu_count = std.Thread.getCpuCount() catch 1;
        return switch (self.device_type) {
            .iot => @min(2, cpu_count),
            .desktop => @min(8, cpu_count),
            .server => cpu_count,
            else => @min(4, cpu_count),
        };
    }
};

/// Model configuration
pub const ModelConfig = struct {
    /// Model format
    format: ModelFormat = .auto,
    
    /// Precision mode
    precision: Precision = .fp32,
    
    /// Enable quantization
    enable_quantization: bool = false,
    
    /// Quantization bits
    quantization_bits: u8 = 8,
    
    /// Enable model optimization
    enable_optimization: bool = true,
    
    /// Batch size
    batch_size: usize = 1,
    
    /// Maximum sequence length
    max_sequence_length: usize = 512,

    pub const ModelFormat = enum {
        auto,
        onnx,
        tensorflow,
        pytorch,
        tflite,
        custom,
    };

    pub const Precision = enum {
        fp32,
        fp16,
        int8,
        mixed,
    };

    pub fn validate(self: *const ModelConfig) errors.ZigAIError!void {
        if (self.batch_size == 0) {
            return errors.CoreError.InvalidConfiguration;
        }
        if (self.max_sequence_length == 0) {
            return errors.CoreError.InvalidConfiguration;
        }
        if (self.enable_quantization and (self.quantization_bits == 0 or self.quantization_bits > 32)) {
            return errors.CoreError.InvalidConfiguration;
        }
    }
};

/// Network configuration
pub const NetworkConfig = struct {
    /// Server host
    host: []const u8 = "127.0.0.1",
    
    /// Server port
    port: u16 = 8080,
    
    /// Maximum concurrent connections
    max_connections: u32 = 100,
    
    /// Request timeout in seconds
    timeout_seconds: u32 = 30,
    
    /// Enable CORS
    enable_cors: bool = true,
    
    /// Enable compression
    enable_compression: bool = true,
    
    /// Maximum request size in MB
    max_request_size_mb: u32 = 10,

    pub fn validate(self: *const NetworkConfig) errors.ZigAIError!void {
        if (self.port == 0) {
            return errors.CoreError.InvalidConfiguration;
        }
        if (self.max_connections == 0) {
            return errors.CoreError.InvalidConfiguration;
        }
        if (self.timeout_seconds == 0) {
            return errors.CoreError.InvalidConfiguration;
        }
        if (self.max_request_size_mb == 0) {
            return errors.CoreError.InvalidConfiguration;
        }
    }
};

/// Logging configuration
pub const LoggingConfig = struct {
    /// Log level
    level: LogLevel = .info,
    
    /// Enable file logging
    enable_file_logging: bool = false,
    
    /// Log file path
    log_file_path: ?[]const u8 = null,
    
    /// Maximum log file size in MB
    max_log_file_size_mb: u32 = 100,
    
    /// Enable console logging
    enable_console_logging: bool = true,
    
    /// Enable structured logging (JSON)
    enable_structured_logging: bool = false,

    pub const LogLevel = enum {
        debug,
        info,
        warn,
        err,
        fatal,
    };

    pub fn validate(self: *const LoggingConfig) errors.ZigAIError!void {
        if (self.enable_file_logging and self.log_file_path == null) {
            return errors.CoreError.InvalidConfiguration;
        }
        if (self.max_log_file_size_mb == 0) {
            return errors.CoreError.InvalidConfiguration;
        }
    }
};

/// Unified configuration for the entire ecosystem
pub const ZigAIConfig = struct {
    memory: MemoryConfig = .{},
    compute: ComputeConfig = .{},
    model: ModelConfig = .{},
    network: NetworkConfig = .{},
    logging: LoggingConfig = .{},
    
    /// Enable profiling
    enable_profiling: bool = false,
    
    /// Enable debugging
    enable_debugging: bool = false,
    
    /// Configuration version
    version: []const u8 = "1.0.0",

    pub fn validate(self: *const ZigAIConfig) errors.ZigAIError!void {
        try self.memory.validate();
        try self.compute.validate();
        try self.model.validate();
        try self.network.validate();
        try self.logging.validate();
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) errors.ZigAIError!ZigAIConfig {
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => errors.IOError.FileNotFound,
                error.AccessDenied => errors.IOError.FileAccessDenied,
                else => errors.IOError.ReadError,
            };
        };
        defer file.close();

        const content = file.readToEndAlloc(allocator, 1024 * 1024) catch |err| {
            return switch (err) {
                error.OutOfMemory => errors.CoreError.OutOfMemory,
                else => errors.IOError.ReadError,
            };
        };
        defer allocator.free(content);

        return parseFromJson(content);
    }

    pub fn saveToFile(self: *const ZigAIConfig, allocator: std.mem.Allocator, path: []const u8) errors.ZigAIError!void {
        const json = try self.toJson(allocator);
        defer allocator.free(json);

        const file = std.fs.cwd().createFile(path, .{}) catch |err| {
            return switch (err) {
                error.AccessDenied => errors.IOError.FileAccessDenied,
                error.OutOfMemory => errors.CoreError.OutOfMemory,
                else => errors.IOError.WriteError,
            };
        };
        defer file.close();

        file.writeAll(json) catch |err| {
            return switch (err) {
                error.OutOfMemory => errors.CoreError.OutOfMemory,
                else => errors.IOError.WriteError,
            };
        };
    }

    pub fn toJson(self: *const ZigAIConfig, allocator: std.mem.Allocator) errors.ZigAIError![]u8 {
        // Simplified JSON serialization - in real implementation, use a proper JSON library
        var json = std.ArrayList(u8).init(allocator);
        defer json.deinit();

        try json.appendSlice("{\n");
        try json.appendSlice("  \"version\": \"");
        try json.appendSlice(self.version);
        try json.appendSlice("\",\n");
        try json.appendSlice("  \"enable_profiling\": ");
        try json.appendSlice(if (self.enable_profiling) "true" else "false");
        try json.appendSlice(",\n");
        try json.appendSlice("  \"enable_debugging\": ");
        try json.appendSlice(if (self.enable_debugging) "true" else "false");
        try json.appendSlice("\n}");

        return json.toOwnedSlice() catch errors.CoreError.OutOfMemory;
    }

    pub fn parseFromJson(json: []const u8) errors.ZigAIError!ZigAIConfig {
        // Simplified JSON parsing - in real implementation, use a proper JSON library
        _ = json;
        return ZigAIConfig{};
    }

    pub fn getPreset(preset: Preset) ZigAIConfig {
        return switch (preset) {
            .iot => ZigAIConfig{
                .memory = MemoryConfig.getRecommendedForDevice(.iot, 512),
                .compute = ComputeConfig{
                    .device_type = .iot,
                    .num_threads = 1,
                    .enable_gpu = false,
                    .optimization_level = .basic,
                },
                .model = ModelConfig{
                    .precision = .int8,
                    .enable_quantization = true,
                    .batch_size = 1,
                    .max_sequence_length = 128,
                },
                .network = NetworkConfig{
                    .max_connections = 10,
                    .timeout_seconds = 10,
                    .max_request_size_mb = 1,
                },
                .logging = LoggingConfig{
                    .level = .warn,
                    .enable_file_logging = false,
                },
                .enable_profiling = false,
                .enable_debugging = false,
            },
            .desktop => ZigAIConfig{
                .memory = MemoryConfig.getRecommendedForDevice(.desktop, 8192),
                .compute = ComputeConfig{
                    .device_type = .desktop,
                    .optimization_level = .balanced,
                },
                .model = ModelConfig{
                    .precision = .fp32,
                    .batch_size = 4,
                    .max_sequence_length = 512,
                },
                .enable_profiling = true,
                .enable_debugging = true,
            },
            .server => ZigAIConfig{
                .memory = MemoryConfig.getRecommendedForDevice(.server, 32768),
                .compute = ComputeConfig{
                    .device_type = .server,
                    .optimization_level = .aggressive,
                },
                .model = ModelConfig{
                    .precision = .fp16,
                    .batch_size = 16,
                    .max_sequence_length = 1024,
                },
                .network = NetworkConfig{
                    .max_connections = 1000,
                    .timeout_seconds = 60,
                    .max_request_size_mb = 100,
                },
                .enable_profiling = true,
                .enable_debugging = false,
            },
        };
    }

    pub const Preset = enum {
        iot,
        desktop,
        server,
    };
};

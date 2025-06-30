const std = @import("std");
const Allocator = std.mem.Allocator;

// Import platform types
const Environment = @import("../platform/core.zig").Environment;
const DeploymentTarget = @import("../platform/core.zig").DeploymentTarget;

/// Component-specific configuration
pub const ComponentConfig = struct {
    // Tensor Core configuration
    tensor_core: struct {
        precision: []const u8 = "fp32",
        simd_level: []const u8 = "auto",
        memory_pool_size_mb: usize = 256,
        enable_gpu: bool = true,
    } = .{},
    
    // ONNX Parser configuration
    onnx_parser: struct {
        validation_level: []const u8 = "strict",
        enable_optimization: bool = true,
        max_model_size_mb: usize = 1024,
    } = .{},
    
    // Inference Engine configuration
    inference_engine: struct {
        optimization_level: []const u8 = "balanced",
        max_batch_size: usize = 4,
        num_threads: ?u32 = null,
        enable_profiling: bool = false,
        device_type: []const u8 = "auto",
    } = .{},
    
    // Model Server configuration
    model_server: struct {
        host: []const u8 = "127.0.0.1",
        port: u16 = 8080,
        max_connections: u32 = 100,
        enable_cors: bool = true,
        enable_metrics: bool = true,
        enable_websockets: bool = true,
        worker_threads: ?u32 = null,
    } = .{},
};

/// Environment-specific configuration
pub const EnvironmentConfig = struct {
    environment: Environment,
    deployment_target: DeploymentTarget,
    
    // Resource limits
    max_memory_mb: ?usize = null,
    max_cpu_cores: ?u32 = null,
    enable_gpu: bool = true,
    
    // Platform settings
    enable_monitoring: bool = true,
    enable_logging: bool = true,
    enable_metrics: bool = true,
    enable_auto_scaling: bool = false,
    
    // Component configurations
    components: ComponentConfig = .{},
    
    // Directories
    data_directory: []const u8 = "./data",
    log_directory: []const u8 = "./logs",
    model_directory: []const u8 = "./models",
    
    // Network settings
    admin_port: u16 = 8081,
    metrics_port: u16 = 9090,
    health_check_interval_ms: u32 = 30000,
    
    pub fn init(env: Environment, target: DeploymentTarget) EnvironmentConfig {
        return EnvironmentConfig{
            .environment = env,
            .deployment_target = target,
        };
    }
};

/// Configuration validation result
pub const ValidationResult = struct {
    valid: bool,
    errors: std.ArrayList([]const u8),
    warnings: std.ArrayList([]const u8),
    
    pub fn init(allocator: Allocator) ValidationResult {
        return ValidationResult{
            .valid = true,
            .errors = std.ArrayList([]const u8).init(allocator),
            .warnings = std.ArrayList([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *ValidationResult) void {
        for (self.errors.items) |error_msg| {
            self.errors.allocator.free(error_msg);
        }
        for (self.warnings.items) |warning_msg| {
            self.warnings.allocator.free(warning_msg);
        }
        self.errors.deinit();
        self.warnings.deinit();
    }
    
    pub fn addError(self: *ValidationResult, message: []const u8) !void {
        self.valid = false;
        try self.errors.append(try self.errors.allocator.dupe(u8, message));
    }
    
    pub fn addWarning(self: *ValidationResult, message: []const u8) !void {
        try self.warnings.append(try self.warnings.allocator.dupe(u8, message));
    }
};

/// Configuration manager
pub const ConfigManager = struct {
    allocator: Allocator,
    config_file_path: ?[]const u8,
    environment_configs: std.EnumMap(Environment, EnvironmentConfig),
    current_config: ?EnvironmentConfig,
    
    const Self = @This();

    /// Initialize configuration manager
    pub fn init(allocator: Allocator, config_file: ?[]const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .config_file_path = config_file,
            .environment_configs = std.EnumMap(Environment, EnvironmentConfig){},
            .current_config = null,
        };
        
        // Initialize default configurations for all environments
        try self.initializeDefaultConfigs();
        
        // Load from file if provided
        if (config_file) |file_path| {
            try self.loadFromFile(file_path);
        }
        
        return self;
    }

    /// Deinitialize configuration manager
    pub fn deinit(self: *Self) void {
        _ = self;
        // Clean up any allocated configuration data
    }

    /// Get configuration for specific environment
    pub fn getEnvironmentConfig(self: *const Self, env: Environment) !EnvironmentConfig {
        return self.environment_configs.get(env) orelse {
            std.log.err("No configuration found for environment: {s}", .{env.toString()});
            return error.ConfigurationNotFound;
        };
    }

    /// Set configuration for specific environment
    pub fn setEnvironmentConfig(self: *Self, env: Environment, config: EnvironmentConfig) !void {
        self.environment_configs.put(env, config);
        
        // If this is the current environment, update current config
        if (self.current_config) |current| {
            if (current.environment == env) {
                self.current_config = config;
            }
        }
    }

    /// Get current active configuration
    pub fn getCurrentConfig(self: *const Self) ?EnvironmentConfig {
        return self.current_config;
    }

    /// Set current active configuration
    pub fn setCurrentConfig(self: *Self, env: Environment) !void {
        self.current_config = try self.getEnvironmentConfig(env);
    }

    /// Validate configuration
    pub fn validateConfig(self: *const Self, config: EnvironmentConfig) !ValidationResult {
        var result = ValidationResult.init(self.allocator);
        
        // Validate resource limits
        if (config.max_memory_mb) |memory| {
            if (memory < 64) {
                try result.addError("Memory limit too low (minimum 64MB)");
            } else if (memory < 256) {
                try result.addWarning("Memory limit is quite low, consider increasing for better performance");
            }
        }
        
        if (config.max_cpu_cores) |cores| {
            if (cores == 0) {
                try result.addError("CPU core count cannot be zero");
            } else if (cores == 1) {
                try result.addWarning("Single CPU core may limit performance");
            }
        }
        
        // Validate network ports
        if (config.admin_port == config.metrics_port) {
            try result.addError("Admin port and metrics port cannot be the same");
        }
        
        if (config.components.model_server.port == config.admin_port or 
            config.components.model_server.port == config.metrics_port) {
            try result.addError("Model server port conflicts with admin or metrics port");
        }
        
        // Validate directories
        const directories = [_][]const u8{
            config.data_directory,
            config.log_directory,
            config.model_directory,
        };
        
        for (directories) |dir| {
            if (dir.len == 0) {
                try result.addError("Directory path cannot be empty");
            }
        }
        
        // Validate component configurations
        try self.validateComponentConfig(&result, config.components);
        
        return result;
    }

    /// Generate configuration file
    pub fn generateConfigFile(self: *const Self, env: Environment, file_path: []const u8) !void {
        const config = try self.getEnvironmentConfig(env);
        
        const file = try std.fs.cwd().createFile(file_path, .{});
        defer file.close();
        
        const writer = file.writer();
        
        // Write YAML-style configuration
        try writer.print("# Zig AI Platform Configuration\n");
        try writer.print("# Environment: {s}\n\n", .{env.toString()});
        
        try writer.print("environment: {s}\n", .{config.environment.toString()});
        try writer.print("deployment_target: {s}\n\n", .{config.deployment_target.toString()});
        
        // Resource limits
        try writer.print("resources:\n");
        if (config.max_memory_mb) |memory| {
            try writer.print("  max_memory_mb: {}\n", .{memory});
        }
        if (config.max_cpu_cores) |cores| {
            try writer.print("  max_cpu_cores: {}\n", .{cores});
        }
        try writer.print("  enable_gpu: {}\n\n", .{config.enable_gpu});
        
        // Platform settings
        try writer.print("platform:\n");
        try writer.print("  enable_monitoring: {}\n", .{config.enable_monitoring});
        try writer.print("  enable_logging: {}\n", .{config.enable_logging});
        try writer.print("  enable_metrics: {}\n", .{config.enable_metrics});
        try writer.print("  enable_auto_scaling: {}\n", .{config.enable_auto_scaling});
        try writer.print("  admin_port: {}\n", .{config.admin_port});
        try writer.print("  metrics_port: {}\n", .{config.metrics_port});
        try writer.print("  health_check_interval_ms: {}\n\n", .{config.health_check_interval_ms});
        
        // Component configurations
        try self.writeComponentConfig(writer, config.components);
        
        std.log.info("Configuration file generated: {s}", .{file_path});
    }

    /// Reload configuration from file
    pub fn reload(self: *Self) !void {
        if (self.config_file_path) |file_path| {
            try self.loadFromFile(file_path);
            std.log.info("Configuration reloaded from: {s}", .{file_path});
        }
    }

    /// Get configuration value by key path
    pub fn getValue(self: *const Self, env: Environment, key_path: []const u8) ![]const u8 {
        const config = try self.getEnvironmentConfig(env);
        
        // Simple key path resolution (in production, this would be more sophisticated)
        if (std.mem.eql(u8, key_path, "environment")) {
            return config.environment.toString();
        } else if (std.mem.eql(u8, key_path, "deployment_target")) {
            return config.deployment_target.toString();
        } else if (std.mem.eql(u8, key_path, "components.model_server.host")) {
            return config.components.model_server.host;
        } else if (std.mem.eql(u8, key_path, "components.model_server.port")) {
            return try std.fmt.allocPrint(self.allocator, "{}", .{config.components.model_server.port});
        }
        
        return error.KeyNotFound;
    }

    /// Set configuration value by key path
    pub fn setValue(self: *Self, env: Environment, key_path: []const u8, value: []const u8) !void {
        // TODO: Implement configuration value setting
        _ = self;
        _ = env;
        _ = key_path;
        _ = value;
        return error.NotImplemented;
    }

    // Private methods

    /// Initialize default configurations
    fn initializeDefaultConfigs(self: *Self) !void {
        // Development configuration
        var dev_config = EnvironmentConfig.init(.development, .desktop);
        dev_config.max_memory_mb = 1024;
        dev_config.max_cpu_cores = 4;
        dev_config.enable_auto_scaling = false;
        dev_config.components.model_server.port = 3000;
        dev_config.components.model_server.max_connections = 50;
        self.environment_configs.put(.development, dev_config);
        
        // Testing configuration
        var test_config = EnvironmentConfig.init(.testing, .desktop);
        test_config.max_memory_mb = 512;
        test_config.max_cpu_cores = 2;
        test_config.enable_monitoring = false;
        test_config.enable_metrics = false;
        test_config.components.model_server.port = 8081;
        test_config.components.model_server.max_connections = 10;
        self.environment_configs.put(.testing, test_config);
        
        // Staging configuration
        var staging_config = EnvironmentConfig.init(.staging, .server);
        staging_config.max_memory_mb = 4096;
        staging_config.max_cpu_cores = 8;
        staging_config.enable_auto_scaling = true;
        staging_config.components.model_server.host = "0.0.0.0";
        staging_config.components.model_server.max_connections = 500;
        self.environment_configs.put(.staging, staging_config);
        
        // Production configuration
        var prod_config = EnvironmentConfig.init(.production, .server);
        prod_config.max_memory_mb = 8192;
        prod_config.max_cpu_cores = 16;
        prod_config.enable_auto_scaling = true;
        prod_config.components.model_server.host = "0.0.0.0";
        prod_config.components.model_server.max_connections = 1000;
        prod_config.components.inference_engine.optimization_level = "max";
        prod_config.components.inference_engine.max_batch_size = 32;
        self.environment_configs.put(.production, prod_config);
    }

    /// Load configuration from file
    fn loadFromFile(self: *Self, file_path: []const u8) !void {
        // TODO: Implement YAML/JSON configuration file parsing
        _ = self;
        _ = file_path;
        std.log.info("Loading configuration from file: {s}", .{file_path});
        // For now, just log that we would load from file
    }

    /// Validate component configuration
    fn validateComponentConfig(self: *const Self, result: *ValidationResult, components: ComponentConfig) !void {
        _ = self;
        
        // Validate tensor core config
        const valid_precisions = [_][]const u8{ "fp16", "fp32", "mixed" };
        var precision_valid = false;
        for (valid_precisions) |precision| {
            if (std.mem.eql(u8, components.tensor_core.precision, precision)) {
                precision_valid = true;
                break;
            }
        }
        if (!precision_valid) {
            try result.addError("Invalid tensor core precision. Must be one of: fp16, fp32, mixed");
        }
        
        // Validate inference engine config
        const valid_opt_levels = [_][]const u8{ "none", "basic", "balanced", "aggressive", "max" };
        var opt_level_valid = false;
        for (valid_opt_levels) |level| {
            if (std.mem.eql(u8, components.inference_engine.optimization_level, level)) {
                opt_level_valid = true;
                break;
            }
        }
        if (!opt_level_valid) {
            try result.addError("Invalid optimization level. Must be one of: none, basic, balanced, aggressive, max");
        }
        
        // Validate model server config
        if (components.model_server.max_connections == 0) {
            try result.addError("Model server max_connections cannot be zero");
        }
        
        if (components.model_server.port < 1024) {
            try result.addWarning("Model server port is below 1024, may require elevated privileges");
        }
    }

    /// Write component configuration to file
    fn writeComponentConfig(self: *const Self, writer: anytype, components: ComponentConfig) !void {
        _ = self;
        
        try writer.print("components:\n");
        
        // Tensor core
        try writer.print("  tensor_core:\n");
        try writer.print("    precision: {s}\n", .{components.tensor_core.precision});
        try writer.print("    simd_level: {s}\n", .{components.tensor_core.simd_level});
        try writer.print("    memory_pool_size_mb: {}\n", .{components.tensor_core.memory_pool_size_mb});
        try writer.print("    enable_gpu: {}\n\n", .{components.tensor_core.enable_gpu});
        
        // ONNX parser
        try writer.print("  onnx_parser:\n");
        try writer.print("    validation_level: {s}\n", .{components.onnx_parser.validation_level});
        try writer.print("    enable_optimization: {}\n", .{components.onnx_parser.enable_optimization});
        try writer.print("    max_model_size_mb: {}\n\n", .{components.onnx_parser.max_model_size_mb});
        
        // Inference engine
        try writer.print("  inference_engine:\n");
        try writer.print("    optimization_level: {s}\n", .{components.inference_engine.optimization_level});
        try writer.print("    max_batch_size: {}\n", .{components.inference_engine.max_batch_size});
        if (components.inference_engine.num_threads) |threads| {
            try writer.print("    num_threads: {}\n", .{threads});
        }
        try writer.print("    enable_profiling: {}\n", .{components.inference_engine.enable_profiling});
        try writer.print("    device_type: {s}\n\n", .{components.inference_engine.device_type});
        
        // Model server
        try writer.print("  model_server:\n");
        try writer.print("    host: {s}\n", .{components.model_server.host});
        try writer.print("    port: {}\n", .{components.model_server.port});
        try writer.print("    max_connections: {}\n", .{components.model_server.max_connections});
        try writer.print("    enable_cors: {}\n", .{components.model_server.enable_cors});
        try writer.print("    enable_metrics: {}\n", .{components.model_server.enable_metrics});
        try writer.print("    enable_websockets: {}\n", .{components.model_server.enable_websockets});
        if (components.model_server.worker_threads) |threads| {
            try writer.print("    worker_threads: {}\n", .{threads});
        }
    }
};

/// Configuration presets for different environments
pub const ConfigPresets = struct {
    /// Get IoT device configuration
    pub fn iot() EnvironmentConfig {
        var config = EnvironmentConfig.init(.production, .iot);
        config.max_memory_mb = 64;
        config.max_cpu_cores = 1;
        config.enable_gpu = false;
        config.enable_auto_scaling = false;
        config.enable_metrics = false;
        
        // Optimize for IoT
        config.components.tensor_core.precision = "fp16";
        config.components.tensor_core.memory_pool_size_mb = 16;
        config.components.inference_engine.optimization_level = "aggressive";
        config.components.inference_engine.max_batch_size = 1;
        config.components.model_server.max_connections = 5;
        config.components.model_server.enable_websockets = false;
        
        return config;
    }
    
    /// Get desktop development configuration
    pub fn desktop() EnvironmentConfig {
        var config = EnvironmentConfig.init(.development, .desktop);
        config.max_memory_mb = 2048;
        config.max_cpu_cores = 4;
        config.enable_gpu = true;
        
        config.components.model_server.port = 3000;
        config.components.model_server.max_connections = 50;
        
        return config;
    }
    
    /// Get production server configuration
    pub fn production() EnvironmentConfig {
        var config = EnvironmentConfig.init(.production, .server);
        config.max_memory_mb = 8192;
        config.max_cpu_cores = 16;
        config.enable_gpu = true;
        config.enable_auto_scaling = true;
        
        config.components.tensor_core.precision = "mixed";
        config.components.inference_engine.optimization_level = "max";
        config.components.inference_engine.max_batch_size = 32;
        config.components.model_server.host = "0.0.0.0";
        config.components.model_server.max_connections = 1000;
        
        return config;
    }
};

/// Global configuration instance
pub var global_config: ?*ConfigManager = null;

/// Initialize global configuration
pub fn initGlobalConfig(allocator: Allocator, config_file: ?[]const u8) !void {
    global_config = try allocator.create(ConfigManager);
    global_config.?.* = try ConfigManager.init(allocator, config_file);
}

/// Get global configuration
pub fn getGlobalConfig() ?*ConfigManager {
    return global_config;
}

/// Cleanup global configuration
pub fn deinitGlobalConfig(allocator: Allocator) void {
    if (global_config) |config| {
        config.deinit();
        allocator.destroy(config);
        global_config = null;
    }
}

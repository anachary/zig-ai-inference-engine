const std = @import("std");

/// Zig AI Platform - Unified orchestrator and platform integration
/// 
/// This library provides the ultimate integration layer for the complete
/// Zig AI Ecosystem, following the Single Responsibility Principle.
/// It orchestrates and coordinates all ecosystem components into a unified,
/// production-ready AI platform.
///
/// Key Features:
/// - Unified orchestration of all ecosystem components
/// - Comprehensive configuration management system
/// - Deployment tools for IoT, desktop, server, cloud, and Kubernetes
/// - Platform-level services (health monitoring, logging, metrics)
/// - Environment-specific optimizations and presets
/// - End-to-end integration testing and validation
/// - Production deployment and scaling tools
///
/// Dependencies:
/// - zig-tensor-core: For tensor operations and memory management
/// - zig-onnx-parser: For ONNX model parsing and validation
/// - zig-inference-engine: For high-performance model execution
/// - zig-model-server: For HTTP API and CLI interfaces
///
/// Usage:
/// ```zig
/// const std = @import("std");
/// const ai_platform = @import("zig-ai-platform");
///
/// pub fn main() !void {
///     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
///     defer _ = gpa.deinit();
///     const allocator = gpa.allocator();
///
///     // Initialize the complete AI platform
///     const platform_config = ai_platform.PlatformConfig{
///         .environment = .production,
///         .deployment_target = .server,
///         .enable_monitoring = true,
///         .enable_auto_scaling = true,
///     };
///
///     var platform = try ai_platform.Platform.init(allocator, platform_config);
///     defer platform.deinit();
///
///     // Start all services
///     try platform.start();
///
///     // Run the platform
///     try platform.run();
/// }
/// ```

// Re-export platform core
pub const Platform = @import("platform/core.zig").Platform;
pub const PlatformConfig = @import("platform/core.zig").PlatformConfig;
pub const PlatformStats = @import("platform/core.zig").PlatformStats;
pub const Environment = @import("platform/core.zig").Environment;
pub const DeploymentTarget = @import("platform/core.zig").DeploymentTarget;
pub const ComponentStatus = @import("platform/core.zig").ComponentStatus;
pub const ComponentInfo = @import("platform/core.zig").ComponentInfo;

// Re-export configuration management
pub const ConfigManager = @import("config/manager.zig").ConfigManager;
pub const EnvironmentConfig = @import("config/manager.zig").EnvironmentConfig;
pub const ComponentConfig = @import("config/manager.zig").ComponentConfig;
pub const ValidationResult = @import("config/manager.zig").ValidationResult;
pub const ConfigPresets = @import("config/manager.zig").ConfigPresets;

// Re-export platform services
pub const HealthMonitor = @import("services/health.zig").HealthMonitor;
pub const HealthStatus = @import("services/health.zig").HealthStatus;
pub const HealthCheckResult = @import("services/health.zig").HealthCheckResult;
pub const LogAggregator = @import("services/logging.zig").LogAggregator;
pub const LogLevel = @import("services/logging.zig").LogLevel;
pub const LogEntry = @import("services/logging.zig").LogEntry;
pub const MetricsCollector = @import("services/metrics.zig").MetricsCollector;
pub const Metric = @import("services/metrics.zig").Metric;
pub const MetricType = @import("services/metrics.zig").MetricType;

// Re-export deployment management
pub const DeploymentManager = @import("deployment/manager.zig").DeploymentManager;
pub const DeploymentConfig = @import("deployment/manager.zig").DeploymentConfig;
pub const DeploymentPlan = @import("deployment/manager.zig").DeploymentPlan;
pub const DeploymentResult = @import("deployment/manager.zig").DeploymentResult;
pub const DeploymentStatus = @import("deployment/manager.zig").DeploymentStatus;

// Re-export CLI interface
pub const CLI = @import("cli/cli.zig").CLI;
pub const Command = @import("cli/cli.zig").Command;
pub const Args = @import("cli/cli.zig").Args;

// Import ecosystem components for re-export
pub const tensor_core = @import("zig-tensor-core");
pub const onnx_parser = @import("zig-onnx-parser");
pub const inference_engine = @import("zig-inference-engine");
pub const model_server = @import("zig-model-server");

/// Library version information
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;
    pub const string = "0.1.0";
};

/// Library information
pub const info = struct {
    pub const name = "zig-ai-platform";
    pub const description = "Unified orchestrator and platform integration for the complete Zig AI Ecosystem";
    pub const author = "Zig AI Ecosystem";
    pub const license = "MIT";
    pub const repository = "https://github.com/zig-ai/zig-ai-platform";
};

/// Supported deployment targets
pub const SupportedTargets = enum {
    iot,
    desktop,
    server,
    cloud,
    kubernetes,
};

/// Supported environments
pub const SupportedEnvironments = enum {
    development,
    testing,
    staging,
    production,
};

/// Create default platform configuration
pub fn defaultPlatformConfig() PlatformConfig {
    return PlatformConfig{
        .environment = .development,
        .deployment_target = .desktop,
        .enable_monitoring = true,
        .enable_logging = true,
        .enable_metrics = true,
        .enable_auto_scaling = false,
        .health_check_interval_ms = 30000,
        .log_level = .info,
        .metrics_port = 9090,
        .admin_port = 8081,
    };
}

/// Create IoT-optimized platform configuration
pub fn iotPlatformConfig() PlatformConfig {
    return PlatformConfig{
        .environment = .production,
        .deployment_target = .iot,
        .enable_monitoring = true,
        .enable_logging = false,
        .enable_metrics = false,
        .enable_auto_scaling = false,
        .health_check_interval_ms = 60000,
        .log_level = .err,
        .metrics_port = 9090,
        .admin_port = 8081,
        .max_memory_mb = 64,
        .max_cpu_cores = 1,
        .enable_gpu = false,
    };
}

/// Create production server platform configuration
pub fn productionPlatformConfig() PlatformConfig {
    return PlatformConfig{
        .environment = .production,
        .deployment_target = .server,
        .enable_monitoring = true,
        .enable_logging = true,
        .enable_metrics = true,
        .enable_auto_scaling = true,
        .health_check_interval_ms = 15000,
        .log_level = .info,
        .metrics_port = 9090,
        .admin_port = 8081,
        .max_memory_mb = 8192,
        .max_cpu_cores = 16,
        .enable_gpu = true,
    };
}

/// Create development platform configuration
pub fn developmentPlatformConfig() PlatformConfig {
    return PlatformConfig{
        .environment = .development,
        .deployment_target = .desktop,
        .enable_monitoring = true,
        .enable_logging = true,
        .enable_metrics = true,
        .enable_auto_scaling = false,
        .health_check_interval_ms = 30000,
        .log_level = .debug,
        .metrics_port = 9091,
        .admin_port = 8082,
        .max_memory_mb = 2048,
        .max_cpu_cores = 4,
        .enable_gpu = true,
    };
}

/// Quick start function - creates and starts platform
pub fn quickStart(allocator: std.mem.Allocator, config: PlatformConfig) !Platform {
    var platform = try Platform.init(allocator, config);
    try platform.start();
    return platform;
}

/// Quick start with default configuration
pub fn quickStartDefault(allocator: std.mem.Allocator) !Platform {
    return quickStart(allocator, defaultPlatformConfig());
}

/// Quick start for IoT deployment
pub fn quickStartIoT(allocator: std.mem.Allocator) !Platform {
    return quickStart(allocator, iotPlatformConfig());
}

/// Quick start for production deployment
pub fn quickStartProduction(allocator: std.mem.Allocator) !Platform {
    return quickStart(allocator, productionPlatformConfig());
}

/// Quick start for development
pub fn quickStartDevelopment(allocator: std.mem.Allocator) !Platform {
    return quickStart(allocator, developmentPlatformConfig());
}

/// Check if deployment target is supported
pub fn isTargetSupported(target: []const u8) bool {
    inline for (std.meta.fields(SupportedTargets)) |field| {
        if (std.mem.eql(u8, target, field.name)) {
            return true;
        }
    }
    return false;
}

/// Check if environment is supported
pub fn isEnvironmentSupported(env: []const u8) bool {
    inline for (std.meta.fields(SupportedEnvironments)) |field| {
        if (std.mem.eql(u8, env, field.name)) {
            return true;
        }
    }
    return false;
}

/// Get all supported deployment targets
pub fn getSupportedTargets() []const []const u8 {
    return &[_][]const u8{ "iot", "desktop", "server", "cloud", "kubernetes" };
}

/// Get all supported environments
pub fn getSupportedEnvironments() []const []const u8 {
    return &[_][]const u8{ "development", "testing", "staging", "production" };
}

/// Library initialization function (optional)
pub fn init() void {
    std.log.info("🎯 Zig AI Platform v{s} initialized", .{version.string});
    std.log.info("   Orchestrating the complete Zig AI Ecosystem", .{});
}

/// Library cleanup function (optional)
pub fn deinit() void {
    std.log.info("🎯 Zig AI Platform v{s} deinitialized", .{version.string});
}

/// Test function to verify library functionality
pub fn test_basic_functionality() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test platform creation
    var platform = try Platform.init(allocator, defaultPlatformConfig());
    defer platform.deinit();

    // Test configuration variants
    const iot_config = iotPlatformConfig();
    const prod_config = productionPlatformConfig();
    const dev_config = developmentPlatformConfig();

    // Basic validation
    try std.testing.expect(iot_config.deployment_target == .iot);
    try std.testing.expect(prod_config.deployment_target == .server);
    try std.testing.expect(dev_config.deployment_target == .desktop);

    // Test target support
    try std.testing.expect(isTargetSupported("iot"));
    try std.testing.expect(isTargetSupported("server"));
    try std.testing.expect(!isTargetSupported("invalid"));

    // Test environment support
    try std.testing.expect(isEnvironmentSupported("production"));
    try std.testing.expect(isEnvironmentSupported("development"));
    try std.testing.expect(!isEnvironmentSupported("invalid"));

    std.log.info("Basic functionality test passed");
}

/// Get ecosystem component versions
pub fn getEcosystemVersions() struct {
    tensor_core: []const u8,
    onnx_parser: []const u8,
    inference_engine: []const u8,
    model_server: []const u8,
    platform: []const u8,
} {
    return .{
        .tensor_core = tensor_core.version.string,
        .onnx_parser = onnx_parser.version.string,
        .inference_engine = inference_engine.version.string,
        .model_server = model_server.version.string,
        .platform = version.string,
    };
}

/// Get ecosystem information
pub fn getEcosystemInfo() struct {
    total_components: u32,
    platform_version: []const u8,
    description: []const u8,
    repository: []const u8,
} {
    return .{
        .total_components = 5, // tensor-core, onnx-parser, inference-engine, model-server, platform
        .platform_version = version.string,
        .description = info.description,
        .repository = info.repository,
    };
}

/// Create deployment configuration for target
pub fn createDeploymentConfig(target: DeploymentTarget, env: Environment) DeploymentConfig {
    return DeploymentConfig{
        .target = target,
        .environment = env,
        .enable_auto_start = true,
        .enable_health_checks = true,
        .enable_monitoring = true,
        .replicas = if (target == .server or target == .cloud or target == .kubernetes) 3 else 1,
    };
}

/// Validate platform configuration
pub fn validatePlatformConfig(config: PlatformConfig) !void {
    // Validate resource limits
    if (config.max_memory_mb) |memory| {
        if (memory < 64) {
            return error.InsufficientMemory;
        }
    }
    
    if (config.max_cpu_cores) |cores| {
        if (cores == 0) {
            return error.InvalidCpuCores;
        }
    }
    
    // Validate ports
    if (config.admin_port == config.metrics_port) {
        return error.PortConflict;
    }
    
    // Validate intervals
    if (config.health_check_interval_ms < 1000) {
        return error.InvalidHealthCheckInterval;
    }
}

// Tests
test "library initialization" {
    init();
    deinit();
}

test "configuration creation" {
    const default_cfg = defaultPlatformConfig();
    const iot_cfg = iotPlatformConfig();
    const prod_cfg = productionPlatformConfig();
    const dev_cfg = developmentPlatformConfig();

    // Basic validation
    try std.testing.expect(default_cfg.environment == .development);
    try std.testing.expect(iot_cfg.deployment_target == .iot);
    try std.testing.expect(prod_cfg.enable_auto_scaling == true);
    try std.testing.expect(dev_cfg.log_level == .debug);
}

test "target and environment support" {
    try std.testing.expect(isTargetSupported("iot"));
    try std.testing.expect(isTargetSupported("desktop"));
    try std.testing.expect(isTargetSupported("server"));
    try std.testing.expect(isTargetSupported("cloud"));
    try std.testing.expect(isTargetSupported("kubernetes"));
    try std.testing.expect(!isTargetSupported("invalid"));

    try std.testing.expect(isEnvironmentSupported("development"));
    try std.testing.expect(isEnvironmentSupported("testing"));
    try std.testing.expect(isEnvironmentSupported("staging"));
    try std.testing.expect(isEnvironmentSupported("production"));
    try std.testing.expect(!isEnvironmentSupported("invalid"));
}

test "platform creation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var platform = try Platform.init(allocator, defaultPlatformConfig());
    defer platform.deinit();

    const stats = platform.getStatus();
    try std.testing.expect(stats.uptime_seconds >= 0);
    try std.testing.expect(stats.total_requests == 0);
}

test "configuration validation" {
    var valid_config = defaultPlatformConfig();
    try validatePlatformConfig(valid_config);

    // Test invalid memory
    var invalid_memory_config = defaultPlatformConfig();
    invalid_memory_config.max_memory_mb = 32;
    try std.testing.expectError(error.InsufficientMemory, validatePlatformConfig(invalid_memory_config));

    // Test invalid CPU cores
    var invalid_cpu_config = defaultPlatformConfig();
    invalid_cpu_config.max_cpu_cores = 0;
    try std.testing.expectError(error.InvalidCpuCores, validatePlatformConfig(invalid_cpu_config));

    // Test port conflict
    var port_conflict_config = defaultPlatformConfig();
    port_conflict_config.admin_port = port_conflict_config.metrics_port;
    try std.testing.expectError(error.PortConflict, validatePlatformConfig(port_conflict_config));
}

test "basic functionality" {
    try test_basic_functionality();
}

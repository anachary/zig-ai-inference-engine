const std = @import("std");
const ai_platform = @import("zig-ai-platform");

/// Configuration Management Example
/// 
/// This example demonstrates how to use the comprehensive configuration
/// management system for different environments and deployment scenarios.
/// 
/// Use Cases: DevOps automation, environment-specific deployments,
/// configuration validation, and runtime configuration updates.

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("⚙️  Configuration Management Demo", .{});
    std.log.info("==================================", .{});

    // Initialize configuration manager
    var config_manager = try ai_platform.ConfigManager.init(allocator, null);
    defer config_manager.deinit();

    // Demonstrate environment-specific configurations
    try demonstrateEnvironmentConfigs(&config_manager);

    // Demonstrate configuration validation
    try demonstrateConfigValidation(&config_manager, allocator);

    // Demonstrate configuration presets
    demonstrateConfigPresets();

    // Demonstrate configuration file generation
    try demonstrateConfigGeneration(&config_manager);

    // Demonstrate runtime configuration management
    try demonstrateRuntimeConfig(&config_manager, allocator);

    std.log.info("✅ Configuration management demo completed!", .{});
}

/// Demonstrate environment-specific configurations
fn demonstrateEnvironmentConfigs(config_manager: *ai_platform.ConfigManager) !void {
    std.log.info("\n🌍 Environment-Specific Configurations:", .{});
    
    const environments = [_]ai_platform.Environment{ .development, .testing, .staging, .production };
    
    for (environments) |env| {
        const env_config = try config_manager.getEnvironmentConfig(env);
        
        std.log.info("   {s}:", .{env.toString()});
        std.log.info("     Target: {s}", .{env_config.deployment_target.toString()});
        
        if (env_config.max_memory_mb) |memory| {
            std.log.info("     Memory: {}MB", .{memory});
        }
        if (env_config.max_cpu_cores) |cores| {
            std.log.info("     CPU Cores: {}", .{cores});
        }
        
        std.log.info("     GPU: {}", .{env_config.enable_gpu});
        std.log.info("     Auto-scaling: {}", .{env_config.enable_auto_scaling});
        std.log.info("     Monitoring: {}", .{env_config.enable_monitoring});
        std.log.info("     Model Server Port: {}", .{env_config.components.model_server.port});
        std.log.info("     Max Connections: {}", .{env_config.components.model_server.max_connections});
    }
}

/// Demonstrate configuration validation
fn demonstrateConfigValidation(config_manager: *ai_platform.ConfigManager, allocator: std.mem.Allocator) !void {
    std.log.info("\n🔍 Configuration Validation:", .{});
    
    // Test valid configurations
    const environments = [_]ai_platform.Environment{ .development, .testing, .staging, .production };
    
    for (environments) |env| {
        const env_config = try config_manager.getEnvironmentConfig(env);
        var validation_result = try config_manager.validateConfig(env_config);
        defer validation_result.deinit();
        
        std.log.info("   {s}: {s}", .{
            env.toString(),
            if (validation_result.valid) "✅ Valid" else "❌ Invalid"
        });
        
        if (validation_result.errors.items.len > 0) {
            std.log.info("     Errors:", .{});
            for (validation_result.errors.items) |error_msg| {
                std.log.info("       - {s}", .{error_msg});
            }
        }
        
        if (validation_result.warnings.items.len > 0) {
            std.log.info("     Warnings:", .{});
            for (validation_result.warnings.items) |warning_msg| {
                std.log.info("       - {s}", .{warning_msg});
            }
        }
    }
    
    // Test invalid configuration
    std.log.info("\n   Testing invalid configuration:", .{});
    var invalid_config = try config_manager.getEnvironmentConfig(.development);
    invalid_config.max_memory_mb = 32; // Too low
    invalid_config.admin_port = invalid_config.metrics_port; // Port conflict
    
    var invalid_result = try config_manager.validateConfig(invalid_config);
    defer invalid_result.deinit();
    
    std.log.info("   Invalid config: {s}", .{
        if (invalid_result.valid) "✅ Valid" else "❌ Invalid (expected)"
    });
    
    for (invalid_result.errors.items) |error_msg| {
        std.log.info("     Error: {s}", .{error_msg});
    }
}

/// Demonstrate configuration presets
fn demonstrateConfigPresets() void {
    std.log.info("\n🎯 Configuration Presets:", .{});
    
    // IoT preset
    const iot_preset = ai_platform.ConfigPresets.iot();
    std.log.info("   IoT Preset:", .{});
    std.log.info("     Environment: {s}", .{iot_preset.environment.toString()});
    std.log.info("     Target: {s}", .{iot_preset.deployment_target.toString()});
    std.log.info("     Memory: {}MB", .{iot_preset.max_memory_mb.?});
    std.log.info("     CPU Cores: {}", .{iot_preset.max_cpu_cores.?});
    std.log.info("     GPU: {}", .{iot_preset.enable_gpu});
    std.log.info("     Tensor Precision: {s}", .{iot_preset.components.tensor_core.precision});
    std.log.info("     Max Batch Size: {}", .{iot_preset.components.inference_engine.max_batch_size});
    
    // Desktop preset
    const desktop_preset = ai_platform.ConfigPresets.desktop();
    std.log.info("   Desktop Preset:", .{});
    std.log.info("     Environment: {s}", .{desktop_preset.environment.toString()});
    std.log.info("     Target: {s}", .{desktop_preset.deployment_target.toString()});
    std.log.info("     Memory: {}MB", .{desktop_preset.max_memory_mb.?});
    std.log.info("     CPU Cores: {}", .{desktop_preset.max_cpu_cores.?});
    std.log.info("     Model Server Port: {}", .{desktop_preset.components.model_server.port});
    
    // Production preset
    const production_preset = ai_platform.ConfigPresets.production();
    std.log.info("   Production Preset:", .{});
    std.log.info("     Environment: {s}", .{production_preset.environment.toString()});
    std.log.info("     Target: {s}", .{production_preset.deployment_target.toString()});
    std.log.info("     Memory: {}MB", .{production_preset.max_memory_mb.?});
    std.log.info("     CPU Cores: {}", .{production_preset.max_cpu_cores.?});
    std.log.info("     Auto-scaling: {}", .{production_preset.enable_auto_scaling});
    std.log.info("     Tensor Precision: {s}", .{production_preset.components.tensor_core.precision});
    std.log.info("     Optimization Level: {s}", .{production_preset.components.inference_engine.optimization_level});
    std.log.info("     Max Connections: {}", .{production_preset.components.model_server.max_connections});
}

/// Demonstrate configuration file generation
fn demonstrateConfigGeneration(config_manager: *ai_platform.ConfigManager) !void {
    std.log.info("\n📄 Configuration File Generation:", .{});
    
    // Create config directory
    std.fs.cwd().makeDir("generated-configs") catch |err| {
        if (err != error.PathAlreadyExists) {
            return err;
        }
    };
    
    const environments = [_]ai_platform.Environment{ .development, .testing, .staging, .production };
    
    for (environments) |env| {
        const filename = try std.fmt.allocPrint(
            config_manager.allocator,
            "generated-configs/{s}.yaml",
            .{env.toString()}
        );
        defer config_manager.allocator.free(filename);
        
        try config_manager.generateConfigFile(env, filename);
        std.log.info("   Generated: {s}", .{filename});
    }
    
    std.log.info("   ✅ All configuration files generated", .{});
}

/// Demonstrate runtime configuration management
fn demonstrateRuntimeConfig(config_manager: *ai_platform.ConfigManager, allocator: std.mem.Allocator) !void {
    std.log.info("\n🔄 Runtime Configuration Management:", .{});
    
    // Create a platform with development configuration
    var platform = try ai_platform.Platform.init(allocator, ai_platform.developmentPlatformConfig());
    defer platform.deinit();
    
    std.log.info("   Initial configuration:", .{});
    std.log.info("     Environment: {s}", .{platform.config.environment.toString()});
    std.log.info("     Memory limit: {}MB", .{platform.config.max_memory_mb.?});
    std.log.info("     Health check interval: {}ms", .{platform.config.health_check_interval_ms});
    
    // Start platform
    try platform.start();
    std.log.info("   ✅ Platform started with initial config", .{});
    
    // Simulate configuration reload
    std.log.info("   Simulating configuration reload...", .{});
    try platform.reloadConfiguration();
    std.log.info("   ✅ Configuration reloaded", .{});
    
    // Show component configuration details
    const components = try platform.listComponents(allocator);
    defer allocator.free(components);
    
    std.log.info("   Component status after config reload:", .{});
    for (components) |component| {
        std.log.info("     {s}: {s}", .{
            component.name,
            component.status.toString()
        });
    }
    
    platform.stop();
    std.log.info("   ✅ Platform stopped", .{});
}

/// Configuration management CLI examples
pub fn demonstrateCLIUsage() void {
    std.log.info("\n💻 CLI Configuration Management Examples:", .{});
    
    std.log.info("   Generate configuration for production:", .{});
    std.log.info("     zig-ai-platform config generate --env production --target server", .{});
    
    std.log.info("   Generate IoT configuration:", .{});
    std.log.info("     zig-ai-platform config generate --env production --target iot", .{});
    
    std.log.info("   Validate configuration file:", .{});
    std.log.info("     zig-ai-platform config validate --config production.yaml", .{});
    
    std.log.info("   Show current configuration:", .{});
    std.log.info("     zig-ai-platform config show", .{});
    
    std.log.info("   Set configuration value:", .{});
    std.log.info("     zig-ai-platform config set --key max_memory_mb --value 4096", .{});
    
    std.log.info("   Deploy with specific configuration:", .{});
    std.log.info("     zig-ai-platform deploy --config production.yaml --target server", .{});
}

/// DevOps integration examples
pub const DevOpsIntegration = struct {
    /// Docker deployment configuration
    pub fn dockerDeployment() void {
        std.log.info("🐳 Docker Deployment:", .{});
        std.log.info("   1. Generate configuration: zig-ai-platform config generate --env production", .{});
        std.log.info("   2. Build Docker image with configuration", .{});
        std.log.info("   3. Deploy container with environment variables", .{});
        std.log.info("   4. Monitor via metrics endpoint", .{});
    }
    
    /// Kubernetes deployment configuration
    pub fn kubernetesDeployment() void {
        std.log.info("☸️  Kubernetes Deployment:", .{});
        std.log.info("   1. Generate K8s manifests: zig-ai-platform deploy --target kubernetes", .{});
        std.log.info("   2. Apply ConfigMaps and Secrets", .{});
        std.log.info("   3. Deploy with Helm charts", .{});
        std.log.info("   4. Configure auto-scaling policies", .{});
    }
    
    /// CI/CD pipeline integration
    pub fn cicdIntegration() void {
        std.log.info("🔄 CI/CD Integration:", .{});
        std.log.info("   1. Validate configs in CI: zig-ai-platform config validate", .{});
        std.log.info("   2. Generate environment-specific configs", .{});
        std.log.info("   3. Deploy to staging: zig-ai-platform deploy --env staging", .{});
        std.log.info("   4. Run health checks and promote to production", .{});
    }
};

test "configuration management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test configuration manager initialization
    var config_manager = try ai_platform.ConfigManager.init(allocator, null);
    defer config_manager.deinit();

    // Test environment configuration retrieval
    const dev_config = try config_manager.getEnvironmentConfig(.development);
    try std.testing.expect(dev_config.environment == .development);

    // Test configuration validation
    var validation_result = try config_manager.validateConfig(dev_config);
    defer validation_result.deinit();
    try std.testing.expect(validation_result.valid == true);

    // Test configuration presets
    const iot_preset = ai_platform.ConfigPresets.iot();
    try std.testing.expect(iot_preset.deployment_target == .iot);
    try std.testing.expect(iot_preset.max_memory_mb.? == 64);

    const prod_preset = ai_platform.ConfigPresets.production();
    try std.testing.expect(prod_preset.deployment_target == .server);
    try std.testing.expect(prod_preset.enable_auto_scaling == true);
}

const std = @import("std");
const ai_platform = @import("zig-ai-platform");

/// Basic platform example demonstrating the complete Zig AI Ecosystem
/// orchestration and integration
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== Zig AI Platform - Basic Example ===", .{});

    // Initialize the platform library
    ai_platform.init();
    defer ai_platform.deinit();

    // Display ecosystem information
    const ecosystem_info = ai_platform.getEcosystemInfo();
    const ecosystem_versions = ai_platform.getEcosystemVersions();
    
    std.log.info("🎯 Ecosystem Information:", .{});
    std.log.info("   Total Components: {}", .{ecosystem_info.total_components});
    std.log.info("   Platform Version: {s}", .{ecosystem_info.platform_version});
    std.log.info("   Description: {s}", .{ecosystem_info.description});
    
    std.log.info("📦 Component Versions:", .{});
    std.log.info("   tensor-core: {s}", .{ecosystem_versions.tensor_core});
    std.log.info("   onnx-parser: {s}", .{ecosystem_versions.onnx_parser});
    std.log.info("   inference-engine: {s}", .{ecosystem_versions.inference_engine});
    std.log.info("   model-server: {s}", .{ecosystem_versions.model_server});
    std.log.info("   platform: {s}", .{ecosystem_versions.platform});

    // Demonstrate different configuration presets
    std.log.info("", .{});
    std.log.info("🔧 Configuration Presets:", .{});
    
    // Development configuration
    {
        const dev_config = ai_platform.developmentPlatformConfig();
        std.log.info("   Development:", .{});
        std.log.info("     Environment: {s}", .{dev_config.environment.toString()});
        std.log.info("     Target: {s}", .{dev_config.deployment_target.toString()});
        std.log.info("     Memory Limit: {}MB", .{dev_config.max_memory_mb.?});
        std.log.info("     CPU Cores: {}", .{dev_config.max_cpu_cores.?});
        std.log.info("     Auto-scaling: {}", .{dev_config.enable_auto_scaling});
    }
    
    // IoT configuration
    {
        const iot_config = ai_platform.iotPlatformConfig();
        std.log.info("   IoT:", .{});
        std.log.info("     Environment: {s}", .{iot_config.environment.toString()});
        std.log.info("     Target: {s}", .{iot_config.deployment_target.toString()});
        std.log.info("     Memory Limit: {}MB", .{iot_config.max_memory_mb.?});
        std.log.info("     CPU Cores: {}", .{iot_config.max_cpu_cores.?});
        std.log.info("     GPU Enabled: {}", .{iot_config.enable_gpu});
    }
    
    // Production configuration
    {
        const prod_config = ai_platform.productionPlatformConfig();
        std.log.info("   Production:", .{});
        std.log.info("     Environment: {s}", .{prod_config.environment.toString()});
        std.log.info("     Target: {s}", .{prod_config.deployment_target.toString()});
        std.log.info("     Memory Limit: {}MB", .{prod_config.max_memory_mb.?});
        std.log.info("     CPU Cores: {}", .{prod_config.max_cpu_cores.?});
        std.log.info("     Auto-scaling: {}", .{prod_config.enable_auto_scaling});
    }

    // Create and initialize platform
    std.log.info("", .{});
    std.log.info("🚀 Initializing Platform...", .{});
    
    const platform_config = ai_platform.developmentPlatformConfig();
    var platform = try ai_platform.Platform.init(allocator, platform_config);
    defer platform.deinit();

    std.log.info("✅ Platform initialized successfully!", .{});

    // Display initial platform status
    const initial_stats = platform.getStatus();
    std.log.info("📊 Initial Platform Status:", .{});
    std.log.info("   Uptime: {} seconds", .{initial_stats.uptime_seconds});
    std.log.info("   Total Requests: {}", .{initial_stats.total_requests});
    std.log.info("   Active Models: {}", .{initial_stats.active_models});
    std.log.info("   Memory Usage: {d:.1}MB", .{initial_stats.memory_usage_mb});

    // Start platform services
    std.log.info("", .{});
    std.log.info("🔧 Starting Platform Services...", .{});
    
    try platform.start();
    
    std.log.info("✅ All services started successfully!", .{});

    // Display component information
    std.log.info("", .{});
    std.log.info("🧩 Component Status:", .{});
    
    const components = try platform.listComponents(allocator);
    defer allocator.free(components);
    
    for (components) |component| {
        std.log.info("   {s}: {s} (health: {d:.2})", .{
            component.name,
            component.status.toString(),
            component.health_score,
        });
    }

    // Demonstrate configuration management
    std.log.info("", .{});
    std.log.info("⚙️  Configuration Management:", .{});
    
    var config_manager = try ai_platform.ConfigManager.init(allocator, null);
    defer config_manager.deinit();
    
    // Validate different environment configurations
    const environments = [_]ai_platform.Environment{ .development, .testing, .staging, .production };
    
    for (environments) |env| {
        const env_config = try config_manager.getEnvironmentConfig(env);
        var validation_result = try config_manager.validateConfig(env_config);
        defer validation_result.deinit();
        
        std.log.info("   {s}: {} (errors: {}, warnings: {})", .{
            env.toString(),
            validation_result.valid,
            validation_result.errors.items.len,
            validation_result.warnings.items.len,
        });
    }

    // Demonstrate deployment planning
    std.log.info("", .{});
    std.log.info("🚀 Deployment Planning:", .{});
    
    const deployment_targets = [_]ai_platform.DeploymentTarget{ .iot, .desktop, .server, .cloud, .kubernetes };
    
    for (deployment_targets) |target| {
        var deployment_manager = try ai_platform.DeploymentManager.init(allocator, target);
        defer deployment_manager.deinit();
        
        const deploy_config = ai_platform.createDeploymentConfig(target, .production);
        var plan = try deployment_manager.createDeploymentPlan(deploy_config);
        defer plan.deinit();
        
        std.log.info("   {s}: {} steps, ~{}ms", .{
            target.toString(),
            plan.steps.items.len,
            plan.estimated_duration_ms,
        });
    }

    // Simulate some platform activity
    std.log.info("", .{});
    std.log.info("🔄 Simulating Platform Activity...", .{});
    
    for (0..5) |i| {
        std.time.sleep(500_000_000); // 500ms
        
        // Update platform statistics (simulated)
        const current_stats = platform.getStatus();
        std.log.info("   Cycle {}: uptime={}s, requests={}, memory={d:.1}MB", .{
            i + 1,
            current_stats.uptime_seconds,
            current_stats.total_requests,
            current_stats.memory_usage_mb,
        });
    }

    // Demonstrate health monitoring
    std.log.info("", .{});
    std.log.info("🏥 Health Monitoring:", .{});
    
    // Check overall platform health
    const overall_health = platform.getOverallHealth();
    std.log.info("   Overall Health: {s}", .{overall_health.toString()});
    
    // Check individual component health
    const component_names = [_][]const u8{ "tensor-core", "onnx-parser", "inference-engine", "model-server" };
    
    for (component_names) |name| {
        if (platform.getComponentInfo(name)) |component| {
            std.log.info("   {s}: {s} (score: {d:.2})", .{
                component.name,
                component.status.toString(),
                component.health_score,
            });
        }
    }

    // Final platform status
    std.log.info("", .{});
    std.log.info("📈 Final Platform Status:", .{});
    
    const final_stats = platform.getStatus();
    std.log.info("   Total Uptime: {} seconds", .{final_stats.uptime_seconds});
    std.log.info("   Total Requests: {}", .{final_stats.total_requests});
    std.log.info("   Total Inferences: {}", .{final_stats.total_inferences});
    std.log.info("   Active Models: {}", .{final_stats.active_models});
    std.log.info("   Memory Usage: {d:.1}MB", .{final_stats.memory_usage_mb});
    std.log.info("   CPU Usage: {d:.1}%", .{final_stats.cpu_usage_percent});
    std.log.info("   Error Count: {}", .{final_stats.error_count});

    // Demonstrate graceful shutdown
    std.log.info("", .{});
    std.log.info("🛑 Shutting Down Platform...", .{});
    
    platform.stop();
    
    std.log.info("✅ Platform shutdown completed successfully!", .{});
    std.log.info("", .{});
    std.log.info("🎯 Basic Platform Example Completed!", .{});
    std.log.info("   The complete Zig AI Ecosystem has been demonstrated:", .{});
    std.log.info("   - Platform orchestration and coordination", .{});
    std.log.info("   - Configuration management", .{});
    std.log.info("   - Component health monitoring", .{});
    std.log.info("   - Deployment planning", .{});
    std.log.info("   - Graceful startup and shutdown", .{});
}

/// Test function for the basic platform example
pub fn test_basic_platform() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test platform creation with different configurations
    {
        const dev_config = ai_platform.developmentPlatformConfig();
        var platform = try ai_platform.Platform.init(allocator, dev_config);
        defer platform.deinit();

        const stats = platform.getStatus();
        try std.testing.expect(stats.uptime_seconds >= 0);
        try std.testing.expect(stats.total_requests == 0);
    }

    {
        const iot_config = ai_platform.iotPlatformConfig();
        var platform = try ai_platform.Platform.init(allocator, iot_config);
        defer platform.deinit();

        try std.testing.expect(platform.config.deployment_target == .iot);
        try std.testing.expect(platform.config.max_memory_mb.? == 64);
        try std.testing.expect(platform.config.enable_gpu == false);
    }

    {
        const prod_config = ai_platform.productionPlatformConfig();
        var platform = try ai_platform.Platform.init(allocator, prod_config);
        defer platform.deinit();

        try std.testing.expect(platform.config.deployment_target == .server);
        try std.testing.expect(platform.config.enable_auto_scaling == true);
        try std.testing.expect(platform.config.max_memory_mb.? == 8192);
    }

    std.log.info("Basic platform test passed!", .{});
}

/// Demonstration of quick start functionality
pub fn demo_quick_start() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== Quick Start Demo ===", .{});

    // Quick start with default configuration
    {
        var platform = try ai_platform.quickStartDefault(allocator);
        defer platform.deinit();

        std.log.info("Default platform started successfully!", .{});
        
        const stats = platform.getStatus();
        std.log.info("   Uptime: {} seconds", .{stats.uptime_seconds});
        
        platform.stop();
    }

    // Quick start for IoT
    {
        var platform = try ai_platform.quickStartIoT(allocator);
        defer platform.deinit();

        std.log.info("IoT platform started successfully!", .{});
        
        try std.testing.expect(platform.config.deployment_target == .iot);
        try std.testing.expect(platform.config.max_memory_mb.? == 64);
        
        platform.stop();
    }

    // Quick start for production
    {
        var platform = try ai_platform.quickStartProduction(allocator);
        defer platform.deinit();

        std.log.info("Production platform started successfully!", .{});
        
        try std.testing.expect(platform.config.deployment_target == .server);
        try std.testing.expect(platform.config.enable_auto_scaling == true);
        
        platform.stop();
    }

    std.log.info("Quick start demo completed!", .{});
}

test "basic platform functionality" {
    try test_basic_platform();
}

test "quick start functionality" {
    try demo_quick_start();
}

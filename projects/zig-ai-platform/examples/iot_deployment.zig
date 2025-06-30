const std = @import("std");
const ai_platform = @import("zig-ai-platform");

/// IoT Deployment Example
/// 
/// This example demonstrates how to deploy the Zig AI Platform on IoT devices
/// such as Raspberry Pi, embedded systems, or edge computing devices.
/// 
/// Use Case: Smart factory sensors, autonomous vehicles, smart home devices
/// Requirements: 64MB RAM, single CPU core, no GPU
/// 
/// Features:
/// - Ultra-low memory footprint
/// - Optimized for single-core processing
/// - Minimal logging to preserve storage
/// - Local inference without cloud dependency
/// - Real-time edge AI processing

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🏠 IoT Edge AI Deployment Starting...", .{});

    // IoT-optimized configuration
    const iot_config = ai_platform.PlatformConfig{
        .environment = .production,
        .deployment_target = .iot,
        .enable_monitoring = true,
        .enable_logging = false,        // Minimal logging for storage
        .enable_metrics = false,        // Disable metrics to save memory
        .enable_auto_scaling = false,   // No auto-scaling on IoT
        .health_check_interval_ms = 60000, // Check every minute
        .log_level = .err,              // Only critical errors
        .max_memory_mb = 64,            // 64MB limit
        .max_cpu_cores = 1,             // Single core
        .enable_gpu = false,            // No GPU on most IoT devices
        .data_directory = "/opt/ai-platform/data",
        .log_directory = "/opt/ai-platform/logs",
    };

    // Initialize platform for IoT
    var platform = try ai_platform.Platform.init(allocator, iot_config);
    defer platform.deinit();

    std.log.info("✅ IoT Platform initialized", .{});
    std.log.info("   Memory limit: {}MB", .{iot_config.max_memory_mb.?});
    std.log.info("   CPU cores: {}", .{iot_config.max_cpu_cores.?});
    std.log.info("   GPU enabled: {}", .{iot_config.enable_gpu});

    // Start platform services
    try platform.start();
    std.log.info("🚀 IoT AI Platform running!", .{});

    // Simulate IoT sensor data processing
    std.log.info("📊 Processing sensor data...", .{});
    
    for (0..10) |i| {
        // Simulate sensor reading
        const sensor_value = @as(f32, @floatFromInt(i)) * 1.5 + 20.0; // Temperature sensor
        
        // Process with AI model (simulated)
        const prediction = processSensorData(sensor_value);
        
        std.log.info("Sensor {}: {d:.1}°C -> Prediction: {s}", .{
            i, sensor_value, if (prediction > 0.5) "ALERT" else "NORMAL"
        });
        
        // IoT devices typically process data continuously
        std.time.sleep(1_000_000_000); // 1 second interval
    }

    // Show final IoT statistics
    const stats = platform.getStatus();
    std.log.info("📈 IoT Session Complete:", .{});
    std.log.info("   Uptime: {} seconds", .{stats.uptime_seconds});
    std.log.info("   Memory usage: {d:.1}MB", .{stats.memory_usage_mb});
    std.log.info("   Total inferences: {}", .{stats.total_inferences});

    platform.stop();
    std.log.info("🏠 IoT AI Platform stopped", .{});
}

/// Simulate AI inference on sensor data
fn processSensorData(sensor_value: f32) f32 {
    // Simple threshold-based "AI" model for demonstration
    // In real use case, this would use the inference engine
    return if (sensor_value > 25.0) 0.8 else 0.2;
}

/// IoT deployment script generator
pub fn generateIoTDeployment() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("📜 Generating IoT deployment scripts...", .{});

    // Create deployment manager
    var deployment_manager = try ai_platform.DeploymentManager.init(allocator, .iot);
    defer deployment_manager.deinit();

    // Create IoT deployment configuration
    const deploy_config = ai_platform.DeploymentConfig{
        .target = .iot,
        .environment = .production,
        .enable_auto_start = true,
        .enable_health_checks = true,
        .enable_monitoring = false, // Minimal monitoring for IoT
        .replicas = 1,
        .resource_limits = .{
            .max_memory_mb = 64,
            .max_cpu_cores = 1,
            .max_disk_gb = 8,
        },
    };

    // Generate deployment scripts
    try deployment_manager.generateDeploymentScripts(deploy_config, "iot-deployment");
    
    std.log.info("✅ IoT deployment scripts generated in: iot-deployment/", .{});
    std.log.info("   Use these scripts to deploy on Raspberry Pi, embedded systems, etc.", .{});
}

test "IoT platform configuration" {
    const iot_config = ai_platform.iotPlatformConfig();
    
    // Verify IoT-specific settings
    try std.testing.expect(iot_config.deployment_target == .iot);
    try std.testing.expect(iot_config.max_memory_mb.? == 64);
    try std.testing.expect(iot_config.max_cpu_cores.? == 1);
    try std.testing.expect(iot_config.enable_gpu == false);
    try std.testing.expect(iot_config.enable_auto_scaling == false);
    try std.testing.expect(iot_config.enable_logging == false);
    try std.testing.expect(iot_config.enable_metrics == false);
}

/// Real-world IoT use cases
pub const IoTUseCases = struct {
    /// Smart Factory Sensor Monitoring
    pub fn smartFactory() void {
        std.log.info("🏭 Smart Factory Use Case:", .{});
        std.log.info("   - Real-time equipment monitoring", .{});
        std.log.info("   - Predictive maintenance alerts", .{});
        std.log.info("   - Quality control automation", .{});
        std.log.info("   - Energy consumption optimization", .{});
    }

    /// Autonomous Vehicle Edge Processing
    pub fn autonomousVehicle() void {
        std.log.info("🚗 Autonomous Vehicle Use Case:", .{});
        std.log.info("   - Real-time object detection", .{});
        std.log.info("   - Traffic sign recognition", .{});
        std.log.info("   - Lane departure warnings", .{});
        std.log.info("   - Emergency braking decisions", .{});
    }

    /// Smart Home Automation
    pub fn smartHome() void {
        std.log.info("🏡 Smart Home Use Case:", .{});
        std.log.info("   - Voice command processing", .{});
        std.log.info("   - Security camera analysis", .{});
        std.log.info("   - Energy usage optimization", .{});
        std.log.info("   - Appliance automation", .{});
    }

    /// Agricultural Monitoring
    pub fn agriculture() void {
        std.log.info("🌾 Agricultural Use Case:", .{});
        std.log.info("   - Crop health monitoring", .{});
        std.log.info("   - Soil condition analysis", .{});
        std.log.info("   - Irrigation optimization", .{});
        std.log.info("   - Pest detection alerts", .{});
    }
};

const std = @import("std");
const ai_platform = @import("zig-ai-platform");

/// Production Server Deployment Example
/// 
/// This example demonstrates how to deploy the Zig AI Platform in production
/// environments for high-scale AI model serving and inference.
/// 
/// Use Case: Enterprise AI APIs, ML model serving, high-throughput inference
/// Requirements: 8GB+ RAM, multi-core CPU, GPU acceleration
/// 
/// Features:
/// - Auto-scaling based on load
/// - High-availability deployment
/// - Comprehensive monitoring and metrics
/// - Load balancing across replicas
/// - Production-grade logging and alerting

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🏢 Production AI Platform Deployment Starting...", .{});

    // Production-optimized configuration
    const prod_config = ai_platform.PlatformConfig{
        .environment = .production,
        .deployment_target = .server,
        .enable_monitoring = true,
        .enable_logging = true,
        .enable_metrics = true,
        .enable_auto_scaling = true,
        .health_check_interval_ms = 15000, // Check every 15 seconds
        .log_level = .info,
        .metrics_port = 9090,
        .admin_port = 8081,
        .max_memory_mb = 8192,          // 8GB memory
        .max_cpu_cores = 16,            // 16 CPU cores
        .enable_gpu = true,             // GPU acceleration
        .data_directory = "/var/lib/ai-platform/data",
        .log_directory = "/var/log/ai-platform",
    };

    // Initialize production platform
    var platform = try ai_platform.Platform.init(allocator, prod_config);
    defer platform.deinit();

    std.log.info("✅ Production Platform initialized", .{});
    std.log.info("   Memory limit: {}MB", .{prod_config.max_memory_mb.?});
    std.log.info("   CPU cores: {}", .{prod_config.max_cpu_cores.?});
    std.log.info("   GPU enabled: {}", .{prod_config.enable_gpu});
    std.log.info("   Auto-scaling: {}", .{prod_config.enable_auto_scaling});

    // Start all production services
    try platform.start();
    std.log.info("🚀 Production AI Platform running!", .{});
    std.log.info("   Admin interface: http://localhost:{}", .{prod_config.admin_port});
    std.log.info("   Metrics endpoint: http://localhost:{}/metrics", .{prod_config.metrics_port});

    // Simulate production workload
    std.log.info("📊 Simulating production workload...", .{});
    
    // Simulate high-throughput inference requests
    for (0..100) |batch| {
        const batch_size = 32; // Process 32 requests per batch
        const start_time = std.time.nanoTimestamp();
        
        // Simulate batch inference
        for (0..batch_size) |i| {
            _ = processInferenceRequest(@intCast(batch * batch_size + i));
        }
        
        const end_time = std.time.nanoTimestamp();
        const batch_duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        
        std.log.info("Batch {}: {} requests processed in {d:.2}ms", .{
            batch, batch_size, batch_duration_ms
        });
        
        // Show platform statistics every 10 batches
        if (batch % 10 == 0) {
            const stats = platform.getStatus();
            std.log.info("📈 Platform Stats: uptime={}s, requests={}, memory={d:.1}MB, cpu={d:.1}%", .{
                stats.uptime_seconds,
                stats.total_requests,
                stats.memory_usage_mb,
                stats.cpu_usage_percent,
            });
        }
        
        // Small delay between batches
        std.time.sleep(100_000_000); // 100ms
    }

    // Generate production report
    try generateProductionReport(&platform, allocator);

    // Graceful shutdown
    std.log.info("🛑 Initiating graceful shutdown...", .{});
    platform.stop();
    std.log.info("✅ Production AI Platform stopped", .{});
}

/// Simulate AI inference request processing
fn processInferenceRequest(request_id: u32) f32 {
    // Simulate complex AI model inference
    // In real use case, this would use the inference engine with actual models
    const complexity = @as(f32, @floatFromInt(request_id % 10)) / 10.0;
    return complexity * 0.95 + 0.05; // Confidence score between 0.05 and 1.0
}

/// Generate comprehensive production report
fn generateProductionReport(platform: *ai_platform.Platform, allocator: std.mem.Allocator) !void {
    std.log.info("📄 Generating production report...", .{});
    
    const stats = platform.getStatus();
    const components = try platform.listComponents(allocator);
    defer allocator.free(components);
    
    std.log.info("=== Production Performance Report ===", .{});
    std.log.info("Total Uptime: {} seconds", .{stats.uptime_seconds});
    std.log.info("Total Requests: {}", .{stats.total_requests});
    std.log.info("Total Inferences: {}", .{stats.total_inferences});
    std.log.info("Active Models: {}", .{stats.active_models});
    std.log.info("Memory Usage: {d:.1}MB", .{stats.memory_usage_mb});
    std.log.info("CPU Usage: {d:.1}%", .{stats.cpu_usage_percent});
    std.log.info("GPU Usage: {d:.1}%", .{stats.gpu_usage_percent});
    std.log.info("Error Count: {}", .{stats.error_count});
    
    std.log.info("\nComponent Health:", .{});
    for (components) |component| {
        std.log.info("  {s}: {s} (health: {d:.2})", .{
            component.name,
            component.status.toString(),
            component.health_score,
        });
    }
    
    // Calculate performance metrics
    const requests_per_second = if (stats.uptime_seconds > 0) 
        @as(f64, @floatFromInt(stats.total_requests)) / @as(f64, @floatFromInt(stats.uptime_seconds))
    else 
        0.0;
    
    const inferences_per_second = if (stats.uptime_seconds > 0)
        @as(f64, @floatFromInt(stats.total_inferences)) / @as(f64, @floatFromInt(stats.uptime_seconds))
    else
        0.0;
    
    std.log.info("\nPerformance Metrics:", .{});
    std.log.info("  Requests/sec: {d:.2}", .{requests_per_second});
    std.log.info("  Inferences/sec: {d:.2}", .{inferences_per_second});
    std.log.info("  Error rate: {d:.3}%", .{
        if (stats.total_requests > 0)
            @as(f64, @floatFromInt(stats.error_count)) / @as(f64, @floatFromInt(stats.total_requests)) * 100.0
        else
            0.0
    });
}

/// Production deployment with multiple replicas
pub fn deployProductionCluster() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 Deploying Production Cluster...", .{});

    // Create deployment manager
    var deployment_manager = try ai_platform.DeploymentManager.init(allocator, .server);
    defer deployment_manager.deinit();

    // Production cluster configuration
    const deploy_config = ai_platform.DeploymentConfig{
        .target = .server,
        .environment = .production,
        .enable_auto_start = true,
        .enable_health_checks = true,
        .enable_monitoring = true,
        .replicas = 3, // 3 replicas for high availability
        .resource_limits = .{
            .max_memory_mb = 8192,
            .max_cpu_cores = 16,
            .max_disk_gb = 100,
        },
    };

    // Validate configuration
    try deployment_manager.validateConfig(deploy_config);

    // Create deployment plan
    var plan = try deployment_manager.createDeploymentPlan(deploy_config);
    defer plan.deinit();

    std.log.info("📋 Production Deployment Plan:", .{});
    for (plan.steps.items, 0..) |step, i| {
        std.log.info("   {}. {s}", .{ i + 1, step.description });
    }
    std.log.info("   Estimated duration: {}ms", .{plan.estimated_duration_ms});
    std.log.info("   Replicas: {}", .{deploy_config.replicas});

    // Execute deployment
    const result = try deployment_manager.deploy(plan, deploy_config);
    
    std.log.info("✅ Production cluster deployed!", .{});
    std.log.info("   Status: {s}", .{result.status.toString()});
    std.log.info("   Duration: {}s", .{result.getDuration()});
    std.log.info("   Steps completed: {}/{}", .{ result.steps_completed, result.total_steps });

    // Generate deployment scripts for infrastructure
    try deployment_manager.generateDeploymentScripts(deploy_config, "production-deployment");
    std.log.info("📜 Deployment scripts generated in: production-deployment/", .{});
}

test "production platform configuration" {
    const prod_config = ai_platform.productionPlatformConfig();
    
    // Verify production-specific settings
    try std.testing.expect(prod_config.deployment_target == .server);
    try std.testing.expect(prod_config.max_memory_mb.? == 8192);
    try std.testing.expect(prod_config.max_cpu_cores.? == 16);
    try std.testing.expect(prod_config.enable_gpu == true);
    try std.testing.expect(prod_config.enable_auto_scaling == true);
    try std.testing.expect(prod_config.enable_monitoring == true);
    try std.testing.expect(prod_config.enable_logging == true);
    try std.testing.expect(prod_config.enable_metrics == true);
}

/// Real-world production use cases
pub const ProductionUseCases = struct {
    /// Enterprise AI API Service
    pub fn enterpriseAPI() void {
        std.log.info("🏢 Enterprise AI API Use Case:", .{});
        std.log.info("   - High-throughput model serving", .{});
        std.log.info("   - Multi-tenant AI services", .{});
        std.log.info("   - SLA-guaranteed response times", .{});
        std.log.info("   - Enterprise security and compliance", .{});
    }

    /// E-commerce Recommendation Engine
    pub fn ecommerce() void {
        std.log.info("🛒 E-commerce Use Case:", .{});
        std.log.info("   - Real-time product recommendations", .{});
        std.log.info("   - Personalized search results", .{});
        std.log.info("   - Dynamic pricing optimization", .{});
        std.log.info("   - Fraud detection and prevention", .{});
    }

    /// Financial Trading Platform
    pub fn financialTrading() void {
        std.log.info("💰 Financial Trading Use Case:", .{});
        std.log.info("   - Real-time market analysis", .{});
        std.log.info("   - Algorithmic trading decisions", .{});
        std.log.info("   - Risk assessment and management", .{});
        std.log.info("   - Regulatory compliance monitoring", .{});
    }

    /// Healthcare AI Diagnostics
    pub fn healthcare() void {
        std.log.info("🏥 Healthcare Use Case:", .{});
        std.log.info("   - Medical image analysis", .{});
        std.log.info("   - Drug discovery acceleration", .{});
        std.log.info("   - Patient risk prediction", .{});
        std.log.info("   - Treatment recommendation systems", .{});
    }
};

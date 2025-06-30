const std = @import("std");
const ai_platform = @import("zig-ai-platform");

/// Comprehensive integration tests for the complete Zig AI Platform
/// These tests verify that all ecosystem components work together correctly

test "complete ecosystem integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize platform library
    ai_platform.init();
    defer ai_platform.deinit();

    // Test ecosystem information
    const ecosystem_info = ai_platform.getEcosystemInfo();
    try std.testing.expect(ecosystem_info.total_components == 5);
    try std.testing.expectEqualStrings("0.1.0", ecosystem_info.platform_version);

    // Test ecosystem versions
    const versions = ai_platform.getEcosystemVersions();
    try std.testing.expectEqualStrings("0.1.0", versions.tensor_core);
    try std.testing.expectEqualStrings("0.1.0", versions.onnx_parser);
    try std.testing.expectEqualStrings("0.1.0", versions.inference_engine);
    try std.testing.expectEqualStrings("0.1.0", versions.model_server);
    try std.testing.expectEqualStrings("0.1.0", versions.platform);
}

test "platform configuration variants" {
    // Test all configuration presets
    const default_config = ai_platform.defaultPlatformConfig();
    const iot_config = ai_platform.iotPlatformConfig();
    const prod_config = ai_platform.productionPlatformConfig();
    const dev_config = ai_platform.developmentPlatformConfig();

    // Validate default configuration
    try std.testing.expect(default_config.environment == .development);
    try std.testing.expect(default_config.deployment_target == .desktop);
    try std.testing.expect(default_config.enable_monitoring == true);

    // Validate IoT configuration
    try std.testing.expect(iot_config.environment == .production);
    try std.testing.expect(iot_config.deployment_target == .iot);
    try std.testing.expect(iot_config.max_memory_mb.? == 64);
    try std.testing.expect(iot_config.max_cpu_cores.? == 1);
    try std.testing.expect(iot_config.enable_gpu == false);
    try std.testing.expect(iot_config.enable_auto_scaling == false);

    // Validate production configuration
    try std.testing.expect(prod_config.environment == .production);
    try std.testing.expect(prod_config.deployment_target == .server);
    try std.testing.expect(prod_config.max_memory_mb.? == 8192);
    try std.testing.expect(prod_config.max_cpu_cores.? == 16);
    try std.testing.expect(prod_config.enable_gpu == true);
    try std.testing.expect(prod_config.enable_auto_scaling == true);

    // Validate development configuration
    try std.testing.expect(dev_config.environment == .development);
    try std.testing.expect(dev_config.deployment_target == .desktop);
    try std.testing.expect(dev_config.log_level == .debug);
}

test "platform lifecycle management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test platform creation and initialization
    var platform = try ai_platform.Platform.init(allocator, ai_platform.defaultPlatformConfig());
    defer platform.deinit();

    // Test initial state
    const initial_stats = platform.getStatus();
    try std.testing.expect(initial_stats.uptime_seconds >= 0);
    try std.testing.expect(initial_stats.total_requests == 0);
    try std.testing.expect(initial_stats.active_models == 0);

    // Test platform startup
    try platform.start();
    try std.testing.expect(platform.running == true);

    // Test component listing
    const components = try platform.listComponents(allocator);
    defer allocator.free(components);
    try std.testing.expect(components.len == 4); // tensor-core, onnx-parser, inference-engine, model-server

    // Verify all components are tracked
    const expected_components = [_][]const u8{ "tensor-core", "onnx-parser", "inference-engine", "model-server" };
    for (expected_components) |expected_name| {
        var found = false;
        for (components) |component| {
            if (std.mem.eql(u8, component.name, expected_name)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }

    // Test platform shutdown
    platform.stop();
    try std.testing.expect(platform.running == false);
}

test "configuration management integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test configuration manager initialization
    var config_manager = try ai_platform.ConfigManager.init(allocator, null);
    defer config_manager.deinit();

    // Test environment configuration retrieval
    const environments = [_]ai_platform.Environment{ .development, .testing, .staging, .production };
    
    for (environments) |env| {
        const env_config = try config_manager.getEnvironmentConfig(env);
        try std.testing.expect(env_config.environment == env);
        
        // Validate configuration
        var validation_result = try config_manager.validateConfig(env_config);
        defer validation_result.deinit();
        
        try std.testing.expect(validation_result.valid == true);
    }

    // Test configuration presets
    const iot_preset = ai_platform.ConfigPresets.iot();
    try std.testing.expect(iot_preset.deployment_target == .iot);
    try std.testing.expect(iot_preset.max_memory_mb.? == 64);

    const desktop_preset = ai_platform.ConfigPresets.desktop();
    try std.testing.expect(desktop_preset.deployment_target == .desktop);
    try std.testing.expect(desktop_preset.max_memory_mb.? == 2048);

    const production_preset = ai_platform.ConfigPresets.production();
    try std.testing.expect(production_preset.deployment_target == .server);
    try std.testing.expect(production_preset.enable_auto_scaling == true);
}

test "deployment management integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const deployment_targets = [_]ai_platform.DeploymentTarget{ .iot, .desktop, .server, .cloud, .kubernetes };
    
    for (deployment_targets) |target| {
        // Test deployment manager creation
        var deployment_manager = try ai_platform.DeploymentManager.init(allocator, target);
        defer deployment_manager.deinit();

        // Test deployment configuration creation
        const deploy_config = ai_platform.createDeploymentConfig(target, .production);
        try std.testing.expect(deploy_config.target == target);
        try std.testing.expect(deploy_config.environment == .production);

        // Test configuration validation
        try deployment_manager.validateConfig(deploy_config);

        // Test deployment plan creation
        var plan = try deployment_manager.createDeploymentPlan(deploy_config);
        defer plan.deinit();

        try std.testing.expect(plan.target == target);
        try std.testing.expect(plan.environment == .production);
        try std.testing.expect(plan.steps.items.len > 0);
        try std.testing.expect(plan.estimated_duration_ms > 0);

        // Verify target-specific step counts
        switch (target) {
            .iot => try std.testing.expect(plan.steps.items.len == 7),
            .desktop => try std.testing.expect(plan.steps.items.len == 7),
            .server => try std.testing.expect(plan.steps.items.len == 8),
            .cloud => try std.testing.expect(plan.steps.items.len == 8),
            .kubernetes => try std.testing.expect(plan.steps.items.len == 8),
        }
    }
}

test "health monitoring integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test health monitor initialization
    var health_monitor = try ai_platform.HealthMonitor.init(allocator, 1000); // 1 second interval
    defer health_monitor.deinit();

    // Test health monitor startup
    try health_monitor.start();
    try std.testing.expect(health_monitor.running == true);

    // Wait for at least one health check cycle
    std.time.sleep(1_500_000_000); // 1.5 seconds

    // Test health results
    const stats = health_monitor.getStats();
    try std.testing.expect(stats.total_checks > 0);

    // Test individual component health
    const component_health = health_monitor.getComponentHealth("system");
    try std.testing.expect(component_health != null);
    try std.testing.expect(component_health.?.status != .unknown);

    // Test overall health
    const overall_health = health_monitor.getOverallHealth();
    try std.testing.expect(overall_health != .unknown);

    // Test health report generation
    const health_report = try health_monitor.generateHealthReport(allocator);
    defer allocator.free(health_report);
    try std.testing.expect(health_report.len > 0);

    // Stop health monitor
    health_monitor.stop();
    try std.testing.expect(health_monitor.running == false);
}

test "logging integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create temporary log directory
    const log_dir = "test_logs";
    std.fs.cwd().makeDir(log_dir) catch |err| {
        if (err != error.PathAlreadyExists) {
            return err;
        }
    };
    defer std.fs.cwd().deleteTree(log_dir) catch {};

    // Test log aggregator initialization
    var log_aggregator = try ai_platform.LogAggregator.init(allocator, log_dir);
    defer log_aggregator.deinit();

    // Test log aggregator startup
    try log_aggregator.start();

    // Test logging functionality
    try log_aggregator.log(.info, "test-component", "Test log message");
    try log_aggregator.log(.warn, "test-component", "Test warning message");
    try log_aggregator.log(.err, "test-component", "Test error message");

    // Wait for logs to be processed
    std.time.sleep(100_000_000); // 100ms

    // Test log statistics
    const stats = log_aggregator.getStats();
    try std.testing.expect(stats.total_logs >= 3);
    try std.testing.expect(stats.logs_by_level[1] >= 1); // INFO
    try std.testing.expect(stats.logs_by_level[2] >= 1); // WARN
    try std.testing.expect(stats.logs_by_level[3] >= 1); // ERROR

    // Test log searching
    const search_results = try log_aggregator.searchLogs(allocator, "test-component", null, null);
    defer allocator.free(search_results);
    try std.testing.expect(search_results.len >= 3);

    // Stop log aggregator
    log_aggregator.stop();
}

test "metrics collection integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test metrics collector initialization
    var metrics_collector = try ai_platform.MetricsCollector.init(allocator, 9091);
    defer metrics_collector.deinit();

    // Test metrics collector startup
    try metrics_collector.start();

    // Test metric creation and manipulation
    const counter = try metrics_collector.getCounter("test_counter", "Test counter metric");
    counter.increment();
    counter.add(5.0);

    const gauge = try metrics_collector.getGauge("test_gauge", "Test gauge metric");
    gauge.set(42.0);
    gauge.add(8.0);

    const histogram = try metrics_collector.getHistogram("test_histogram", "Test histogram metric");
    try histogram.observe(1.5);
    try histogram.observe(2.3);
    try histogram.observe(0.8);

    // Wait for metrics collection
    std.time.sleep(100_000_000); // 100ms

    // Test metrics export
    const prometheus_output = try metrics_collector.exportPrometheusFormat(allocator);
    defer allocator.free(prometheus_output);
    try std.testing.expect(prometheus_output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, prometheus_output, "test_counter") != null);
    try std.testing.expect(std.mem.indexOf(u8, prometheus_output, "test_gauge") != null);
    try std.testing.expect(std.mem.indexOf(u8, prometheus_output, "test_histogram") != null);

    // Test metrics statistics
    const stats = metrics_collector.getStats();
    try std.testing.expect(stats.total_metrics > 0);

    // Stop metrics collector
    metrics_collector.stop();
}

test "CLI argument parsing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test basic command parsing
    {
        const args = [_][]const u8{ "zig-ai-platform", "init" };
        const parsed = try ai_platform.Args.parse(allocator, &args);
        try std.testing.expect(parsed.command == .init);
    }

    // Test command with environment and target
    {
        const args = [_][]const u8{ "zig-ai-platform", "deploy", "--env", "production", "--target", "server" };
        const parsed = try ai_platform.Args.parse(allocator, &args);
        try std.testing.expect(parsed.command == .deploy);
        try std.testing.expect(parsed.environment.? == .production);
        try std.testing.expect(parsed.deployment_target.? == .server);
    }

    // Test config subcommand
    {
        const args = [_][]const u8{ "zig-ai-platform", "config", "generate", "--env", "development" };
        const parsed = try ai_platform.Args.parse(allocator, &args);
        try std.testing.expect(parsed.command == .config);
        try std.testing.expect(parsed.subcommand.? == .generate);
        try std.testing.expect(parsed.environment.? == .development);
    }

    // Test flags
    {
        const args = [_][]const u8{ "zig-ai-platform", "start", "--verbose", "--force" };
        const parsed = try ai_platform.Args.parse(allocator, &args);
        try std.testing.expect(parsed.command == .start);
        try std.testing.expect(parsed.verbose == true);
        try std.testing.expect(parsed.force == true);
    }
}

test "target and environment validation" {
    // Test supported targets
    try std.testing.expect(ai_platform.isTargetSupported("iot"));
    try std.testing.expect(ai_platform.isTargetSupported("desktop"));
    try std.testing.expect(ai_platform.isTargetSupported("server"));
    try std.testing.expect(ai_platform.isTargetSupported("cloud"));
    try std.testing.expect(ai_platform.isTargetSupported("kubernetes"));
    try std.testing.expect(!ai_platform.isTargetSupported("invalid"));

    // Test supported environments
    try std.testing.expect(ai_platform.isEnvironmentSupported("development"));
    try std.testing.expect(ai_platform.isEnvironmentSupported("testing"));
    try std.testing.expect(ai_platform.isEnvironmentSupported("staging"));
    try std.testing.expect(ai_platform.isEnvironmentSupported("production"));
    try std.testing.expect(!ai_platform.isEnvironmentSupported("invalid"));

    // Test target and environment lists
    const targets = ai_platform.getSupportedTargets();
    try std.testing.expect(targets.len == 5);

    const environments = ai_platform.getSupportedEnvironments();
    try std.testing.expect(environments.len == 4);
}

test "configuration validation" {
    // Test valid configuration
    var valid_config = ai_platform.defaultPlatformConfig();
    try ai_platform.validatePlatformConfig(valid_config);

    // Test invalid memory configuration
    var invalid_memory_config = ai_platform.defaultPlatformConfig();
    invalid_memory_config.max_memory_mb = 32;
    try std.testing.expectError(error.InsufficientMemory, ai_platform.validatePlatformConfig(invalid_memory_config));

    // Test invalid CPU configuration
    var invalid_cpu_config = ai_platform.defaultPlatformConfig();
    invalid_cpu_config.max_cpu_cores = 0;
    try std.testing.expectError(error.InvalidCpuCores, ai_platform.validatePlatformConfig(invalid_cpu_config));

    // Test port conflict
    var port_conflict_config = ai_platform.defaultPlatformConfig();
    port_conflict_config.admin_port = port_conflict_config.metrics_port;
    try std.testing.expectError(error.PortConflict, ai_platform.validatePlatformConfig(port_conflict_config));

    // Test invalid health check interval
    var invalid_interval_config = ai_platform.defaultPlatformConfig();
    invalid_interval_config.health_check_interval_ms = 500;
    try std.testing.expectError(error.InvalidHealthCheckInterval, ai_platform.validatePlatformConfig(invalid_interval_config));
}

test "quick start functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test default quick start
    {
        var platform = try ai_platform.quickStartDefault(allocator);
        defer platform.deinit();
        
        try std.testing.expect(platform.running == true);
        try std.testing.expect(platform.config.environment == .development);
        
        platform.stop();
    }

    // Test IoT quick start
    {
        var platform = try ai_platform.quickStartIoT(allocator);
        defer platform.deinit();
        
        try std.testing.expect(platform.running == true);
        try std.testing.expect(platform.config.deployment_target == .iot);
        try std.testing.expect(platform.config.max_memory_mb.? == 64);
        
        platform.stop();
    }

    // Test production quick start
    {
        var platform = try ai_platform.quickStartProduction(allocator);
        defer platform.deinit();
        
        try std.testing.expect(platform.running == true);
        try std.testing.expect(platform.config.deployment_target == .server);
        try std.testing.expect(platform.config.enable_auto_scaling == true);
        
        platform.stop();
    }
}

test "end-to-end platform workflow" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 1. Initialize platform with production configuration
    var platform = try ai_platform.Platform.init(allocator, ai_platform.productionPlatformConfig());
    defer platform.deinit();

    // 2. Start all services
    try platform.start();
    try std.testing.expect(platform.running == true);

    // 3. Verify all components are running
    const components = try platform.listComponents(allocator);
    defer allocator.free(components);
    try std.testing.expect(components.len == 4);

    // 4. Check overall health
    const overall_health = platform.getOverallHealth();
    try std.testing.expect(overall_health != .unknown);

    // 5. Simulate some activity
    std.time.sleep(100_000_000); // 100ms

    // 6. Check final statistics
    const final_stats = platform.getStatus();
    try std.testing.expect(final_stats.uptime_seconds >= 0);

    // 7. Graceful shutdown
    platform.stop();
    try std.testing.expect(platform.running == false);
}

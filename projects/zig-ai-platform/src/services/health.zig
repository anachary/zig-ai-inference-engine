const std = @import("std");
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;

/// Health check status
pub const HealthStatus = enum {
    healthy,
    degraded,
    unhealthy,
    unknown,
    
    pub fn toString(self: HealthStatus) []const u8 {
        return switch (self) {
            .healthy => "healthy",
            .degraded => "degraded",
            .unhealthy => "unhealthy",
            .unknown => "unknown",
        };
    }
    
    pub fn fromScore(score: f32) HealthStatus {
        if (score >= 0.9) return .healthy;
        if (score >= 0.7) return .degraded;
        if (score >= 0.3) return .unhealthy;
        return .unknown;
    }
};

/// Health check result
pub const HealthCheckResult = struct {
    component_name: []const u8,
    status: HealthStatus,
    score: f32, // 0.0 to 1.0
    timestamp: i64,
    response_time_ms: f32,
    error_message: ?[]const u8,
    details: std.StringHashMap([]const u8),
    
    pub fn init(allocator: Allocator, component_name: []const u8) HealthCheckResult {
        return HealthCheckResult{
            .component_name = component_name,
            .status = .unknown,
            .score = 0.0,
            .timestamp = std.time.timestamp(),
            .response_time_ms = 0.0,
            .error_message = null,
            .details = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *HealthCheckResult) void {
        self.details.deinit();
        if (self.error_message) |msg| {
            // Note: In real implementation, we'd need the allocator to free this
            _ = msg;
        }
    }
    
    pub fn setDetail(self: *HealthCheckResult, key: []const u8, value: []const u8) !void {
        try self.details.put(key, value);
    }
};

/// Health check function signature
pub const HealthCheckFn = *const fn (allocator: Allocator) anyerror!HealthCheckResult;

/// Health check configuration
pub const HealthCheckConfig = struct {
    name: []const u8,
    check_fn: HealthCheckFn,
    interval_ms: u32,
    timeout_ms: u32,
    enabled: bool,
    
    pub fn init(name: []const u8, check_fn: HealthCheckFn) HealthCheckConfig {
        return HealthCheckConfig{
            .name = name,
            .check_fn = check_fn,
            .interval_ms = 30000, // 30 seconds
            .timeout_ms = 5000,   // 5 seconds
            .enabled = true,
        };
    }
};

/// Health monitor statistics
pub const HealthMonitorStats = struct {
    total_checks: u64 = 0,
    successful_checks: u64 = 0,
    failed_checks: u64 = 0,
    average_response_time_ms: f32 = 0.0,
    uptime_seconds: u64 = 0,
    start_time: i64 = 0,
};

/// Health monitor for platform components
pub const HealthMonitor = struct {
    allocator: Allocator,
    check_interval_ms: u32,
    health_checks: std.ArrayList(HealthCheckConfig),
    health_results: std.StringHashMap(HealthCheckResult),
    stats: HealthMonitorStats,
    
    // Threading
    running: bool,
    worker_thread: ?Thread,
    mutex: Mutex,
    
    const Self = @This();

    /// Initialize health monitor
    pub fn init(allocator: Allocator, check_interval_ms: u32) !Self {
        var self = Self{
            .allocator = allocator,
            .check_interval_ms = check_interval_ms,
            .health_checks = std.ArrayList(HealthCheckConfig).init(allocator),
            .health_results = std.StringHashMap(HealthCheckResult).init(allocator),
            .stats = HealthMonitorStats{
                .start_time = std.time.timestamp(),
            },
            .running = false,
            .worker_thread = null,
            .mutex = Mutex{},
        };
        
        // Register default health checks
        try self.registerDefaultHealthChecks();
        
        return self;
    }

    /// Deinitialize health monitor
    pub fn deinit(self: *Self) void {
        self.stop();
        
        if (self.worker_thread) |thread| {
            thread.join();
        }
        
        self.mutex.lock();
        
        // Clean up health results
        var result_iter = self.health_results.iterator();
        while (result_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.health_results.deinit();
        
        self.health_checks.deinit();
        self.mutex.unlock();
    }

    /// Start health monitoring
    pub fn start(self: *Self) !void {
        if (self.running) {
            return;
        }
        
        self.running = true;
        self.worker_thread = try Thread.spawn(.{}, monitorWorker, .{self});
        
        std.log.info("🏥 Health monitor started with {} checks", .{self.health_checks.items.len});
    }

    /// Stop health monitoring
    pub fn stop(self: *Self) void {
        if (!self.running) {
            return;
        }
        
        self.running = false;
        std.log.info("🏥 Health monitor stopped", .{});
    }

    /// Register a health check
    pub fn registerHealthCheck(self: *Self, config: HealthCheckConfig) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.health_checks.append(config);
        std.log.info("Registered health check: {s}", .{config.name});
    }

    /// Get health status for a component
    pub fn getComponentHealth(self: *const Self, component_name: []const u8) ?HealthCheckResult {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        return self.health_results.get(component_name);
    }

    /// Get overall platform health
    pub fn getOverallHealth(self: *const Self) HealthStatus {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.health_results.count() == 0) {
            return .unknown;
        }
        
        var total_score: f32 = 0.0;
        var count: u32 = 0;
        
        var result_iter = self.health_results.iterator();
        while (result_iter.next()) |entry| {
            total_score += entry.value_ptr.score;
            count += 1;
        }
        
        const average_score = total_score / @as(f32, @floatFromInt(count));
        return HealthStatus.fromScore(average_score);
    }

    /// Get all health results
    pub fn getAllHealthResults(self: *const Self, allocator: Allocator) ![]HealthCheckResult {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var results = std.ArrayList(HealthCheckResult).init(allocator);
        
        var result_iter = self.health_results.iterator();
        while (result_iter.next()) |entry| {
            try results.append(entry.value_ptr.*);
        }
        
        return results.toOwnedSlice();
    }

    /// Get health monitor statistics
    pub fn getStats(self: *const Self) HealthMonitorStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var stats = self.stats;
        stats.uptime_seconds = @intCast(std.time.timestamp() - stats.start_time);
        
        // Calculate average response time
        if (stats.total_checks > 0) {
            var total_response_time: f32 = 0.0;
            var result_iter = self.health_results.iterator();
            while (result_iter.next()) |entry| {
                total_response_time += entry.value_ptr.response_time_ms;
            }
            stats.average_response_time_ms = total_response_time / @as(f32, @floatFromInt(self.health_results.count()));
        }
        
        return stats;
    }

    /// Perform immediate health check for all components
    pub fn performHealthCheck(self: *Self) !void {
        for (self.health_checks.items) |check_config| {
            if (!check_config.enabled) continue;
            
            const start_time = std.time.nanoTimestamp();
            
            var result = check_config.check_fn(self.allocator) catch |err| {
                var error_result = HealthCheckResult.init(self.allocator, check_config.name);
                error_result.status = .unhealthy;
                error_result.score = 0.0;
                error_result.error_message = @errorName(err);
                error_result;
            };
            
            const end_time = std.time.nanoTimestamp();
            result.response_time_ms = @as(f32, @floatFromInt(end_time - start_time)) / 1_000_000.0;
            result.timestamp = std.time.timestamp();
            
            // Update statistics
            self.mutex.lock();
            self.stats.total_checks += 1;
            if (result.status == .healthy or result.status == .degraded) {
                self.stats.successful_checks += 1;
            } else {
                self.stats.failed_checks += 1;
            }
            
            // Store result
            self.health_results.put(check_config.name, result) catch |err| {
                std.log.err("Failed to store health check result for {s}: {}", .{ check_config.name, err });
            };
            self.mutex.unlock();
        }
    }

    /// Generate health report
    pub fn generateHealthReport(self: *const Self, allocator: Allocator) ![]u8 {
        const overall_health = self.getOverallHealth();
        const stats = self.getStats();
        
        var report = std.ArrayList(u8).init(allocator);
        const writer = report.writer();
        
        try writer.print("=== Zig AI Platform Health Report ===\n");
        try writer.print("Generated: {}\n", .{std.time.timestamp()});
        try writer.print("Overall Status: {s}\n", .{overall_health.toString()});
        try writer.print("Uptime: {} seconds\n", .{stats.uptime_seconds});
        try writer.print("Total Checks: {}\n", .{stats.total_checks});
        try writer.print("Success Rate: {d:.1}%\n", .{
            if (stats.total_checks > 0) 
                @as(f32, @floatFromInt(stats.successful_checks)) / @as(f32, @floatFromInt(stats.total_checks)) * 100.0 
            else 
                0.0
        });
        try writer.print("Average Response Time: {d:.2}ms\n\n", .{stats.average_response_time_ms});
        
        // Component details
        try writer.print("Component Health:\n");
        
        self.mutex.lock();
        var result_iter = self.health_results.iterator();
        while (result_iter.next()) |entry| {
            const result = entry.value_ptr;
            try writer.print("  {s}: {s} (score: {d:.2}, response: {d:.2}ms)\n", .{
                result.component_name,
                result.status.toString(),
                result.score,
                result.response_time_ms,
            });
            
            if (result.error_message) |error_msg| {
                try writer.print("    Error: {s}\n", .{error_msg});
            }
        }
        self.mutex.unlock();
        
        return report.toOwnedSlice();
    }

    // Private methods

    /// Register default health checks
    fn registerDefaultHealthChecks(self: *Self) !void {
        // System health check
        try self.registerHealthCheck(HealthCheckConfig.init("system", systemHealthCheck));
        
        // Memory health check
        try self.registerHealthCheck(HealthCheckConfig.init("memory", memoryHealthCheck));
        
        // Disk health check
        try self.registerHealthCheck(HealthCheckConfig.init("disk", diskHealthCheck));
        
        // Network health check
        try self.registerHealthCheck(HealthCheckConfig.init("network", networkHealthCheck));
    }

    /// Monitor worker thread
    fn monitorWorker(self: *Self) void {
        while (self.running) {
            self.performHealthCheck() catch |err| {
                std.log.err("Health check failed: {}", .{err});
            };
            
            // Sleep for check interval
            std.time.sleep(self.check_interval_ms * 1_000_000);
        }
    }
};

// Default health check implementations

/// System health check
fn systemHealthCheck(allocator: Allocator) !HealthCheckResult {
    var result = HealthCheckResult.init(allocator, "system");
    
    // Check basic system health
    result.status = .healthy;
    result.score = 1.0;
    
    try result.setDetail("cpu_count", "4"); // Placeholder
    try result.setDetail("load_average", "0.5"); // Placeholder
    
    return result;
}

/// Memory health check
fn memoryHealthCheck(allocator: Allocator) !HealthCheckResult {
    var result = HealthCheckResult.init(allocator, "memory");
    
    // Check memory usage (simplified)
    const memory_usage_percent: f32 = 45.0; // Placeholder
    
    if (memory_usage_percent < 80.0) {
        result.status = .healthy;
        result.score = 1.0 - (memory_usage_percent / 100.0) * 0.5;
    } else if (memory_usage_percent < 90.0) {
        result.status = .degraded;
        result.score = 0.5;
    } else {
        result.status = .unhealthy;
        result.score = 0.2;
    }
    
    const usage_str = try std.fmt.allocPrint(allocator, "{d:.1}%", .{memory_usage_percent});
    try result.setDetail("usage_percent", usage_str);
    
    return result;
}

/// Disk health check
fn diskHealthCheck(allocator: Allocator) !HealthCheckResult {
    var result = HealthCheckResult.init(allocator, "disk");
    
    // Check disk space (simplified)
    const disk_usage_percent: f32 = 60.0; // Placeholder
    
    if (disk_usage_percent < 85.0) {
        result.status = .healthy;
        result.score = 1.0 - (disk_usage_percent / 100.0) * 0.3;
    } else if (disk_usage_percent < 95.0) {
        result.status = .degraded;
        result.score = 0.4;
    } else {
        result.status = .unhealthy;
        result.score = 0.1;
    }
    
    const usage_str = try std.fmt.allocPrint(allocator, "{d:.1}%", .{disk_usage_percent});
    try result.setDetail("usage_percent", usage_str);
    
    return result;
}

/// Network health check
fn networkHealthCheck(allocator: Allocator) !HealthCheckResult {
    var result = HealthCheckResult.init(allocator, "network");
    
    // Check network connectivity (simplified)
    result.status = .healthy;
    result.score = 0.95;
    
    try result.setDetail("connectivity", "good");
    try result.setDetail("latency_ms", "15.2");
    
    return result;
}

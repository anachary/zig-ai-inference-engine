const std = @import("std");
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const net = std.net;

/// Metric type enumeration
pub const MetricType = enum {
    counter,
    gauge,
    histogram,
    summary,
    
    pub fn toString(self: MetricType) []const u8 {
        return switch (self) {
            .counter => "counter",
            .gauge => "gauge",
            .histogram => "histogram",
            .summary => "summary",
        };
    }
};

/// Metric value
pub const MetricValue = union(MetricType) {
    counter: u64,
    gauge: f64,
    histogram: HistogramValue,
    summary: SummaryValue,
    
    pub fn asFloat(self: MetricValue) f64 {
        return switch (self) {
            .counter => |val| @floatFromInt(val),
            .gauge => |val| val,
            .histogram => |val| val.sum,
            .summary => |val| val.sum,
        };
    }
};

/// Histogram value structure
pub const HistogramValue = struct {
    count: u64,
    sum: f64,
    buckets: std.ArrayList(HistogramBucket),
    
    pub fn init(allocator: Allocator) HistogramValue {
        return HistogramValue{
            .count = 0,
            .sum = 0.0,
            .buckets = std.ArrayList(HistogramBucket).init(allocator),
        };
    }
    
    pub fn deinit(self: *HistogramValue) void {
        self.buckets.deinit();
    }
    
    pub fn observe(self: *HistogramValue, value: f64) !void {
        self.count += 1;
        self.sum += value;
        
        // Update buckets
        for (self.buckets.items) |*bucket| {
            if (value <= bucket.upper_bound) {
                bucket.count += 1;
            }
        }
    }
};

/// Histogram bucket
pub const HistogramBucket = struct {
    upper_bound: f64,
    count: u64,
};

/// Summary value structure
pub const SummaryValue = struct {
    count: u64,
    sum: f64,
    quantiles: std.ArrayList(Quantile),
    
    pub fn init(allocator: Allocator) SummaryValue {
        return SummaryValue{
            .count = 0,
            .sum = 0.0,
            .quantiles = std.ArrayList(Quantile).init(allocator),
        };
    }
    
    pub fn deinit(self: *SummaryValue) void {
        self.quantiles.deinit();
    }
};

/// Quantile structure
pub const Quantile = struct {
    quantile: f64, // 0.0 to 1.0
    value: f64,
};

/// Metric labels
pub const MetricLabels = std.StringHashMap([]const u8);

/// Metric definition
pub const Metric = struct {
    name: []const u8,
    help: []const u8,
    metric_type: MetricType,
    value: MetricValue,
    labels: MetricLabels,
    timestamp: i64,
    
    pub fn init(allocator: Allocator, name: []const u8, help: []const u8, metric_type: MetricType) Metric {
        const value = switch (metric_type) {
            .counter => MetricValue{ .counter = 0 },
            .gauge => MetricValue{ .gauge = 0.0 },
            .histogram => MetricValue{ .histogram = HistogramValue.init(allocator) },
            .summary => MetricValue{ .summary = SummaryValue.init(allocator) },
        };
        
        return Metric{
            .name = name,
            .help = help,
            .metric_type = metric_type,
            .value = value,
            .labels = MetricLabels.init(allocator),
            .timestamp = std.time.timestamp(),
        };
    }
    
    pub fn deinit(self: *Metric) void {
        switch (self.value) {
            .histogram => |*hist| hist.deinit(),
            .summary => |*summ| summ.deinit(),
            else => {},
        }
        self.labels.deinit();
    }
    
    pub fn addLabel(self: *Metric, key: []const u8, value: []const u8) !void {
        try self.labels.put(key, value);
    }
    
    pub fn increment(self: *Metric) void {
        switch (self.value) {
            .counter => |*counter| counter.* += 1,
            else => {},
        }
        self.timestamp = std.time.timestamp();
    }
    
    pub fn add(self: *Metric, amount: f64) void {
        switch (self.value) {
            .counter => |*counter| counter.* += @intFromFloat(amount),
            .gauge => |*gauge| gauge.* += amount,
            else => {},
        }
        self.timestamp = std.time.timestamp();
    }
    
    pub fn set(self: *Metric, value: f64) void {
        switch (self.value) {
            .gauge => |*gauge| gauge.* = value,
            else => {},
        }
        self.timestamp = std.time.timestamp();
    }
    
    pub fn observe(self: *Metric, value: f64) !void {
        switch (self.value) {
            .histogram => |*hist| try hist.observe(value),
            .summary => |*summ| {
                summ.count += 1;
                summ.sum += value;
            },
            else => {},
        }
        self.timestamp = std.time.timestamp();
    }
};

/// Metrics collector configuration
pub const MetricsCollectorConfig = struct {
    port: u16 = 9090,
    path: []const u8 = "/metrics",
    enable_http_server: bool = true,
    collection_interval_ms: u32 = 15000,
    retention_period_hours: u32 = 24,
    max_metrics: usize = 10000,
};

/// Metrics collector statistics
pub const MetricsCollectorStats = struct {
    total_metrics: u64 = 0,
    http_requests: u64 = 0,
    collection_cycles: u64 = 0,
    bytes_served: u64 = 0,
    start_time: i64 = 0,
};

/// Metrics collector for Prometheus-style metrics
pub const MetricsCollector = struct {
    allocator: Allocator,
    config: MetricsCollectorConfig,
    metrics: std.StringHashMap(Metric),
    stats: MetricsCollectorStats,
    
    // HTTP server
    http_server: ?net.Server,
    server_thread: ?Thread,
    
    // Collection
    running: bool,
    collector_thread: ?Thread,
    metrics_mutex: Mutex,
    
    const Self = @This();

    /// Initialize metrics collector
    pub fn init(allocator: Allocator, port: u16) !Self {
        const config = MetricsCollectorConfig{
            .port = port,
        };
        
        var self = Self{
            .allocator = allocator,
            .config = config,
            .metrics = std.StringHashMap(Metric).init(allocator),
            .stats = MetricsCollectorStats{
                .start_time = std.time.timestamp(),
            },
            .http_server = null,
            .server_thread = null,
            .running = false,
            .collector_thread = null,
            .metrics_mutex = Mutex{},
        };
        
        // Register default platform metrics
        try self.registerDefaultMetrics();
        
        return self;
    }

    /// Deinitialize metrics collector
    pub fn deinit(self: *Self) void {
        self.stop();
        
        if (self.server_thread) |thread| {
            thread.join();
        }
        if (self.collector_thread) |thread| {
            thread.join();
        }
        
        // Clean up metrics
        self.metrics_mutex.lock();
        var metric_iter = self.metrics.iterator();
        while (metric_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.metrics.deinit();
        self.metrics_mutex.unlock();
    }

    /// Start metrics collector
    pub fn start(self: *Self) !void {
        if (self.running) {
            return;
        }
        
        self.running = true;
        
        // Start HTTP server if enabled
        if (self.config.enable_http_server) {
            try self.startHttpServer();
        }
        
        // Start collection thread
        self.collector_thread = try Thread.spawn(.{}, collectionWorker, .{self});
        
        std.log.info("📊 Metrics collector started on port {}", .{self.config.port});
    }

    /// Stop metrics collector
    pub fn stop(self: *Self) void {
        if (!self.running) {
            return;
        }
        
        self.running = false;
        
        // Stop HTTP server
        if (self.http_server) |*server| {
            server.deinit();
            self.http_server = null;
        }
        
        std.log.info("📊 Metrics collector stopped", .{});
    }

    /// Register a metric
    pub fn registerMetric(self: *Self, metric: Metric) !void {
        self.metrics_mutex.lock();
        defer self.metrics_mutex.unlock();
        
        try self.metrics.put(metric.name, metric);
        self.stats.total_metrics += 1;
    }

    /// Get or create a counter metric
    pub fn getCounter(self: *Self, name: []const u8, help: []const u8) !*Metric {
        self.metrics_mutex.lock();
        defer self.metrics_mutex.unlock();
        
        if (self.metrics.getPtr(name)) |metric| {
            return metric;
        }
        
        const metric = Metric.init(self.allocator, name, help, .counter);
        try self.metrics.put(name, metric);
        return self.metrics.getPtr(name).?;
    }

    /// Get or create a gauge metric
    pub fn getGauge(self: *Self, name: []const u8, help: []const u8) !*Metric {
        self.metrics_mutex.lock();
        defer self.metrics_mutex.unlock();
        
        if (self.metrics.getPtr(name)) |metric| {
            return metric;
        }
        
        const metric = Metric.init(self.allocator, name, help, .gauge);
        try self.metrics.put(name, metric);
        return self.metrics.getPtr(name).?;
    }

    /// Get or create a histogram metric
    pub fn getHistogram(self: *Self, name: []const u8, help: []const u8) !*Metric {
        self.metrics_mutex.lock();
        defer self.metrics_mutex.unlock();
        
        if (self.metrics.getPtr(name)) |metric| {
            return metric;
        }
        
        const metric = Metric.init(self.allocator, name, help, .histogram);
        try self.metrics.put(name, metric);
        return self.metrics.getPtr(name).?;
    }

    /// Export metrics in Prometheus format
    pub fn exportPrometheusFormat(self: *const Self, allocator: Allocator) ![]u8 {
        self.metrics_mutex.lock();
        defer self.metrics_mutex.unlock();
        
        var output = std.ArrayList(u8).init(allocator);
        const writer = output.writer();
        
        var metric_iter = self.metrics.iterator();
        while (metric_iter.next()) |entry| {
            const metric = entry.value_ptr;
            
            // Write help comment
            try writer.print("# HELP {s} {s}\n", .{ metric.name, metric.help });
            
            // Write type comment
            try writer.print("# TYPE {s} {s}\n", .{ metric.name, metric.metric_type.toString() });
            
            // Write metric value
            try writer.print("{s}", .{metric.name});
            
            // Write labels
            if (metric.labels.count() > 0) {
                try writer.print("{");
                var label_iter = metric.labels.iterator();
                var first = true;
                while (label_iter.next()) |label_entry| {
                    if (!first) try writer.print(",");
                    try writer.print("{s}=\"{s}\"", .{ label_entry.key_ptr.*, label_entry.value_ptr.* });
                    first = false;
                }
                try writer.print("}");
            }
            
            // Write value
            try writer.print(" {d} {}\n", .{ metric.value.asFloat(), metric.timestamp * 1000 });
            
            // Handle histogram buckets
            if (metric.value == .histogram) {
                const hist = metric.value.histogram;
                for (hist.buckets.items) |bucket| {
                    try writer.print("{s}_bucket{{le=\"{d}\"}} {} {}\n", .{
                        metric.name,
                        bucket.upper_bound,
                        bucket.count,
                        metric.timestamp * 1000,
                    });
                }
                try writer.print("{s}_count {} {}\n", .{ metric.name, hist.count, metric.timestamp * 1000 });
                try writer.print("{s}_sum {d} {}\n", .{ metric.name, hist.sum, metric.timestamp * 1000 });
            }
            
            try writer.print("\n");
        }
        
        return output.toOwnedSlice();
    }

    /// Get collector statistics
    pub fn getStats(self: *const Self) MetricsCollectorStats {
        return self.stats;
    }

    /// Collect system metrics
    pub fn collectSystemMetrics(self: *Self) !void {
        // CPU usage
        const cpu_gauge = try self.getGauge("platform_cpu_usage_percent", "CPU usage percentage");
        cpu_gauge.set(45.0); // Placeholder
        
        // Memory usage
        const memory_gauge = try self.getGauge("platform_memory_usage_bytes", "Memory usage in bytes");
        memory_gauge.set(1024 * 1024 * 512); // Placeholder: 512MB
        
        // Disk usage
        const disk_gauge = try self.getGauge("platform_disk_usage_percent", "Disk usage percentage");
        disk_gauge.set(60.0); // Placeholder
        
        // Network bytes
        const network_counter = try self.getCounter("platform_network_bytes_total", "Total network bytes");
        network_counter.add(1024); // Placeholder
        
        // Update collection cycle counter
        const collection_counter = try self.getCounter("platform_collection_cycles_total", "Total collection cycles");
        collection_counter.increment();
        
        self.stats.collection_cycles += 1;
    }

    // Private methods

    /// Register default platform metrics
    fn registerDefaultMetrics(self: *Self) !void {
        // Platform uptime
        var uptime_gauge = Metric.init(self.allocator, "platform_uptime_seconds", "Platform uptime in seconds", .gauge);
        try self.registerMetric(uptime_gauge);
        
        // Component health scores
        var health_gauge = Metric.init(self.allocator, "platform_component_health_score", "Component health score (0-1)", .gauge);
        try self.registerMetric(health_gauge);
        
        // Request latency histogram
        var latency_hist = Metric.init(self.allocator, "platform_request_duration_seconds", "Request duration in seconds", .histogram);
        try self.registerMetric(latency_hist);
        
        // Error counter
        var error_counter = Metric.init(self.allocator, "platform_errors_total", "Total platform errors", .counter);
        try self.registerMetric(error_counter);
    }

    /// Start HTTP server for metrics endpoint
    fn startHttpServer(self: *Self) !void {
        const address = net.Address.parseIp("0.0.0.0", self.config.port) catch |err| {
            std.log.err("Failed to parse metrics server address: {}", .{err});
            return err;
        };
        
        self.http_server = net.Address.listen(address, .{}) catch |err| {
            std.log.err("Failed to start metrics HTTP server: {}", .{err});
            return err;
        };
        
        self.server_thread = try Thread.spawn(.{}, httpServerWorker, .{self});
    }

    /// HTTP server worker thread
    fn httpServerWorker(self: *Self) void {
        while (self.running) {
            if (self.http_server) |*server| {
                const connection = server.accept() catch |err| {
                    if (err == error.SocketNotListening) break;
                    std.log.warn("Failed to accept metrics connection: {}", .{err});
                    continue;
                };
                
                self.handleHttpRequest(connection) catch |err| {
                    std.log.warn("Failed to handle metrics request: {}", .{err});
                };
                
                connection.stream.close();
            }
        }
    }

    /// Handle HTTP request for metrics
    fn handleHttpRequest(self: *Self, connection: net.Server.Connection) !void {
        var buffer: [1024]u8 = undefined;
        const bytes_read = try connection.stream.read(&buffer);
        const request = buffer[0..bytes_read];
        
        // Simple HTTP request parsing
        if (std.mem.indexOf(u8, request, "GET /metrics")) |_| {
            const metrics_data = try self.exportPrometheusFormat(self.allocator);
            defer self.allocator.free(metrics_data);
            
            const response = try std.fmt.allocPrint(self.allocator,
                "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n" ++
                "Content-Length: {}\r\n" ++
                "\r\n" ++
                "{s}",
                .{ metrics_data.len, metrics_data }
            );
            defer self.allocator.free(response);
            
            _ = try connection.stream.writeAll(response);
            
            self.stats.http_requests += 1;
            self.stats.bytes_served += response.len;
        } else {
            const not_found = "HTTP/1.1 404 Not Found\r\n\r\n";
            _ = try connection.stream.writeAll(not_found);
        }
    }

    /// Collection worker thread
    fn collectionWorker(self: *Self) void {
        while (self.running) {
            self.collectSystemMetrics() catch |err| {
                std.log.err("Failed to collect system metrics: {}", .{err});
            };
            
            // Sleep for collection interval
            std.time.sleep(self.config.collection_interval_ms * 1_000_000);
        }
    }
};

/// Global metrics collector instance
pub var global_metrics_collector: ?*MetricsCollector = null;

/// Initialize global metrics collector
pub fn initGlobalMetricsCollector(allocator: Allocator, port: u16) !void {
    global_metrics_collector = try allocator.create(MetricsCollector);
    global_metrics_collector.?.* = try MetricsCollector.init(allocator, port);
    try global_metrics_collector.?.start();
}

/// Get global metrics collector
pub fn getGlobalMetricsCollector() ?*MetricsCollector {
    return global_metrics_collector;
}

/// Cleanup global metrics collector
pub fn deinitGlobalMetricsCollector(allocator: Allocator) void {
    if (global_metrics_collector) |collector| {
        collector.deinit();
        allocator.destroy(collector);
        global_metrics_collector = null;
    }
}

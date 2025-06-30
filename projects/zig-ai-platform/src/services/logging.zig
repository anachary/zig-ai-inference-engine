const std = @import("std");
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;

/// Log level enumeration
pub const LogLevel = enum(u8) {
    debug = 0,
    info = 1,
    warn = 2,
    err = 3,
    
    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
        };
    }
    
    pub fn fromString(level_str: []const u8) ?LogLevel {
        if (std.mem.eql(u8, level_str, "DEBUG")) return .debug;
        if (std.mem.eql(u8, level_str, "INFO")) return .info;
        if (std.mem.eql(u8, level_str, "WARN")) return .warn;
        if (std.mem.eql(u8, level_str, "ERROR")) return .err;
        return null;
    }
};

/// Log entry structure
pub const LogEntry = struct {
    timestamp: i64,
    level: LogLevel,
    component: []const u8,
    message: []const u8,
    context: std.StringHashMap([]const u8),
    
    pub fn init(allocator: Allocator, level: LogLevel, component: []const u8, message: []const u8) LogEntry {
        return LogEntry{
            .timestamp = std.time.timestamp(),
            .level = level,
            .component = component,
            .message = message,
            .context = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *LogEntry) void {
        self.context.deinit();
    }
    
    pub fn addContext(self: *LogEntry, key: []const u8, value: []const u8) !void {
        try self.context.put(key, value);
    }
    
    pub fn format(self: *const LogEntry, allocator: Allocator) ![]u8 {
        const timestamp_str = try formatTimestamp(allocator, self.timestamp);
        defer allocator.free(timestamp_str);
        
        var formatted = std.ArrayList(u8).init(allocator);
        const writer = formatted.writer();
        
        try writer.print("[{s}] {s} [{s}] {s}", .{
            timestamp_str,
            self.level.toString(),
            self.component,
            self.message,
        });
        
        // Add context if present
        if (self.context.count() > 0) {
            try writer.print(" {");
            var iter = self.context.iterator();
            var first = true;
            while (iter.next()) |entry| {
                if (!first) try writer.print(", ");
                try writer.print("{s}={s}", .{ entry.key_ptr.*, entry.value_ptr.* });
                first = false;
            }
            try writer.print("}");
        }
        
        try writer.print("\n");
        return formatted.toOwnedSlice();
    }
};

/// Log aggregator configuration
pub const LogAggregatorConfig = struct {
    log_directory: []const u8,
    max_file_size_mb: usize = 100,
    max_files: usize = 10,
    buffer_size: usize = 1000,
    flush_interval_ms: u32 = 5000,
    min_log_level: LogLevel = .info,
    enable_console_output: bool = true,
    enable_file_output: bool = true,
    enable_structured_logging: bool = false,
};

/// Log aggregator statistics
pub const LogAggregatorStats = struct {
    total_logs: u64 = 0,
    logs_by_level: [4]u64 = [_]u64{0} ** 4,
    logs_by_component: std.StringHashMap(u64),
    bytes_written: u64 = 0,
    files_rotated: u32 = 0,
    buffer_overflows: u32 = 0,
    start_time: i64 = 0,
    
    pub fn init(allocator: Allocator) LogAggregatorStats {
        return LogAggregatorStats{
            .logs_by_component = std.StringHashMap(u64).init(allocator),
            .start_time = std.time.timestamp(),
        };
    }
    
    pub fn deinit(self: *LogAggregatorStats) void {
        self.logs_by_component.deinit();
    }
};

/// Log aggregator for centralized logging
pub const LogAggregator = struct {
    allocator: Allocator,
    config: LogAggregatorConfig,
    log_buffer: std.ArrayList(LogEntry),
    current_log_file: ?std.fs.File,
    current_file_size: usize,
    stats: LogAggregatorStats,
    
    // Threading
    running: bool,
    worker_thread: ?Thread,
    buffer_mutex: Mutex,
    
    const Self = @This();

    /// Initialize log aggregator
    pub fn init(allocator: Allocator, log_directory: []const u8) !Self {
        // Create log directory if it doesn't exist
        std.fs.cwd().makeDir(log_directory) catch |err| {
            if (err != error.PathAlreadyExists) {
                return err;
            }
        };
        
        const config = LogAggregatorConfig{
            .log_directory = log_directory,
        };
        
        var self = Self{
            .allocator = allocator,
            .config = config,
            .log_buffer = std.ArrayList(LogEntry).init(allocator),
            .current_log_file = null,
            .current_file_size = 0,
            .stats = LogAggregatorStats.init(allocator),
            .running = false,
            .worker_thread = null,
            .buffer_mutex = Mutex{},
        };
        
        // Open initial log file
        try self.rotateLogFile();
        
        return self;
    }

    /// Deinitialize log aggregator
    pub fn deinit(self: *Self) void {
        self.stop();
        
        if (self.worker_thread) |thread| {
            thread.join();
        }
        
        // Flush remaining logs
        self.flushLogs() catch {};
        
        // Clean up
        if (self.current_log_file) |file| {
            file.close();
        }
        
        self.buffer_mutex.lock();
        for (self.log_buffer.items) |*entry| {
            entry.deinit();
        }
        self.log_buffer.deinit();
        self.buffer_mutex.unlock();
        
        self.stats.deinit();
    }

    /// Start log aggregator
    pub fn start(self: *Self) !void {
        if (self.running) {
            return;
        }
        
        self.running = true;
        self.worker_thread = try Thread.spawn(.{}, logWorker, .{self});
        
        std.log.info("📝 Log aggregator started, writing to: {s}", .{self.config.log_directory});
    }

    /// Stop log aggregator
    pub fn stop(self: *Self) void {
        if (!self.running) {
            return;
        }
        
        self.running = false;
        std.log.info("📝 Log aggregator stopped", .{});
    }

    /// Log a message
    pub fn log(self: *Self, level: LogLevel, component: []const u8, message: []const u8) !void {
        if (@intFromEnum(level) < @intFromEnum(self.config.min_log_level)) {
            return; // Skip logs below minimum level
        }
        
        var entry = LogEntry.init(self.allocator, level, component, message);
        
        // Add to buffer
        self.buffer_mutex.lock();
        defer self.buffer_mutex.unlock();
        
        if (self.log_buffer.items.len >= self.config.buffer_size) {
            // Buffer overflow - remove oldest entry
            var oldest = self.log_buffer.orderedRemove(0);
            oldest.deinit();
            self.stats.buffer_overflows += 1;
        }
        
        try self.log_buffer.append(entry);
        
        // Update statistics
        self.stats.total_logs += 1;
        self.stats.logs_by_level[@intFromEnum(level)] += 1;
        
        const component_count = self.stats.logs_by_component.get(component) orelse 0;
        try self.stats.logs_by_component.put(component, component_count + 1);
        
        // Console output if enabled
        if (self.config.enable_console_output) {
            const formatted = try entry.format(self.allocator);
            defer self.allocator.free(formatted);
            std.debug.print("{s}", .{formatted});
        }
    }

    /// Log with context
    pub fn logWithContext(
        self: *Self,
        level: LogLevel,
        component: []const u8,
        message: []const u8,
        context: std.StringHashMap([]const u8),
    ) !void {
        if (@intFromEnum(level) < @intFromEnum(self.config.min_log_level)) {
            return;
        }
        
        var entry = LogEntry.init(self.allocator, level, component, message);
        
        // Copy context
        var context_iter = context.iterator();
        while (context_iter.next()) |ctx_entry| {
            try entry.addContext(ctx_entry.key_ptr.*, ctx_entry.value_ptr.*);
        }
        
        // Add to buffer (similar to regular log)
        self.buffer_mutex.lock();
        defer self.buffer_mutex.unlock();
        
        if (self.log_buffer.items.len >= self.config.buffer_size) {
            var oldest = self.log_buffer.orderedRemove(0);
            oldest.deinit();
            self.stats.buffer_overflows += 1;
        }
        
        try self.log_buffer.append(entry);
        
        // Update statistics
        self.stats.total_logs += 1;
        self.stats.logs_by_level[@intFromEnum(level)] += 1;
        
        const component_count = self.stats.logs_by_component.get(component) orelse 0;
        try self.stats.logs_by_component.put(component, component_count + 1);
    }

    /// Get aggregator statistics
    pub fn getStats(self: *const Self) LogAggregatorStats {
        return self.stats;
    }

    /// Search logs by criteria
    pub fn searchLogs(
        self: *const Self,
        allocator: Allocator,
        component: ?[]const u8,
        level: ?LogLevel,
        since: ?i64,
    ) ![]LogEntry {
        self.buffer_mutex.lock();
        defer self.buffer_mutex.unlock();
        
        var matching_logs = std.ArrayList(LogEntry).init(allocator);
        
        for (self.log_buffer.items) |entry| {
            // Filter by component
            if (component) |comp| {
                if (!std.mem.eql(u8, entry.component, comp)) {
                    continue;
                }
            }
            
            // Filter by level
            if (level) |lvl| {
                if (entry.level != lvl) {
                    continue;
                }
            }
            
            // Filter by timestamp
            if (since) |timestamp| {
                if (entry.timestamp < timestamp) {
                    continue;
                }
            }
            
            try matching_logs.append(entry);
        }
        
        return matching_logs.toOwnedSlice();
    }

    /// Generate log report
    pub fn generateReport(self: *const Self, allocator: Allocator) ![]u8 {
        var report = std.ArrayList(u8).init(allocator);
        const writer = report.writer();
        
        try writer.print("=== Log Aggregator Report ===\n");
        try writer.print("Total Logs: {}\n", .{self.stats.total_logs});
        try writer.print("Buffer Overflows: {}\n", .{self.stats.buffer_overflows});
        try writer.print("Files Rotated: {}\n", .{self.stats.files_rotated});
        try writer.print("Bytes Written: {}\n", .{self.stats.bytes_written});
        
        try writer.print("\nLogs by Level:\n");
        const level_names = [_][]const u8{ "DEBUG", "INFO", "WARN", "ERROR" };
        for (level_names, 0..) |name, i| {
            try writer.print("  {s}: {}\n", .{ name, self.stats.logs_by_level[i] });
        }
        
        try writer.print("\nTop Components:\n");
        var component_iter = self.stats.logs_by_component.iterator();
        while (component_iter.next()) |entry| {
            try writer.print("  {s}: {}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }
        
        return report.toOwnedSlice();
    }

    // Private methods

    /// Flush logs to file
    fn flushLogs(self: *Self) !void {
        if (!self.config.enable_file_output) {
            return;
        }
        
        self.buffer_mutex.lock();
        defer self.buffer_mutex.unlock();
        
        if (self.current_log_file == null) {
            return;
        }
        
        const file = self.current_log_file.?;
        
        for (self.log_buffer.items) |*entry| {
            const formatted = try entry.format(self.allocator);
            defer self.allocator.free(formatted);
            
            try file.writeAll(formatted);
            self.current_file_size += formatted.len;
            self.stats.bytes_written += formatted.len;
            
            // Check if file rotation is needed
            if (self.current_file_size >= self.config.max_file_size_mb * 1024 * 1024) {
                try self.rotateLogFile();
            }
        }
        
        // Clear buffer after flushing
        for (self.log_buffer.items) |*entry| {
            entry.deinit();
        }
        self.log_buffer.clearRetainingCapacity();
        
        try file.sync();
    }

    /// Rotate log file
    fn rotateLogFile(self: *Self) !void {
        // Close current file
        if (self.current_log_file) |file| {
            file.close();
        }
        
        // Generate new filename
        const timestamp = std.time.timestamp();
        const filename = try std.fmt.allocPrint(
            self.allocator,
            "{s}/platform-{}.log",
            .{ self.config.log_directory, timestamp }
        );
        defer self.allocator.free(filename);
        
        // Open new file
        self.current_log_file = try std.fs.cwd().createFile(filename, .{});
        self.current_file_size = 0;
        self.stats.files_rotated += 1;
        
        std.log.info("Log file rotated: {s}", .{filename});
        
        // TODO: Clean up old log files based on max_files setting
    }

    /// Log worker thread
    fn logWorker(self: *Self) void {
        while (self.running) {
            self.flushLogs() catch |err| {
                std.log.err("Failed to flush logs: {}", .{err});
            };
            
            // Sleep for flush interval
            std.time.sleep(self.config.flush_interval_ms * 1_000_000);
        }
        
        // Final flush on shutdown
        self.flushLogs() catch {};
    }
};

/// Format timestamp for logging
fn formatTimestamp(allocator: Allocator, timestamp: i64) ![]u8 {
    // Simple timestamp formatting - in production, use proper date/time formatting
    return std.fmt.allocPrint(allocator, "{}", .{timestamp});
}

/// Global log aggregator instance
pub var global_log_aggregator: ?*LogAggregator = null;

/// Initialize global log aggregator
pub fn initGlobalLogAggregator(allocator: Allocator, log_directory: []const u8) !void {
    global_log_aggregator = try allocator.create(LogAggregator);
    global_log_aggregator.?.* = try LogAggregator.init(allocator, log_directory);
    try global_log_aggregator.?.start();
}

/// Get global log aggregator
pub fn getGlobalLogAggregator() ?*LogAggregator {
    return global_log_aggregator;
}

/// Cleanup global log aggregator
pub fn deinitGlobalLogAggregator(allocator: Allocator) void {
    if (global_log_aggregator) |aggregator| {
        aggregator.deinit();
        allocator.destroy(aggregator);
        global_log_aggregator = null;
    }
}

/// Convenience logging functions
pub fn logDebug(component: []const u8, message: []const u8) void {
    if (global_log_aggregator) |aggregator| {
        aggregator.log(.debug, component, message) catch {};
    }
}

pub fn logInfo(component: []const u8, message: []const u8) void {
    if (global_log_aggregator) |aggregator| {
        aggregator.log(.info, component, message) catch {};
    }
}

pub fn logWarn(component: []const u8, message: []const u8) void {
    if (global_log_aggregator) |aggregator| {
        aggregator.log(.warn, component, message) catch {};
    }
}

pub fn logError(component: []const u8, message: []const u8) void {
    if (global_log_aggregator) |aggregator| {
        aggregator.log(.err, component, message) catch {};
    }
}

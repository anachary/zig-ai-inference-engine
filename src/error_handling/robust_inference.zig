const std = @import("std");

/// Comprehensive error handling for production deployment
pub const ErrorHandler = struct {
    allocator: std.mem.Allocator,
    error_log: std.ArrayList(ErrorRecord),
    recovery_strategies: std.HashMap(ErrorType, RecoveryStrategy, std.hash_map.AutoContext(ErrorType), std.hash_map.default_max_load_percentage),
    
    pub fn init(allocator: std.mem.Allocator) ErrorHandler {
        var handler = ErrorHandler{
            .allocator = allocator,
            .error_log = std.ArrayList(ErrorRecord).init(allocator),
            .recovery_strategies = std.HashMap(ErrorType, RecoveryStrategy, std.hash_map.AutoContext(ErrorType), std.hash_map.default_max_load_percentage).init(allocator),
        };
        
        // Initialize default recovery strategies
        handler.initDefaultStrategies() catch {};
        
        return handler;
    }
    
    pub fn deinit(self: *ErrorHandler) void {
        for (self.error_log.items) |*record| {
            self.allocator.free(record.message);
            if (record.context) |ctx| {
                self.allocator.free(ctx);
            }
        }
        self.error_log.deinit();
        self.recovery_strategies.deinit();
    }
    
    fn initDefaultStrategies(self: *ErrorHandler) !void {
        try self.recovery_strategies.put(.ModelLoadError, .RetryWithFallback);
        try self.recovery_strategies.put(.MemoryError, .ReduceMemoryUsage);
        try self.recovery_strategies.put(.ComputationError, .RetryWithSimplification);
        try self.recovery_strategies.put(.InputValidationError, .SanitizeInput);
        try self.recovery_strategies.put(.NetworkError, .RetryWithBackoff);
    }
    
    /// Handle an error with automatic recovery
    pub fn handleError(
        self: *ErrorHandler,
        error_type: ErrorType,
        error_msg: []const u8,
        context: ?[]const u8,
    ) !RecoveryAction {
        // Log the error
        try self.logError(error_type, error_msg, context);
        
        // Get recovery strategy
        const strategy = self.recovery_strategies.get(error_type) orelse .LogAndContinue;
        
        // Execute recovery
        return self.executeRecovery(strategy, error_type, context);
    }
    
    fn logError(self: *ErrorHandler, error_type: ErrorType, message: []const u8, context: ?[]const u8) !void {
        const record = ErrorRecord{
            .timestamp = std.time.timestamp(),
            .error_type = error_type,
            .message = try self.allocator.dupe(u8, message),
            .context = if (context) |ctx| try self.allocator.dupe(u8, ctx) else null,
        };
        
        try self.error_log.append(record);
        
        // Print error for immediate visibility
        std.debug.print("[ERROR] {s}: {s}\n", .{ @tagName(error_type), message });
        if (context) |ctx| {
            std.debug.print("        Context: {s}\n", .{ctx});
        }
    }
    
    fn executeRecovery(self: *ErrorHandler, strategy: RecoveryStrategy, error_type: ErrorType, context: ?[]const u8) RecoveryAction {
        _ = self;
        _ = context;
        
        return switch (strategy) {
            .RetryWithFallback => .{ .action = .Retry, .fallback_available = true },
            .ReduceMemoryUsage => .{ .action = .ReduceMemory, .fallback_available = false },
            .RetryWithSimplification => .{ .action = .Simplify, .fallback_available = true },
            .SanitizeInput => .{ .action = .SanitizeInput, .fallback_available = false },
            .RetryWithBackoff => .{ .action = .RetryWithDelay, .fallback_available = true },
            .LogAndContinue => .{ .action = .Continue, .fallback_available = false },
            .FailFast => blk: {
                std.debug.print("CRITICAL ERROR: {s} - Failing fast\n", .{@tagName(error_type)});
                break :blk .{ .action = .Abort, .fallback_available = false };
            },
        };
    }
    
    /// Generate error report for debugging
    pub fn generateErrorReport(self: *ErrorHandler) void {
        std.debug.print("\nERROR HANDLING REPORT\n", .{});
        std.debug.print("=====================\n", .{});
        std.debug.print("Total Errors: {}\n\n", .{self.error_log.items.len});
        
        if (self.error_log.items.len == 0) {
            std.debug.print("No errors recorded - system running smoothly!\n\n", .{});
            return;
        }
        
        // Group errors by type
        var error_counts = std.HashMap(ErrorType, u32, std.hash_map.AutoContext(ErrorType), std.hash_map.default_max_load_percentage).init(self.allocator);
        defer error_counts.deinit();
        
        for (self.error_log.items) |record| {
            const count = error_counts.get(record.error_type) orelse 0;
            error_counts.put(record.error_type, count + 1) catch {};
        }
        
        std.debug.print("Error Summary by Type:\n", .{});
        var iterator = error_counts.iterator();
        while (iterator.next()) |entry| {
            std.debug.print("  {s}: {} occurrences\n", .{ @tagName(entry.key_ptr.*), entry.value_ptr.* });
        }
        
        std.debug.print("\nRecent Errors (last 5):\n", .{});
        const start_idx = if (self.error_log.items.len > 5) self.error_log.items.len - 5 else 0;
        for (self.error_log.items[start_idx..]) |record| {
            std.debug.print("  [{d}] {s}: {s}\n", .{ record.timestamp, @tagName(record.error_type), record.message });
        }
        std.debug.print("\n", .{});
    }
};

pub const ErrorType = enum {
    ModelLoadError,
    MemoryError,
    ComputationError,
    InputValidationError,
    NetworkError,
    FileSystemError,
    QuantizationError,
    TokenizationError,
    InferenceError,
    ConfigurationError,
};

pub const RecoveryStrategy = enum {
    RetryWithFallback,
    ReduceMemoryUsage,
    RetryWithSimplification,
    SanitizeInput,
    RetryWithBackoff,
    LogAndContinue,
    FailFast,
};

pub const RecoveryAction = struct {
    action: ActionType,
    fallback_available: bool,
    
    pub const ActionType = enum {
        Retry,
        ReduceMemory,
        Simplify,
        SanitizeInput,
        RetryWithDelay,
        Continue,
        Abort,
    };
};

pub const ErrorRecord = struct {
    timestamp: i64,
    error_type: ErrorType,
    message: []u8,
    context: ?[]u8,
};

/// Input validation and sanitization
pub const InputValidator = struct {
    pub fn validateModelPath(path: []const u8) !void {
        if (path.len == 0) {
            return error.EmptyPath;
        }
        
        if (path.len > 4096) {
            return error.PathTooLong;
        }
        
        // Check for valid file extensions
        const valid_extensions = [_][]const u8{ ".gguf", ".onnx", ".pt", ".pth", ".bin" };
        var has_valid_extension = false;
        for (valid_extensions) |ext| {
            if (std.mem.endsWith(u8, path, ext)) {
                has_valid_extension = true;
                break;
            }
        }
        
        if (!has_valid_extension) {
            return error.InvalidFileExtension;
        }
    }
    
    pub fn validateTokenSequence(tokens: []const u32, max_length: usize) !void {
        if (tokens.len == 0) {
            return error.EmptyTokenSequence;
        }
        
        if (tokens.len > max_length) {
            return error.TokenSequenceTooLong;
        }
        
        // Check for invalid token IDs (assuming vocab size limit)
        const max_token_id = 200000; // Reasonable upper bound
        for (tokens) |token| {
            if (token >= max_token_id) {
                return error.InvalidTokenId;
            }
        }
    }
    
    pub fn validateGenerationParams(max_tokens: usize, temperature: f32) !void {
        if (max_tokens == 0 or max_tokens > 32768) {
            return error.InvalidMaxTokens;
        }
        
        if (temperature < 0.0 or temperature > 2.0) {
            return error.InvalidTemperature;
        }
    }
};

/// Memory management and monitoring
pub const MemoryMonitor = struct {
    peak_usage: usize = 0,
    current_usage: usize = 0,
    allocation_count: usize = 0,
    
    pub fn recordAllocation(self: *MemoryMonitor, size: usize) void {
        self.current_usage += size;
        self.allocation_count += 1;
        self.peak_usage = @max(self.peak_usage, self.current_usage);
    }
    
    pub fn recordDeallocation(self: *MemoryMonitor, size: usize) void {
        self.current_usage = if (self.current_usage >= size) self.current_usage - size else 0;
    }
    
    pub fn checkMemoryPressure(self: *MemoryMonitor, limit_mb: usize) bool {
        const current_mb = self.current_usage / (1024 * 1024);
        return current_mb > limit_mb;
    }
    
    pub fn generateReport(self: *MemoryMonitor) void {
        const current_mb = @as(f64, @floatFromInt(self.current_usage)) / (1024.0 * 1024.0);
        const peak_mb = @as(f64, @floatFromInt(self.peak_usage)) / (1024.0 * 1024.0);
        
        std.debug.print("Memory Usage Report:\n", .{});
        std.debug.print("  Current: {d:.2} MB\n", .{current_mb});
        std.debug.print("  Peak: {d:.2} MB\n", .{peak_mb});
        std.debug.print("  Allocations: {}\n", .{self.allocation_count});
    }
};

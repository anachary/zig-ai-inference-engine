const std = @import("std");
const Allocator = std.mem.Allocator;
const Request = @import("request.zig").Request;
const Response = @import("response.zig").Response;

/// Middleware function signature
pub const MiddlewareFn = *const fn (request: *Request) anyerror!void;

/// Middleware wrapper
pub const Middleware = struct {
    name: []const u8,
    process_fn: MiddlewareFn,
    
    const Self = @This();

    /// Initialize middleware
    pub fn init(name: []const u8, process_fn: MiddlewareFn) Self {
        return Self{
            .name = name,
            .process_fn = process_fn,
        };
    }

    /// Process request through middleware
    pub fn process(self: *const Self, request: *Request) !void {
        try self.process_fn(request);
    }

    /// CORS middleware
    pub fn cors() Self {
        return Self.init("CORS", corsMiddleware);
    }

    /// Logging middleware
    pub fn logging() Self {
        return Self.init("Logging", loggingMiddleware);
    }

    /// Request size limit middleware
    pub fn requestSizeLimit(max_size: usize) Self {
        // Note: In a real implementation, we'd need to store the max_size
        // For now, we'll use a fixed limit in the middleware function
        _ = max_size;
        return Self.init("RequestSizeLimit", requestSizeLimitMiddleware);
    }

    /// Authentication middleware
    pub fn auth() Self {
        return Self.init("Auth", authMiddleware);
    }

    /// Rate limiting middleware
    pub fn rateLimit() Self {
        return Self.init("RateLimit", rateLimitMiddleware);
    }

    /// Request timeout middleware
    pub fn timeout(timeout_ms: u32) Self {
        _ = timeout_ms;
        return Self.init("Timeout", timeoutMiddleware);
    }

    /// Compression middleware
    pub fn compression() Self {
        return Self.init("Compression", compressionMiddleware);
    }

    /// Security headers middleware
    pub fn security() Self {
        return Self.init("Security", securityMiddleware);
    }

    /// Request ID middleware
    pub fn requestId() Self {
        return Self.init("RequestId", requestIdMiddleware);
    }

    /// Metrics collection middleware
    pub fn metrics() Self {
        return Self.init("Metrics", metricsMiddleware);
    }
};

/// CORS middleware implementation
fn corsMiddleware(request: *Request) !void {
    // Add CORS headers to request context
    // In a real implementation, we'd modify the response
    _ = request;
    
    // For now, just log that CORS middleware was applied
    std.log.debug("CORS middleware applied", .{});
}

/// Logging middleware implementation
fn loggingMiddleware(request: *Request) !void {
    const timestamp = std.time.timestamp();
    const method = request.method.toString();
    const path = request.path;
    const user_agent = request.getUserAgent() orelse "Unknown";
    const client_ip = request.getClientIP() orelse "Unknown";
    
    std.log.info("[{}] {} {} - {} - {s}", .{ timestamp, method, path, client_ip, user_agent });
}

/// Request size limit middleware implementation
fn requestSizeLimitMiddleware(request: *Request) !void {
    const max_size: usize = 10 * 1024 * 1024; // 10MB
    
    if (request.body) |body| {
        if (body.len > max_size) {
            std.log.warn("Request body too large: {} bytes (max: {} bytes)", .{ body.len, max_size });
            return error.PayloadTooLarge;
        }
    }
    
    // Check Content-Length header
    if (request.getHeader("Content-Length")) |content_length_str| {
        const content_length = std.fmt.parseInt(usize, content_length_str, 10) catch 0;
        if (content_length > max_size) {
            std.log.warn("Content-Length too large: {} bytes (max: {} bytes)", .{ content_length, max_size });
            return error.PayloadTooLarge;
        }
    }
}

/// Authentication middleware implementation
fn authMiddleware(request: *Request) !void {
    // Check for Authorization header
    const auth_header = request.getHeader("Authorization");
    
    if (auth_header == null) {
        // For now, we'll allow requests without authentication
        // In production, you'd return an error here
        std.log.debug("No authorization header found", .{});
        return;
    }
    
    const auth = auth_header.?;
    
    // Check for Bearer token
    if (std.mem.startsWith(u8, auth, "Bearer ")) {
        const token = auth[7..]; // Skip "Bearer "
        
        // Validate token (simplified)
        if (token.len < 10) {
            std.log.warn("Invalid token: too short", .{});
            return error.Unauthorized;
        }
        
        std.log.debug("Valid bearer token found", .{});
    } else if (std.mem.startsWith(u8, auth, "Basic ")) {
        // Handle Basic authentication
        const credentials = auth[6..]; // Skip "Basic "
        
        // Decode base64 credentials (simplified)
        if (credentials.len < 4) {
            std.log.warn("Invalid basic auth: too short", .{});
            return error.Unauthorized;
        }
        
        std.log.debug("Basic auth credentials found", .{});
    } else {
        std.log.warn("Unsupported authentication scheme", .{});
        return error.Unauthorized;
    }
}

/// Rate limiting middleware implementation
fn rateLimitMiddleware(request: *Request) !void {
    // Simple rate limiting based on client IP
    const client_ip = request.getClientIP() orelse "unknown";
    
    // In a real implementation, you'd track requests per IP
    // For now, just log the rate limit check
    std.log.debug("Rate limit check for IP: {s}", .{client_ip});
    
    // Simulate rate limit exceeded (for demonstration)
    const current_time = std.time.timestamp();
    if (current_time % 100 == 0) { // Randomly trigger rate limit
        std.log.warn("Rate limit exceeded for IP: {s}", .{client_ip});
        return error.TooManyRequests;
    }
}

/// Request timeout middleware implementation
fn timeoutMiddleware(request: *Request) !void {
    // Set timeout context (simplified)
    _ = request;
    
    // In a real implementation, you'd set up a timer
    std.log.debug("Request timeout middleware applied", .{});
}

/// Compression middleware implementation
fn compressionMiddleware(request: *Request) !void {
    // Check Accept-Encoding header
    const accept_encoding = request.getHeader("Accept-Encoding");
    
    if (accept_encoding) |encoding| {
        if (std.mem.indexOf(u8, encoding, "gzip") != null) {
            std.log.debug("Client supports gzip compression", .{});
            // Mark request for gzip compression in response
        } else if (std.mem.indexOf(u8, encoding, "deflate") != null) {
            std.log.debug("Client supports deflate compression", .{});
            // Mark request for deflate compression in response
        }
    }
}

/// Security headers middleware implementation
fn securityMiddleware(request: *Request) !void {
    // Security checks and header preparation
    _ = request;
    
    // In a real implementation, you'd:
    // - Check for suspicious patterns
    // - Validate input formats
    // - Prepare security headers for response
    
    std.log.debug("Security middleware applied", .{});
}

/// Request ID middleware implementation
fn requestIdMiddleware(request: *Request) !void {
    // Generate or extract request ID
    var request_id: [16]u8 = undefined;
    std.crypto.random.bytes(&request_id);
    
    // Convert to hex string
    var hex_id: [32]u8 = undefined;
    _ = std.fmt.bufPrint(&hex_id, "{}", .{std.fmt.fmtSliceHexLower(&request_id)}) catch "";
    
    // In a real implementation, you'd store this in the request context
    _ = request;
    
    std.log.debug("Request ID generated: {s}", .{hex_id});
}

/// Metrics collection middleware implementation
fn metricsMiddleware(request: *Request) !void {
    // Collect request metrics
    const method = request.method.toString();
    const path = request.path;
    const timestamp = std.time.timestamp();
    
    // In a real implementation, you'd store these metrics
    std.log.debug("Metrics collected: {} {s} at {}", .{ method, path, timestamp });
    
    // Track request count, response times, etc.
    // This would integrate with a metrics system like Prometheus
}

/// Middleware chain for processing multiple middleware
pub const MiddlewareChain = struct {
    allocator: Allocator,
    middleware_list: std.ArrayList(Middleware),
    
    const Self = @This();

    /// Initialize middleware chain
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .middleware_list = std.ArrayList(Middleware).init(allocator),
        };
    }

    /// Deinitialize middleware chain
    pub fn deinit(self: *Self) void {
        self.middleware_list.deinit();
    }

    /// Add middleware to chain
    pub fn add(self: *Self, middleware: Middleware) !void {
        try self.middleware_list.append(middleware);
        std.log.info("Added middleware: {s}", .{middleware.name});
    }

    /// Process request through all middleware
    pub fn process(self: *const Self, request: *Request) !void {
        for (self.middleware_list.items) |middleware| {
            middleware.process(request) catch |err| {
                std.log.err("Middleware '{}' failed: {}", .{ middleware.name, err });
                return err;
            };
        }
    }

    /// Get middleware count
    pub fn count(self: *const Self) usize {
        return self.middleware_list.items.len;
    }

    /// List middleware names
    pub fn listNames(self: *const Self, allocator: Allocator) ![][]const u8 {
        var names = std.ArrayList([]const u8).init(allocator);
        
        for (self.middleware_list.items) |middleware| {
            try names.append(middleware.name);
        }
        
        return names.toOwnedSlice();
    }
};

/// Middleware configuration
pub const MiddlewareConfig = struct {
    enable_cors: bool = true,
    enable_logging: bool = true,
    enable_auth: bool = false,
    enable_rate_limit: bool = false,
    enable_compression: bool = true,
    enable_security: bool = true,
    enable_metrics: bool = true,
    max_request_size: usize = 10 * 1024 * 1024, // 10MB
    request_timeout_ms: u32 = 30000, // 30 seconds
    
    /// Create default middleware chain
    pub fn createDefaultChain(self: *const MiddlewareConfig, allocator: Allocator) !MiddlewareChain {
        var chain = MiddlewareChain.init(allocator);
        
        // Add middleware in order of execution
        if (self.enable_logging) {
            try chain.add(Middleware.logging());
        }
        
        if (self.enable_metrics) {
            try chain.add(Middleware.metrics());
        }
        
        try chain.add(Middleware.requestId());
        
        if (self.enable_security) {
            try chain.add(Middleware.security());
        }
        
        if (self.enable_cors) {
            try chain.add(Middleware.cors());
        }
        
        try chain.add(Middleware.requestSizeLimit(self.max_request_size));
        
        if (self.enable_compression) {
            try chain.add(Middleware.compression());
        }
        
        if (self.enable_rate_limit) {
            try chain.add(Middleware.rateLimit());
        }
        
        if (self.enable_auth) {
            try chain.add(Middleware.auth());
        }
        
        try chain.add(Middleware.timeout(self.request_timeout_ms));
        
        return chain;
    }
};

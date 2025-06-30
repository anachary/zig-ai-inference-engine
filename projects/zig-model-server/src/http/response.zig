const std = @import("std");
const Allocator = std.mem.Allocator;

/// HTTP status codes
pub const StatusCode = enum(u16) {
    // 1xx Informational
    Continue = 100,
    SwitchingProtocols = 101,
    
    // 2xx Success
    OK = 200,
    Created = 201,
    Accepted = 202,
    NoContent = 204,
    
    // 3xx Redirection
    MovedPermanently = 301,
    Found = 302,
    NotModified = 304,
    
    // 4xx Client Error
    BadRequest = 400,
    Unauthorized = 401,
    Forbidden = 403,
    NotFound = 404,
    MethodNotAllowed = 405,
    NotAcceptable = 406,
    RequestTimeout = 408,
    Conflict = 409,
    Gone = 410,
    LengthRequired = 411,
    PayloadTooLarge = 413,
    UnsupportedMediaType = 415,
    UnprocessableEntity = 422,
    TooManyRequests = 429,
    
    // 5xx Server Error
    InternalServerError = 500,
    NotImplemented = 501,
    BadGateway = 502,
    ServiceUnavailable = 503,
    GatewayTimeout = 504,
    
    pub fn toInt(self: StatusCode) u16 {
        return @intFromEnum(self);
    }
    
    pub fn toString(self: StatusCode) []const u8 {
        return switch (self) {
            .Continue => "Continue",
            .SwitchingProtocols => "Switching Protocols",
            .OK => "OK",
            .Created => "Created",
            .Accepted => "Accepted",
            .NoContent => "No Content",
            .MovedPermanently => "Moved Permanently",
            .Found => "Found",
            .NotModified => "Not Modified",
            .BadRequest => "Bad Request",
            .Unauthorized => "Unauthorized",
            .Forbidden => "Forbidden",
            .NotFound => "Not Found",
            .MethodNotAllowed => "Method Not Allowed",
            .NotAcceptable => "Not Acceptable",
            .RequestTimeout => "Request Timeout",
            .Conflict => "Conflict",
            .Gone => "Gone",
            .LengthRequired => "Length Required",
            .PayloadTooLarge => "Payload Too Large",
            .UnsupportedMediaType => "Unsupported Media Type",
            .UnprocessableEntity => "Unprocessable Entity",
            .TooManyRequests => "Too Many Requests",
            .InternalServerError => "Internal Server Error",
            .NotImplemented => "Not Implemented",
            .BadGateway => "Bad Gateway",
            .ServiceUnavailable => "Service Unavailable",
            .GatewayTimeout => "Gateway Timeout",
        };
    }
};

/// HTTP response
pub const Response = struct {
    allocator: Allocator,
    status_code: u16,
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8,
    
    const Self = @This();

    /// Initialize response
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .status_code = 200,
            .headers = std.StringHashMap([]const u8).init(allocator),
            .body = null,
        };
    }

    /// Deinitialize response
    pub fn deinit(self: *Self) void {
        self.headers.deinit();
        if (self.body) |body| {
            self.allocator.free(body);
        }
    }

    /// Set status code
    pub fn setStatus(self: *Self, status: StatusCode) void {
        self.status_code = status.toInt();
    }

    /// Set status code by integer
    pub fn setStatusCode(self: *Self, code: u16) void {
        self.status_code = code;
    }

    /// Set header
    pub fn setHeader(self: *Self, name: []const u8, value: []const u8) !void {
        try self.headers.put(
            try self.allocator.dupe(u8, name),
            try self.allocator.dupe(u8, value)
        );
    }

    /// Set multiple headers
    pub fn setHeaders(self: *Self, headers: []const struct { []const u8, []const u8 }) !void {
        for (headers) |header| {
            try self.setHeader(header[0], header[1]);
        }
    }

    /// Set body
    pub fn setBody(self: *Self, body: []const u8) !void {
        if (self.body) |old_body| {
            self.allocator.free(old_body);
        }
        self.body = try self.allocator.dupe(u8, body);
        
        // Automatically set Content-Length
        const content_length = try std.fmt.allocPrint(self.allocator, "{}", .{body.len});
        defer self.allocator.free(content_length);
        try self.setHeader("Content-Length", content_length);
    }

    /// Set JSON body
    pub fn setJsonBody(self: *Self, json_data: anytype) !void {
        const json_string = try std.json.stringifyAlloc(self.allocator, json_data, .{});
        defer self.allocator.free(json_string);
        
        try self.setHeader("Content-Type", "application/json");
        try self.setBody(json_string);
    }

    /// Set HTML body
    pub fn setHtmlBody(self: *Self, html: []const u8) !void {
        try self.setHeader("Content-Type", "text/html; charset=utf-8");
        try self.setBody(html);
    }

    /// Set plain text body
    pub fn setTextBody(self: *Self, text: []const u8) !void {
        try self.setHeader("Content-Type", "text/plain; charset=utf-8");
        try self.setBody(text);
    }

    /// Set file body
    pub fn setFileBody(self: *Self, file_path: []const u8, content_type: ?[]const u8) !void {
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();
        
        const file_size = try file.getEndPos();
        const file_content = try self.allocator.alloc(u8, file_size);
        _ = try file.readAll(file_content);
        
        if (content_type) |ct| {
            try self.setHeader("Content-Type", ct);
        } else {
            // Try to guess content type from extension
            const guessed_type = guessContentType(file_path);
            try self.setHeader("Content-Type", guessed_type);
        }
        
        try self.setBody(file_content);
        self.allocator.free(file_content);
    }

    /// Add CORS headers
    pub fn addCorsHeaders(self: *Self) !void {
        try self.setHeader("Access-Control-Allow-Origin", "*");
        try self.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        try self.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
        try self.setHeader("Access-Control-Max-Age", "86400");
    }

    /// Add security headers
    pub fn addSecurityHeaders(self: *Self) !void {
        try self.setHeader("X-Content-Type-Options", "nosniff");
        try self.setHeader("X-Frame-Options", "DENY");
        try self.setHeader("X-XSS-Protection", "1; mode=block");
        try self.setHeader("Strict-Transport-Security", "max-age=31536000; includeSubDomains");
    }

    /// Add cache headers
    pub fn addCacheHeaders(self: *Self, max_age_seconds: u32) !void {
        const cache_control = try std.fmt.allocPrint(self.allocator, "public, max-age={}", .{max_age_seconds});
        defer self.allocator.free(cache_control);
        try self.setHeader("Cache-Control", cache_control);
        
        // Add ETag for better caching
        if (self.body) |body| {
            const etag = try generateETag(self.allocator, body);
            defer self.allocator.free(etag);
            try self.setHeader("ETag", etag);
        }
    }

    /// Set redirect
    pub fn setRedirect(self: *Self, location: []const u8, permanent: bool) !void {
        if (permanent) {
            self.setStatus(.MovedPermanently);
        } else {
            self.setStatus(.Found);
        }
        try self.setHeader("Location", location);
    }

    /// Serialize response to HTTP format
    pub fn serialize(self: *const Self, allocator: Allocator) ![]u8 {
        var response_parts = std.ArrayList([]const u8).init(allocator);
        defer response_parts.deinit();
        
        // Status line
        const status_text = getStatusText(self.status_code);
        const status_line = try std.fmt.allocPrint(allocator, "HTTP/1.1 {} {}\r\n", .{ self.status_code, status_text });
        try response_parts.append(status_line);
        
        // Headers
        var header_iter = self.headers.iterator();
        while (header_iter.next()) |entry| {
            const header_line = try std.fmt.allocPrint(allocator, "{s}: {s}\r\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            try response_parts.append(header_line);
        }
        
        // Empty line between headers and body
        try response_parts.append("\r\n");
        
        // Body
        if (self.body) |body| {
            try response_parts.append(body);
        }
        
        // Calculate total length
        var total_length: usize = 0;
        for (response_parts.items) |part| {
            total_length += part.len;
        }
        
        // Concatenate all parts
        var result = try allocator.alloc(u8, total_length);
        var offset: usize = 0;
        for (response_parts.items) |part| {
            @memcpy(result[offset..offset + part.len], part);
            offset += part.len;
        }
        
        // Free temporary allocations
        for (response_parts.items[0..response_parts.items.len - 1]) |part| {
            if (part.ptr != self.body.?.ptr) { // Don't free the body
                allocator.free(part);
            }
        }
        
        return result;
    }

    /// Create error response
    pub fn createError(allocator: Allocator, status: StatusCode, message: []const u8) !Response {
        var response = Response.init(allocator);
        response.setStatus(status);
        
        const error_json = try std.fmt.allocPrint(allocator, 
            "{{\"error\": \"{s}\", \"status\": {}, \"timestamp\": \"{s}\"}}", 
            .{ message, status.toInt(), getCurrentTimestamp() });
        defer allocator.free(error_json);
        
        try response.setJsonBody(error_json);
        return response;
    }

    /// Create success response
    pub fn createSuccess(allocator: Allocator, data: anytype) !Response {
        var response = Response.init(allocator);
        response.setStatus(.OK);
        try response.setJsonBody(data);
        return response;
    }

    /// Create not found response
    pub fn createNotFound(allocator: Allocator, resource: []const u8) !Response {
        const message = try std.fmt.allocPrint(allocator, "Resource not found: {s}", .{resource});
        defer allocator.free(message);
        return createError(allocator, .NotFound, message);
    }
};

/// Guess content type from file extension
fn guessContentType(file_path: []const u8) []const u8 {
    if (std.mem.endsWith(u8, file_path, ".html") or std.mem.endsWith(u8, file_path, ".htm")) {
        return "text/html";
    } else if (std.mem.endsWith(u8, file_path, ".css")) {
        return "text/css";
    } else if (std.mem.endsWith(u8, file_path, ".js")) {
        return "application/javascript";
    } else if (std.mem.endsWith(u8, file_path, ".json")) {
        return "application/json";
    } else if (std.mem.endsWith(u8, file_path, ".png")) {
        return "image/png";
    } else if (std.mem.endsWith(u8, file_path, ".jpg") or std.mem.endsWith(u8, file_path, ".jpeg")) {
        return "image/jpeg";
    } else if (std.mem.endsWith(u8, file_path, ".gif")) {
        return "image/gif";
    } else if (std.mem.endsWith(u8, file_path, ".svg")) {
        return "image/svg+xml";
    } else if (std.mem.endsWith(u8, file_path, ".pdf")) {
        return "application/pdf";
    } else if (std.mem.endsWith(u8, file_path, ".txt")) {
        return "text/plain";
    } else {
        return "application/octet-stream";
    }
}

/// Get status text for status code
fn getStatusText(status_code: u16) []const u8 {
    return switch (status_code) {
        100 => "Continue",
        101 => "Switching Protocols",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        408 => "Request Timeout",
        409 => "Conflict",
        410 => "Gone",
        411 => "Length Required",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        else => "Unknown",
    };
}

/// Generate ETag for content
fn generateETag(allocator: Allocator, content: []const u8) ![]u8 {
    // Simple hash-based ETag
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(content);
    const hash = hasher.final();
    
    return std.fmt.allocPrint(allocator, "\"{x}\"", .{hash});
}

/// Get current timestamp in ISO format
fn getCurrentTimestamp() []const u8 {
    // Simplified timestamp - in real implementation, use proper time formatting
    return "2024-01-01T00:00:00Z";
}

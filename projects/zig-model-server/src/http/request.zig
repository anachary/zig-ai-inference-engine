const std = @import("std");
const Allocator = std.mem.Allocator;
const net = std.net;

/// HTTP request method
pub const Method = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
    
    pub fn fromString(method_str: []const u8) ?Method {
        if (std.mem.eql(u8, method_str, "GET")) return .GET;
        if (std.mem.eql(u8, method_str, "POST")) return .POST;
        if (std.mem.eql(u8, method_str, "PUT")) return .PUT;
        if (std.mem.eql(u8, method_str, "DELETE")) return .DELETE;
        if (std.mem.eql(u8, method_str, "PATCH")) return .PATCH;
        if (std.mem.eql(u8, method_str, "HEAD")) return .HEAD;
        if (std.mem.eql(u8, method_str, "OPTIONS")) return .OPTIONS;
        return null;
    }
    
    pub fn toString(self: Method) []const u8 {
        return switch (self) {
            .GET => "GET",
            .POST => "POST",
            .PUT => "PUT",
            .DELETE => "DELETE",
            .PATCH => "PATCH",
            .HEAD => "HEAD",
            .OPTIONS => "OPTIONS",
        };
    }
};

/// HTTP version
pub const Version = enum {
    HTTP_1_0,
    HTTP_1_1,
    HTTP_2_0,
    
    pub fn fromString(version_str: []const u8) ?Version {
        if (std.mem.eql(u8, version_str, "HTTP/1.0")) return .HTTP_1_0;
        if (std.mem.eql(u8, version_str, "HTTP/1.1")) return .HTTP_1_1;
        if (std.mem.eql(u8, version_str, "HTTP/2.0")) return .HTTP_2_0;
        return null;
    }
    
    pub fn toString(self: Version) []const u8 {
        return switch (self) {
            .HTTP_1_0 => "HTTP/1.0",
            .HTTP_1_1 => "HTTP/1.1",
            .HTTP_2_0 => "HTTP/2.0",
        };
    }
};

/// Query parameter
pub const QueryParam = struct {
    key: []const u8,
    value: []const u8,
};

/// HTTP request
pub const Request = struct {
    allocator: Allocator,
    method: Method,
    path: []const u8,
    query_string: ?[]const u8,
    version: Version,
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8,
    query_params: std.ArrayList(QueryParam),
    path_params: std.StringHashMap([]const u8),
    
    const Self = @This();

    /// Initialize empty request
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .method = .GET,
            .path = "",
            .query_string = null,
            .version = .HTTP_1_1,
            .headers = std.StringHashMap([]const u8).init(allocator),
            .body = null,
            .query_params = std.ArrayList(QueryParam).init(allocator),
            .path_params = std.StringHashMap([]const u8).init(allocator),
        };
    }

    /// Deinitialize request
    pub fn deinit(self: *Self) void {
        self.headers.deinit();
        self.query_params.deinit();
        self.path_params.deinit();
        
        if (self.body) |body| {
            self.allocator.free(body);
        }
    }

    /// Parse HTTP request from socket
    pub fn parse(socket: net.Stream) !Self {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();
        
        // Read request line by line
        var buffer: [8192]u8 = undefined;
        var total_read: usize = 0;
        
        // Read until we have the complete headers
        while (total_read < buffer.len - 1) {
            const bytes_read = try socket.read(buffer[total_read..]);
            if (bytes_read == 0) break;
            total_read += bytes_read;
            
            // Check if we have complete headers (double CRLF)
            if (std.mem.indexOf(u8, buffer[0..total_read], "\r\n\r\n")) |_| {
                break;
            }
        }
        
        const request_data = buffer[0..total_read];
        
        // Split headers and body
        const header_end = std.mem.indexOf(u8, request_data, "\r\n\r\n") orelse request_data.len;
        const headers_section = request_data[0..header_end];
        
        // Parse request line
        var lines = std.mem.split(u8, headers_section, "\r\n");
        const request_line = lines.next() orelse return error.InvalidRequest;
        
        var request_parts = std.mem.split(u8, request_line, " ");
        const method_str = request_parts.next() orelse return error.InvalidRequest;
        const uri = request_parts.next() orelse return error.InvalidRequest;
        const version_str = request_parts.next() orelse return error.InvalidRequest;
        
        const method = Method.fromString(method_str) orelse return error.InvalidRequest;
        const version = Version.fromString(version_str) orelse return error.InvalidRequest;
        
        // Parse URI (path and query string)
        var uri_parts = std.mem.split(u8, uri, "?");
        const path = uri_parts.next() orelse return error.InvalidRequest;
        const query_string = uri_parts.next();
        
        // Create request with permanent allocator
        var request = Self.init(std.heap.page_allocator);
        request.method = method;
        request.path = try std.heap.page_allocator.dupe(u8, path);
        request.version = version;
        
        if (query_string) |qs| {
            request.query_string = try std.heap.page_allocator.dupe(u8, qs);
            try request.parseQueryParams(qs);
        }
        
        // Parse headers
        while (lines.next()) |line| {
            if (line.len == 0) break;
            
            const colon_pos = std.mem.indexOf(u8, line, ":") orelse continue;
            const header_name = std.mem.trim(u8, line[0..colon_pos], " \t");
            const header_value = std.mem.trim(u8, line[colon_pos + 1..], " \t");
            
            try request.headers.put(
                try std.heap.page_allocator.dupe(u8, header_name),
                try std.heap.page_allocator.dupe(u8, header_value)
            );
        }
        
        // Read body if present
        if (request.getHeader("Content-Length")) |content_length_str| {
            const content_length = std.fmt.parseInt(usize, content_length_str, 10) catch 0;
            if (content_length > 0) {
                const body_start = header_end + 4; // Skip "\r\n\r\n"
                const body_in_buffer = if (body_start < total_read) 
                    request_data[body_start..total_read] 
                else 
                    "";
                
                var body = try std.heap.page_allocator.alloc(u8, content_length);
                
                // Copy what we already have
                const already_read = @min(body_in_buffer.len, content_length);
                @memcpy(body[0..already_read], body_in_buffer[0..already_read]);
                
                // Read remaining body
                var body_read = already_read;
                while (body_read < content_length) {
                    const bytes_read = try socket.read(body[body_read..]);
                    if (bytes_read == 0) break;
                    body_read += bytes_read;
                }
                
                request.body = body[0..body_read];
            }
        }
        
        return request;
    }

    /// Get header value
    pub fn getHeader(self: *const Self, name: []const u8) ?[]const u8 {
        return self.headers.get(name);
    }

    /// Get query parameter value
    pub fn getQueryParam(self: *const Self, name: []const u8) ?[]const u8 {
        for (self.query_params.items) |param| {
            if (std.mem.eql(u8, param.key, name)) {
                return param.value;
            }
        }
        return null;
    }

    /// Get path parameter value
    pub fn getPathParam(self: *const Self, name: []const u8) ?[]const u8 {
        return self.path_params.get(name);
    }

    /// Set path parameter (used by router)
    pub fn setPathParam(self: *Self, name: []const u8, value: []const u8) !void {
        try self.path_params.put(
            try self.allocator.dupe(u8, name),
            try self.allocator.dupe(u8, value)
        );
    }

    /// Check if request should keep connection alive
    pub fn shouldKeepAlive(self: *const Self) bool {
        if (self.version == .HTTP_1_0) {
            // HTTP/1.0 requires explicit Connection: keep-alive
            if (self.getHeader("Connection")) |connection| {
                return std.ascii.eqlIgnoreCase(connection, "keep-alive");
            }
            return false;
        } else {
            // HTTP/1.1 defaults to keep-alive unless Connection: close
            if (self.getHeader("Connection")) |connection| {
                return !std.ascii.eqlIgnoreCase(connection, "close");
            }
            return true;
        }
    }

    /// Get content type
    pub fn getContentType(self: *const Self) ?[]const u8 {
        return self.getHeader("Content-Type");
    }

    /// Check if request is JSON
    pub fn isJson(self: *const Self) bool {
        if (self.getContentType()) |content_type| {
            return std.mem.indexOf(u8, content_type, "application/json") != null;
        }
        return false;
    }

    /// Get request body as string
    pub fn getBodyAsString(self: *const Self) ?[]const u8 {
        return self.body;
    }

    /// Parse JSON body
    pub fn parseJsonBody(self: *const Self, comptime T: type) !T {
        const body_str = self.getBodyAsString() orelse return error.NoBody;
        return std.json.parseFromSlice(T, self.allocator, body_str, .{});
    }

    /// Get user agent
    pub fn getUserAgent(self: *const Self) ?[]const u8 {
        return self.getHeader("User-Agent");
    }

    /// Get client IP address (considering X-Forwarded-For)
    pub fn getClientIP(self: *const Self) ?[]const u8 {
        // Check X-Forwarded-For header first (for proxies)
        if (self.getHeader("X-Forwarded-For")) |forwarded| {
            // Take the first IP in the list
            var ips = std.mem.split(u8, forwarded, ",");
            if (ips.next()) |first_ip| {
                return std.mem.trim(u8, first_ip, " \t");
            }
        }
        
        // Check X-Real-IP header
        if (self.getHeader("X-Real-IP")) |real_ip| {
            return real_ip;
        }
        
        // Fall back to direct connection (would need socket info)
        return null;
    }

    /// Parse query parameters from query string
    fn parseQueryParams(self: *Self, query_string: []const u8) !void {
        var params = std.mem.split(u8, query_string, "&");
        while (params.next()) |param| {
            var kv = std.mem.split(u8, param, "=");
            const key = kv.next() orelse continue;
            const value = kv.next() orelse "";
            
            // URL decode key and value
            const decoded_key = try self.urlDecode(key);
            const decoded_value = try self.urlDecode(value);
            
            try self.query_params.append(QueryParam{
                .key = decoded_key,
                .value = decoded_value,
            });
        }
    }

    /// URL decode a string
    fn urlDecode(self: *Self, encoded: []const u8) ![]const u8 {
        var decoded = try self.allocator.alloc(u8, encoded.len);
        var decoded_len: usize = 0;
        var i: usize = 0;
        
        while (i < encoded.len) {
            if (encoded[i] == '%' and i + 2 < encoded.len) {
                // Decode hex escape
                const hex_str = encoded[i + 1..i + 3];
                const byte_value = std.fmt.parseInt(u8, hex_str, 16) catch {
                    // Invalid hex, keep as-is
                    decoded[decoded_len] = encoded[i];
                    decoded_len += 1;
                    i += 1;
                    continue;
                };
                decoded[decoded_len] = byte_value;
                decoded_len += 1;
                i += 3;
            } else if (encoded[i] == '+') {
                // Convert + to space
                decoded[decoded_len] = ' ';
                decoded_len += 1;
                i += 1;
            } else {
                decoded[decoded_len] = encoded[i];
                decoded_len += 1;
                i += 1;
            }
        }
        
        return decoded[0..decoded_len];
    }
};

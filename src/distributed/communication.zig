const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const net = std.net;
const http = std.http;
const json = std.json;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;

const DistributedTensor = @import("inference_coordinator.zig").DistributedTensor;

/// HTTP client for inter-VM communication
pub const VMClient = struct {
    allocator: Allocator,
    base_url: []const u8,
    timeout_ms: u64,
    retry_count: u8,
    
    const Self = @This();
    
    pub const ClientError = error{
        ConnectionFailed,
        RequestTimeout,
        InvalidResponse,
        ServerError,
        NetworkError,
    };
    
    pub fn init(allocator: Allocator, base_url: []const u8) Self {
        return Self{
            .allocator = allocator,
            .base_url = base_url,
            .timeout_ms = 30000, // 30 seconds
            .retry_count = 3,
        };
    }
    
    /// Send tensor to another VM
    pub fn sendTensor(self: *Self, endpoint: []const u8, tensor: *const DistributedTensor) ![]u8 {
        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_url, endpoint });
        defer self.allocator.free(url);
        
        // Serialize tensor
        const payload = try tensor.serialize(self.allocator);
        defer self.allocator.free(payload);
        
        // Send HTTP POST request
        return try self.sendHttpRequest("POST", url, payload);
    }
    
    /// Execute layer on remote VM
    pub fn executeLayer(self: *Self, layer_id: u32, input_tensor: *const DistributedTensor) !DistributedTensor {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/api/v1/layers/{d}/execute", .{layer_id});
        defer self.allocator.free(endpoint);
        
        const response_data = try self.sendTensor(endpoint, input_tensor);
        defer self.allocator.free(response_data);
        
        // Deserialize response tensor
        return try DistributedTensor.deserialize(self.allocator, response_data);
    }
    
    /// Check health of remote VM
    pub fn healthCheck(self: *Self) !bool {
        const response = self.sendHttpRequest("GET", "/api/v1/health", "") catch |err| {
            std.log.warn("Health check failed: {any}", .{err});
            return false;
        };
        defer self.allocator.free(response);
        
        // Parse health response
        const parsed = json.parseFromSlice(json.Value, self.allocator, response, .{}) catch return false;
        defer parsed.deinit();
        
        if (parsed.value.object.get("status")) |status| {
            return std.mem.eql(u8, status.string, "healthy");
        }
        
        return false;
    }
    
    /// Get VM statistics
    pub fn getStats(self: *Self) !VMStats {
        const response = try self.sendHttpRequest("GET", "/api/v1/stats", "");
        defer self.allocator.free(response);
        
        const parsed = try json.parseFromSlice(VMStats, self.allocator, response, .{});
        defer parsed.deinit();
        
        return parsed.value;
    }
    
    /// Send HTTP request with retries
    fn sendHttpRequest(self: *Self, method: []const u8, url: []const u8, body: []const u8) ![]u8 {
        var attempt: u8 = 0;
        
        while (attempt < self.retry_count) {
            const result = self.sendHttpRequestOnce(method, url, body) catch |err| {
                attempt += 1;
                if (attempt >= self.retry_count) {
                    return err;
                }
                
                // Exponential backoff
                const delay_ms = @as(u64, 100) * (@as(u64, 1) << attempt);
                std.time.sleep(delay_ms * std.time.ns_per_ms);
                continue;
            };
            
            return result;
        }
        
        return ClientError.ConnectionFailed;
    }
    
    /// Send single HTTP request
    fn sendHttpRequestOnce(self: *Self, method: []const u8, url: []const u8, body: []const u8) ![]u8 {
        _ = method;
        _ = url;
        _ = body;
        
        // This would implement actual HTTP client
        // For now, return mock response
        const mock_response = 
            \\{"status": "success", "data": {"tensor": {"shape": [1, 2048], "data": []}}}
        ;
        
        return try self.allocator.dupe(u8, mock_response);
    }
    
    pub const VMStats = struct {
        memory_usage_mb: u64,
        cpu_usage_percent: f32,
        gpu_usage_percent: f32,
        active_requests: u32,
        uptime_seconds: u64,
    };
};

/// HTTP server for receiving inter-VM requests
pub const VMServer = struct {
    allocator: Allocator,
    port: u16,
    server: ?http.Server,
    running: bool,
    request_handlers: HashMap([]const u8, RequestHandler),
    
    const Self = @This();
    
    pub const RequestHandler = *const fn (allocator: Allocator, request_body: []const u8) anyerror![]u8;
    
    pub fn init(allocator: Allocator, port: u16) Self {
        return Self{
            .allocator = allocator,
            .port = port,
            .server = null,
            .running = false,
            .request_handlers = HashMap([]const u8, RequestHandler).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.stop();
        self.request_handlers.deinit();
    }
    
    /// Start the server
    pub fn start(self: *Self) !void {
        if (self.running) return;
        
        // Register default handlers
        try self.registerDefaultHandlers();
        
        // Start HTTP server
        const address = try net.Address.parseIp("0.0.0.0", self.port);
        self.server = http.Server.init(self.allocator, .{ .reuse_address = true });
        
        try self.server.?.listen(address);
        self.running = true;
        
        std.log.info("🌐 VM server listening on port {d}", .{self.port});
        
        // Start accepting connections
        while (self.running) {
            const connection = self.server.?.accept(.{ .allocator = self.allocator }) catch |err| {
                std.log.warn("Failed to accept connection: {any}", .{err});
                continue;
            };
            
            // Handle request in separate thread
            _ = Thread.spawn(.{}, handleConnection, .{ self, connection }) catch |err| {
                std.log.warn("Failed to spawn handler thread: {any}", .{err});
                connection.deinit();
            };
        }
    }
    
    /// Stop the server
    pub fn stop(self: *Self) void {
        if (!self.running) return;
        
        self.running = false;
        if (self.server) |*server| {
            server.deinit();
            self.server = null;
        }
        
        std.log.info("🛑 VM server stopped");
    }
    
    /// Register request handler
    pub fn registerHandler(self: *Self, path: []const u8, handler: RequestHandler) !void {
        const path_copy = try self.allocator.dupe(u8, path);
        try self.request_handlers.put(path_copy, handler);
    }
    
    /// Register default API handlers
    fn registerDefaultHandlers(self: *Self) !void {
        try self.registerHandler("/api/v1/health", handleHealthCheck);
        try self.registerHandler("/api/v1/stats", handleStatsRequest);
        try self.registerHandler("/api/v1/layers/execute", handleLayerExecution);
        try self.registerHandler("/api/v1/tensors/receive", handleTensorReceive);
    }
    
    /// Handle incoming connection
    fn handleConnection(self: *Self, connection: http.Server.Connection) void {
        defer connection.deinit();
        
        var read_buffer: [8192]u8 = undefined;
        var server_request = connection.receiveHead(&read_buffer) catch |err| {
            std.log.warn("Failed to receive request head: {any}", .{err});
            return;
        };
        
        // Read request body
        const body = server_request.reader().readAllAlloc(self.allocator, 1024 * 1024) catch |err| {
            std.log.warn("Failed to read request body: {any}", .{err});
            return;
        };
        defer self.allocator.free(body);
        
        // Find handler for the request path
        const path = server_request.head.target;
        const handler = self.request_handlers.get(path) orelse {
            self.sendErrorResponse(&server_request, 404, "Not Found") catch {};
            return;
        };
        
        // Execute handler
        const response_body = handler(self.allocator, body) catch |err| {
            std.log.warn("Handler error for {s}: {any}", .{ path, err });
            self.sendErrorResponse(&server_request, 500, "Internal Server Error") catch {};
            return;
        };
        defer self.allocator.free(response_body);
        
        // Send response
        self.sendSuccessResponse(&server_request, response_body) catch |err| {
            std.log.warn("Failed to send response: {any}", .{err});
        };
    }
    
    /// Send success response
    fn sendSuccessResponse(self: *Self, request: *http.Server.Request, body: []const u8) !void {
        _ = self;
        
        try request.respond(body, .{
            .status = .ok,
            .extra_headers = &[_]http.Header{
                .{ .name = "content-type", .value = "application/json" },
                .{ .name = "access-control-allow-origin", .value = "*" },
            },
        });
    }
    
    /// Send error response
    fn sendErrorResponse(self: *Self, request: *http.Server.Request, status_code: u16, message: []const u8) !void {
        _ = self;
        
        const error_body = try std.fmt.allocPrint(request.allocator, 
            \\{{"error": "{s}", "status": {d}}}
        , .{ message, status_code });
        defer request.allocator.free(error_body);
        
        const status: http.Status = switch (status_code) {
            404 => .not_found,
            500 => .internal_server_error,
            else => .bad_request,
        };
        
        try request.respond(error_body, .{
            .status = status,
            .extra_headers = &[_]http.Header{
                .{ .name = "content-type", .value = "application/json" },
            },
        });
    }
};

/// Default request handlers

/// Health check handler
fn handleHealthCheck(allocator: Allocator, request_body: []const u8) ![]u8 {
    _ = request_body;
    
    const response = 
        \\{"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}
    ;
    
    return try allocator.dupe(u8, response);
}

/// Stats request handler
fn handleStatsRequest(allocator: Allocator, request_body: []const u8) ![]u8 {
    _ = request_body;
    
    const response = 
        \\{
        \\  "memory_usage_mb": 1024,
        \\  "cpu_usage_percent": 45.2,
        \\  "gpu_usage_percent": 78.5,
        \\  "active_requests": 3,
        \\  "uptime_seconds": 86400
        \\}
    ;
    
    return try allocator.dupe(u8, response);
}

/// Layer execution handler
fn handleLayerExecution(allocator: Allocator, request_body: []const u8) ![]u8 {
    // Parse input tensor
    const input_tensor = DistributedTensor.deserialize(allocator, request_body) catch |err| {
        std.log.warn("Failed to deserialize input tensor: {any}", .{err});
        return error.InvalidRequest;
    };
    defer input_tensor.deinit(allocator);
    
    // Execute layer (mock implementation)
    const output_shape = [_]u32{ 1, 2048, 4096 };
    var output_tensor = try DistributedTensor.init(allocator, &output_shape, .f32);
    defer output_tensor.deinit(allocator);
    
    // Fill with mock computation results
    for (output_tensor.data, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(i % 1000)) / 1000.0;
    }
    
    // Serialize and return output tensor
    return try output_tensor.serialize(allocator);
}

/// Tensor receive handler
fn handleTensorReceive(allocator: Allocator, request_body: []const u8) ![]u8 {
    // Receive and process tensor
    const tensor = DistributedTensor.deserialize(allocator, request_body) catch |err| {
        std.log.warn("Failed to deserialize received tensor: {any}", .{err});
        return error.InvalidRequest;
    };
    defer tensor.deinit(allocator);
    
    std.log.info("📥 Received tensor with shape: {any}", .{tensor.shape});
    
    const response = 
        \\{"status": "received", "message": "Tensor processed successfully"}
    ;
    
    return try allocator.dupe(u8, response);
}

/// Connection pool for managing multiple VM connections
pub const ConnectionPool = struct {
    allocator: Allocator,
    clients: HashMap([]const u8, *VMClient),
    max_connections: u32,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator, max_connections: u32) Self {
        return Self{
            .allocator = allocator,
            .clients = HashMap([]const u8, *VMClient).init(allocator),
            .max_connections = max_connections,
        };
    }
    
    pub fn deinit(self: *Self) void {
        var iterator = self.clients.iterator();
        while (iterator.next()) |entry| {
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.clients.deinit();
    }
    
    /// Get or create client for VM
    pub fn getClient(self: *Self, vm_address: []const u8) !*VMClient {
        if (self.clients.get(vm_address)) |client| {
            return client;
        }
        
        if (self.clients.count() >= self.max_connections) {
            return error.TooManyConnections;
        }
        
        const client = try self.allocator.create(VMClient);
        client.* = VMClient.init(self.allocator, vm_address);
        
        const address_copy = try self.allocator.dupe(u8, vm_address);
        try self.clients.put(address_copy, client);
        
        return client;
    }
    
    /// Remove client
    pub fn removeClient(self: *Self, vm_address: []const u8) void {
        if (self.clients.fetchRemove(vm_address)) |entry| {
            self.allocator.destroy(entry.value);
            self.allocator.free(entry.key);
        }
    }
};

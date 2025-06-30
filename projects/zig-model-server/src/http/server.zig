const std = @import("std");
const Allocator = std.mem.Allocator;
const net = std.net;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;

// Import dependencies
const inference_engine = @import("zig-inference-engine");
const ModelManager = @import("../models/manager.zig").ModelManager;
const Router = @import("router.zig").Router;
const Request = @import("request.zig").Request;
const Response = @import("response.zig").Response;
const Middleware = @import("middleware.zig").Middleware;

/// HTTP server configuration
pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_connections: u32 = 100,
    request_timeout_ms: u32 = 30000,
    enable_cors: bool = true,
    enable_metrics: bool = true,
    enable_websockets: bool = true,
    static_files_dir: ?[]const u8 = null,
    tls_cert_path: ?[]const u8 = null,
    tls_key_path: ?[]const u8 = null,
    worker_threads: ?u32 = null,
    max_request_size: usize = 10 * 1024 * 1024, // 10MB
    keep_alive_timeout_ms: u32 = 5000,
};

/// Server statistics
pub const ServerStats = struct {
    total_requests: u64 = 0,
    active_connections: u32 = 0,
    total_connections: u64 = 0,
    requests_per_second: f32 = 0.0,
    average_response_time_ms: f32 = 0.0,
    error_count: u64 = 0,
    bytes_sent: u64 = 0,
    bytes_received: u64 = 0,
    uptime_seconds: u64 = 0,
    start_time: i64 = 0,
};

/// Connection state
const ConnectionState = struct {
    socket: net.Stream,
    address: net.Address,
    thread: ?Thread,
    active: bool,
    created_at: i64,
    last_activity: i64,
};

/// HTTP server errors
pub const ServerError = error{
    BindFailed,
    ListenFailed,
    AcceptFailed,
    InvalidRequest,
    RequestTooLarge,
    Timeout,
    InternalError,
    ModelNotFound,
    InferenceError,
};

/// Main HTTP server
pub const HTTPServer = struct {
    allocator: Allocator,
    config: ServerConfig,
    stats: ServerStats,
    
    // Core components
    router: Router,
    model_manager: ?*ModelManager,
    inference_engine: ?*inference_engine.Engine,
    middleware_stack: std.ArrayList(Middleware),
    
    // Network state
    listener: ?net.Server,
    connections: std.ArrayList(ConnectionState),
    connection_mutex: Mutex,
    
    // Server state
    running: bool,
    shutdown_requested: bool,
    worker_pool: std.ArrayList(Thread),
    
    const Self = @This();

    /// Initialize HTTP server
    pub fn init(allocator: Allocator, config: ServerConfig) !Self {
        var self = Self{
            .allocator = allocator,
            .config = config,
            .stats = ServerStats{
                .start_time = std.time.timestamp(),
            },
            .router = try Router.init(allocator),
            .model_manager = null,
            .inference_engine = null,
            .middleware_stack = std.ArrayList(Middleware).init(allocator),
            .listener = null,
            .connections = std.ArrayList(ConnectionState).init(allocator),
            .connection_mutex = Mutex{},
            .running = false,
            .shutdown_requested = false,
            .worker_pool = std.ArrayList(Thread).init(allocator),
        };

        // Register default routes
        try self.registerDefaultRoutes();
        
        // Add default middleware
        try self.addDefaultMiddleware();

        return self;
    }

    /// Deinitialize server
    pub fn deinit(self: *Self) void {
        self.stop();
        
        // Clean up connections
        self.connection_mutex.lock();
        for (self.connections.items) |*conn| {
            if (conn.active) {
                conn.socket.close();
                if (conn.thread) |thread| {
                    thread.join();
                }
            }
        }
        self.connections.deinit();
        self.connection_mutex.unlock();
        
        // Clean up worker pool
        for (self.worker_pool.items) |thread| {
            thread.join();
        }
        self.worker_pool.deinit();
        
        self.middleware_stack.deinit();
        self.router.deinit();
    }

    /// Attach inference engine
    pub fn attachInferenceEngine(self: *Self, engine: *inference_engine.Engine) !void {
        self.inference_engine = engine;
        
        // Initialize model manager with the engine
        if (self.model_manager == null) {
            self.model_manager = try self.allocator.create(ModelManager);
            self.model_manager.?.* = try ModelManager.init(self.allocator, engine);
        }
    }

    /// Load a model
    pub fn loadModel(self: *Self, name: []const u8, path: []const u8) !void {
        if (self.model_manager == null) {
            return ServerError.InternalError;
        }
        
        const config = ModelManager.ModelConfig{
            .max_batch_size = 4,
            .optimization_level = .balanced,
            .enable_caching = true,
        };
        
        try self.model_manager.?.loadModel(name, path, config);
        std.log.info("Model '{}' loaded from '{}'", .{ name, path });
    }

    /// Unload a model
    pub fn unloadModel(self: *Self, name: []const u8) !void {
        if (self.model_manager == null) {
            return ServerError.ModelNotFound;
        }
        
        try self.model_manager.?.unloadModel(name);
        std.log.info("Model '{}' unloaded", .{name});
    }

    /// Add middleware to the stack
    pub fn addMiddleware(self: *Self, middleware: Middleware) !void {
        try self.middleware_stack.append(middleware);
    }

    /// Add a custom route
    pub fn addRoute(self: *Self, method: []const u8, path: []const u8, handler: Router.HandlerFn) !void {
        try self.router.addRoute(method, path, handler);
    }

    /// Start the server
    pub fn start(self: *Self) !void {
        if (self.running) {
            return;
        }

        // Parse address
        const address = net.Address.parseIp(self.config.host, self.config.port) catch |err| {
            std.log.err("Failed to parse address {}:{}: {}", .{ self.config.host, self.config.port, err });
            return ServerError.BindFailed;
        };

        // Create listener
        self.listener = net.Address.listen(address, .{
            .reuse_address = true,
            .reuse_port = true,
        }) catch |err| {
            std.log.err("Failed to bind to {}:{}: {}", .{ self.config.host, self.config.port, err });
            return ServerError.BindFailed;
        };

        self.running = true;
        self.shutdown_requested = false;

        std.log.info("HTTP server listening on {}:{}", .{ self.config.host, self.config.port });

        // Start worker threads
        const worker_count = self.config.worker_threads orelse @max(1, std.Thread.getCpuCount() catch 4);
        try self.worker_pool.ensureTotalCapacity(worker_count);
        
        for (0..worker_count) |i| {
            const thread = try Thread.spawn(.{}, workerLoop, .{ self, i });
            try self.worker_pool.append(thread);
        }

        // Main accept loop
        while (self.running and !self.shutdown_requested) {
            if (self.listener) |*listener| {
                const connection = listener.accept() catch |err| {
                    if (err == error.SocketNotListening) break;
                    std.log.warn("Failed to accept connection: {}", .{err});
                    continue;
                };

                try self.handleNewConnection(connection);
            }
        }
    }

    /// Stop the server
    pub fn stop(self: *Self) void {
        if (!self.running) return;

        std.log.info("Stopping HTTP server...", .{});
        self.shutdown_requested = true;
        self.running = false;

        if (self.listener) |*listener| {
            listener.deinit();
            self.listener = null;
        }

        std.log.info("HTTP server stopped", .{});
    }

    /// Get server statistics
    pub fn getStats(self: *const Self) ServerStats {
        var stats = self.stats;
        stats.uptime_seconds = @intCast(std.time.timestamp() - stats.start_time);
        
        // Calculate requests per second
        if (stats.uptime_seconds > 0) {
            stats.requests_per_second = @as(f32, @floatFromInt(stats.total_requests)) / @as(f32, @floatFromInt(stats.uptime_seconds));
        }
        
        return stats;
    }

    /// Handle new connection
    fn handleNewConnection(self: *Self, connection: net.Server.Connection) !void {
        self.connection_mutex.lock();
        defer self.connection_mutex.unlock();

        // Check connection limit
        if (self.connections.items.len >= self.config.max_connections) {
            connection.stream.close();
            return;
        }

        const conn_state = ConnectionState{
            .socket = connection.stream,
            .address = connection.address,
            .thread = null,
            .active = true,
            .created_at = std.time.timestamp(),
            .last_activity = std.time.timestamp(),
        };

        try self.connections.append(conn_state);
        self.stats.total_connections += 1;
        self.stats.active_connections += 1;

        // Spawn thread to handle connection
        const thread = try Thread.spawn(.{}, handleConnection, .{ self, self.connections.items.len - 1 });
        self.connections.items[self.connections.items.len - 1].thread = thread;
    }

    /// Worker thread loop
    fn workerLoop(self: *Self, worker_id: usize) void {
        std.log.info("Worker thread {} started", .{worker_id});
        
        while (self.running and !self.shutdown_requested) {
            // Worker threads can handle background tasks
            // For now, just sleep
            std.time.sleep(100_000_000); // 100ms
        }
        
        std.log.info("Worker thread {} stopped", .{worker_id});
    }

    /// Handle individual connection
    fn handleConnection(self: *Self, connection_index: usize) void {
        defer {
            self.connection_mutex.lock();
            if (connection_index < self.connections.items.len) {
                self.connections.items[connection_index].active = false;
                self.connections.items[connection_index].socket.close();
                self.stats.active_connections -= 1;
            }
            self.connection_mutex.unlock();
        }

        const socket = self.connections.items[connection_index].socket;
        
        while (self.running and !self.shutdown_requested) {
            // Read request
            const request = self.readRequest(socket) catch |err| {
                if (err == error.EndOfStream or err == error.ConnectionResetByPeer) {
                    break; // Client disconnected
                }
                std.log.warn("Failed to read request: {}", .{err});
                break;
            };
            defer request.deinit();

            // Update activity timestamp
            self.connections.items[connection_index].last_activity = std.time.timestamp();

            // Process request
            const response = self.processRequest(request) catch |err| {
                std.log.err("Failed to process request: {}", .{err});
                self.sendErrorResponse(socket, 500, "Internal Server Error") catch {};
                continue;
            };
            defer response.deinit();

            // Send response
            self.sendResponse(socket, response) catch |err| {
                std.log.warn("Failed to send response: {}", .{err});
                break;
            };

            self.stats.total_requests += 1;

            // Check if connection should be kept alive
            if (!request.shouldKeepAlive()) {
                break;
            }
        }
    }

    /// Read HTTP request from socket
    fn readRequest(self: *Self, socket: net.Stream) !Request {
        _ = self;
        return Request.parse(socket);
    }

    /// Process HTTP request
    fn processRequest(self: *Self, request: Request) !Response {
        const start_time = std.time.nanoTimestamp();
        defer {
            const end_time = std.time.nanoTimestamp();
            const duration_ms = @as(f32, @floatFromInt(end_time - start_time)) / 1_000_000.0;
            
            // Update average response time
            const total_time = self.stats.average_response_time_ms * @as(f32, @floatFromInt(self.stats.total_requests));
            self.stats.average_response_time_ms = (total_time + duration_ms) / @as(f32, @floatFromInt(self.stats.total_requests + 1));
        }

        // Apply middleware stack
        for (self.middleware_stack.items) |middleware| {
            try middleware.process(&request);
        }

        // Route request
        return self.router.route(request);
    }

    /// Send HTTP response
    fn sendResponse(self: *Self, socket: net.Stream, response: Response) !void {
        const response_data = try response.serialize(self.allocator);
        defer self.allocator.free(response_data);
        
        _ = try socket.writeAll(response_data);
        self.stats.bytes_sent += response_data.len;
    }

    /// Send error response
    fn sendErrorResponse(self: *Self, socket: net.Stream, status_code: u16, message: []const u8) !void {
        var response = Response.init(self.allocator);
        defer response.deinit();
        
        response.status_code = status_code;
        try response.setHeader("Content-Type", "application/json");
        
        const error_json = try std.fmt.allocPrint(self.allocator, 
            "{{\"error\": \"{s}\", \"status\": {}}}", .{ message, status_code });
        defer self.allocator.free(error_json);
        
        try response.setBody(error_json);
        try self.sendResponse(socket, response);
    }

    /// Register default routes
    fn registerDefaultRoutes(self: *Self) !void {
        // Health check
        try self.router.addRoute("GET", "/health", healthHandler);
        
        // Server info
        try self.router.addRoute("GET", "/api/v1/info", infoHandler);
        
        // Metrics
        try self.router.addRoute("GET", "/metrics", metricsHandler);
        
        // Model management routes will be added by the router module
    }

    /// Add default middleware
    fn addDefaultMiddleware(self: *Self) !void {
        // CORS middleware
        if (self.config.enable_cors) {
            try self.addMiddleware(Middleware.cors());
        }
        
        // Logging middleware
        try self.addMiddleware(Middleware.logging());
        
        // Request size limit middleware
        try self.addMiddleware(Middleware.requestSizeLimit(self.config.max_request_size));
    }
};

// Default route handlers
fn healthHandler(request: Request, allocator: Allocator) !Response {
    _ = request;
    var response = Response.init(allocator);
    try response.setHeader("Content-Type", "application/json");
    try response.setBody("{\"status\": \"healthy\", \"timestamp\": \"" ++ "2024-01-01T00:00:00Z" ++ "\"}");
    return response;
}

fn infoHandler(request: Request, allocator: Allocator) !Response {
    _ = request;
    var response = Response.init(allocator);
    try response.setHeader("Content-Type", "application/json");
    const info_json = 
        \\{
        \\  "name": "zig-model-server",
        \\  "version": "0.1.0",
        \\  "description": "HTTP API and CLI interfaces for neural network model serving",
        \\  "endpoints": [
        \\    "GET /health",
        \\    "GET /api/v1/info",
        \\    "GET /metrics",
        \\    "GET /api/v1/models",
        \\    "POST /api/v1/models",
        \\    "POST /api/v1/models/{name}/infer"
        \\  ]
        \\}
    ;
    try response.setBody(info_json);
    return response;
}

fn metricsHandler(request: Request, allocator: Allocator) !Response {
    _ = request;
    var response = Response.init(allocator);
    try response.setHeader("Content-Type", "application/json");
    try response.setBody("{\"metrics\": \"TODO: Implement metrics collection\"}");
    return response;
}

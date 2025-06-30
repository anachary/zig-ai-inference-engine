const std = @import("std");
const model_server = @import("zig-model-server");
const inference_engine = @import("zig-inference-engine");

/// Basic HTTP server example demonstrating the core functionality
/// of the Zig Model Server
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== Zig Model Server - Basic HTTP Server Example ===", .{});

    // Initialize inference engine
    std.log.info("Initializing inference engine...", .{});
    var engine = try inference_engine.createServerEngine(allocator);
    defer engine.deinit();

    std.log.info("Inference engine initialized successfully!", .{});

    // Create server configuration
    const server_config = model_server.ServerConfig{
        .host = "127.0.0.1",
        .port = 8080,
        .max_connections = 100,
        .enable_cors = true,
        .enable_metrics = true,
        .enable_websockets = true,
        .worker_threads = 4,
    };

    // Initialize HTTP server
    std.log.info("Initializing HTTP server...", .{});
    var server = try model_server.HTTPServer.init(allocator, server_config);
    defer server.deinit();

    // Attach inference engine to server
    try server.attachInferenceEngine(&engine);
    std.log.info("Inference engine attached to HTTP server", .{});

    // Display server configuration
    std.log.info("Server Configuration:", .{});
    std.log.info("  Host: {s}", .{server_config.host});
    std.log.info("  Port: {}", .{server_config.port});
    std.log.info("  Max Connections: {}", .{server_config.max_connections});
    std.log.info("  Worker Threads: {}", .{server_config.worker_threads.?});
    std.log.info("  CORS Enabled: {}", .{server_config.enable_cors});
    std.log.info("  Metrics Enabled: {}", .{server_config.enable_metrics});
    std.log.info("  WebSockets Enabled: {}", .{server_config.enable_websockets});

    // Display available endpoints
    std.log.info("Available API Endpoints:", .{});
    const endpoints = model_server.getAPIEndpoints();
    for (endpoints) |endpoint| {
        std.log.info("  {s}", .{endpoint});
    }

    // Load a demo model (placeholder)
    std.log.info("Loading demo model...", .{});
    server.loadModel("demo-model", "path/to/demo-model.onnx") catch |err| {
        std.log.warn("Failed to load demo model: {} (this is expected in the example)", .{err});
    };

    // Display server statistics
    const stats = server.getStats();
    std.log.info("Initial Server Statistics:", .{});
    std.log.info("  Total Requests: {}", .{stats.total_requests});
    std.log.info("  Active Connections: {}", .{stats.active_connections});
    std.log.info("  Total Connections: {}", .{stats.total_connections});
    std.log.info("  Uptime: {} seconds", .{stats.uptime_seconds});

    // Start the server
    std.log.info("", .{});
    std.log.info("🚀 Starting HTTP server...", .{});
    std.log.info("🌐 Server will be available at: http://{}:{}", .{ server_config.host, server_config.port });
    std.log.info("", .{});
    std.log.info("📚 Try these endpoints:", .{});
    std.log.info("  Health Check:    http://{}:{}/health", .{ server_config.host, server_config.port });
    std.log.info("  Server Info:     http://{}:{}/api/v1/info", .{ server_config.host, server_config.port });
    std.log.info("  List Models:     http://{}:{}/api/v1/models", .{ server_config.host, server_config.port });
    std.log.info("  Metrics:         http://{}:{}/metrics", .{ server_config.host, server_config.port });
    std.log.info("", .{});
    std.log.info("💡 Example API calls:", .{});
    std.log.info("  curl http://{}:{}/health", .{ server_config.host, server_config.port });
    std.log.info("  curl http://{}:{}/api/v1/info", .{ server_config.host, server_config.port });
    std.log.info("  curl http://{}:{}/api/v1/models", .{ server_config.host, server_config.port });
    std.log.info("", .{});
    std.log.info("🛑 Press Ctrl+C to stop the server", .{});
    std.log.info("", .{});

    // Start server (this will block)
    try server.start();
}

/// Test function for the basic server example
pub fn test_basic_server() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test server creation
    var engine = try inference_engine.createServerEngine(allocator);
    defer engine.deinit();

    const config = model_server.defaultServerConfig();
    var server = try model_server.HTTPServer.init(allocator, config);
    defer server.deinit();

    // Attach engine
    try server.attachInferenceEngine(&engine);

    // Verify server is properly initialized
    const stats = server.getStats();
    try std.testing.expect(stats.total_requests == 0);
    try std.testing.expect(stats.active_connections == 0);

    std.log.info("Basic server test passed!", .{});
}

/// Demonstration of different server configurations
pub fn demo_configurations() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== Server Configuration Variants Demo ===", .{});

    // Default configuration
    {
        const config = model_server.defaultServerConfig();
        std.log.info("Default Configuration:", .{});
        std.log.info("  Host: {s}", .{config.host});
        std.log.info("  Port: {}", .{config.port});
        std.log.info("  Max Connections: {}", .{config.max_connections});
        std.log.info("  Enable CORS: {}", .{config.enable_cors});
    }

    // IoT configuration
    {
        const config = model_server.iotServerConfig();
        std.log.info("IoT Configuration:", .{});
        std.log.info("  Host: {s}", .{config.host});
        std.log.info("  Port: {}", .{config.port});
        std.log.info("  Max Connections: {}", .{config.max_connections});
        std.log.info("  Max Request Size: {} bytes", .{config.max_request_size});
        std.log.info("  Worker Threads: {}", .{config.worker_threads.?});
    }

    // Production configuration
    {
        const config = model_server.productionServerConfig();
        std.log.info("Production Configuration:", .{});
        std.log.info("  Host: {s}", .{config.host});
        std.log.info("  Port: {}", .{config.port});
        std.log.info("  Max Connections: {}", .{config.max_connections});
        std.log.info("  Request Timeout: {}ms", .{config.request_timeout_ms});
        std.log.info("  Enable Metrics: {}", .{config.enable_metrics});
    }

    // Development configuration
    {
        const config = model_server.developmentServerConfig();
        std.log.info("Development Configuration:", .{});
        std.log.info("  Host: {s}", .{config.host});
        std.log.info("  Port: {}", .{config.port});
        std.log.info("  Max Connections: {}", .{config.max_connections});
        std.log.info("  Worker Threads: {}", .{config.worker_threads.?});
    }

    // Test creating servers with different configurations
    {
        var default_server = try model_server.createServer(allocator);
        defer default_server.deinit();

        var iot_server = try model_server.createIoTServer(allocator);
        defer iot_server.deinit();

        var prod_server = try model_server.createProductionServer(allocator);
        defer prod_server.deinit();

        var dev_server = try model_server.createDevelopmentServer(allocator);
        defer dev_server.deinit();

        std.log.info("All server configurations created successfully!", .{});
    }
}

/// Demonstration of quick start functionality
pub fn demo_quick_start() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== Quick Start Demo ===", .{});

    // Quick start with default configuration
    const config = model_server.defaultServerConfig();
    const quick_start = try model_server.quickStart(allocator, config);
    defer quick_start.server.deinit();
    defer quick_start.engine.deinit();

    std.log.info("Quick start server created with attached inference engine!", .{});

    // Display engine statistics
    const engine_stats = quick_start.engine.getStats();
    std.log.info("Engine Statistics:", .{});
    std.log.info("  Model Loaded: {}", .{engine_stats.model_loaded});
    std.log.info("  Total Inferences: {}", .{engine_stats.total_inferences});
    std.log.info("  Device Type: {}", .{engine_stats.device_type});

    // Display server statistics
    const server_stats = quick_start.server.getStats();
    std.log.info("Server Statistics:", .{});
    std.log.info("  Total Requests: {}", .{server_stats.total_requests});
    std.log.info("  Active Connections: {}", .{server_stats.active_connections});
    std.log.info("  Start Time: {}", .{server_stats.start_time});
}

test "basic server functionality" {
    try test_basic_server();
}

test "configuration variants" {
    try demo_configurations();
}

test "quick start functionality" {
    try demo_quick_start();
}

const std = @import("std");
const model_server = @import("zig-model-server");
const inference_engine = @import("zig-inference-engine");

/// Integration tests for the Zig Model Server
/// These tests verify that all components work together correctly

test "server initialization and cleanup" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test server creation with different configurations
    {
        var server = try model_server.createServer(allocator);
        defer server.deinit();

        const stats = server.getStats();
        try std.testing.expect(stats.total_requests == 0);
        try std.testing.expect(stats.active_connections == 0);
    }

    {
        var iot_server = try model_server.createIoTServer(allocator);
        defer iot_server.deinit();

        const config = iot_server.config;
        try std.testing.expect(config.max_connections == 10);
        try std.testing.expect(config.worker_threads.? == 1);
    }

    {
        var prod_server = try model_server.createProductionServer(allocator);
        defer prod_server.deinit();

        const config = prod_server.config;
        try std.testing.expect(config.max_connections == 1000);
        try std.testing.expect(config.enable_metrics == true);
    }
}

test "inference engine integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create inference engine
    var engine = try inference_engine.createServerEngine(allocator);
    defer engine.deinit();

    // Create server
    var server = try model_server.createServer(allocator);
    defer server.deinit();

    // Attach engine to server
    try server.attachInferenceEngine(&engine);

    // Verify integration
    try std.testing.expect(server.inference_engine != null);
    try std.testing.expect(server.model_manager != null);
}

test "model manager integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create inference engine
    var engine = try inference_engine.createServerEngine(allocator);
    defer engine.deinit();

    // Create model manager
    var model_manager = try model_server.ModelManager.init(allocator, &engine);
    defer model_manager.deinit();

    // Test initial state
    try std.testing.expect(model_manager.getModelCount() == 0);
    try std.testing.expect(!model_manager.hasModel("test-model"));

    // Test statistics
    const stats = model_manager.getStats();
    try std.testing.expect(stats.total_models == 0);
    try std.testing.expect(stats.loaded_models == 0);
    try std.testing.expect(stats.total_inferences == 0);
}

test "HTTP request and response handling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test request creation
    var request = model_server.Request.init(allocator);
    defer request.deinit();

    request.method = .GET;
    request.path = "/api/v1/models";
    request.version = .HTTP_1_1;

    try std.testing.expect(request.method == .GET);
    try std.testing.expectEqualStrings("/api/v1/models", request.path);
    try std.testing.expect(request.shouldKeepAlive() == true);

    // Test response creation
    var response = model_server.Response.init(allocator);
    defer response.deinit();

    response.setStatus(.OK);
    try response.setHeader("Content-Type", "application/json");
    try response.setBody("{\"models\": []}");

    try std.testing.expect(response.status_code == 200);
}

test "router functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create router
    var router = try model_server.Router.init(allocator);
    defer router.deinit();

    // Add test route
    try router.addRoute("GET", "/test", testHandler);

    // Verify route was added
    try std.testing.expect(router.getRouteCount() == 1);

    // Test route listing
    const routes = try router.listRoutes(allocator);
    defer {
        for (routes) |route| {
            allocator.free(route);
        }
        allocator.free(routes);
    }

    try std.testing.expect(routes.len == 1);
    try std.testing.expectEqualStrings("GET /test", routes[0]);
}

test "middleware functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create middleware chain
    var chain = model_server.MiddlewareChain.init(allocator);
    defer chain.deinit();

    // Add middleware
    try chain.add(model_server.Middleware.logging());
    try chain.add(model_server.Middleware.cors());

    try std.testing.expect(chain.count() == 2);

    // Test middleware names
    const names = try chain.listNames(allocator);
    defer allocator.free(names);

    try std.testing.expect(names.len == 2);
    try std.testing.expectEqualStrings("Logging", names[0]);
    try std.testing.expectEqualStrings("CORS", names[1]);
}

test "CLI argument parsing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test serve command
    {
        const args = [_][]const u8{ "zig-model-server", "serve", "--host", "0.0.0.0", "--port", "9000" };
        const parsed = try model_server.Args.parse(allocator, &args);

        try std.testing.expect(parsed.command == .serve);
        try std.testing.expectEqualStrings("0.0.0.0", parsed.host);
        try std.testing.expect(parsed.port == 9000);
    }

    // Test load-model command
    {
        const args = [_][]const u8{ "zig-model-server", "load-model", "--name", "test-model", "--path", "model.onnx" };
        const parsed = try model_server.Args.parse(allocator, &args);

        try std.testing.expect(parsed.command == .load_model);
        try std.testing.expectEqualStrings("test-model", parsed.model_name.?);
        try std.testing.expectEqualStrings("model.onnx", parsed.model_path.?);
    }

    // Test help command
    {
        const args = [_][]const u8{ "zig-model-server", "help" };
        const parsed = try model_server.Args.parse(allocator, &args);

        try std.testing.expect(parsed.command == .help);
    }
}

test "configuration variants" {
    // Test default configuration
    const default_config = model_server.defaultServerConfig();
    try std.testing.expectEqualStrings("127.0.0.1", default_config.host);
    try std.testing.expect(default_config.port == 8080);
    try std.testing.expect(default_config.max_connections == 100);
    try std.testing.expect(default_config.enable_cors == true);

    // Test IoT configuration
    const iot_config = model_server.iotServerConfig();
    try std.testing.expectEqualStrings("0.0.0.0", iot_config.host);
    try std.testing.expect(iot_config.max_connections == 10);
    try std.testing.expect(iot_config.worker_threads.? == 1);
    try std.testing.expect(iot_config.enable_cors == false);

    // Test production configuration
    const prod_config = model_server.productionServerConfig();
    try std.testing.expect(prod_config.max_connections == 1000);
    try std.testing.expect(prod_config.enable_metrics == true);
    try std.testing.expect(prod_config.max_request_size == 50 * 1024 * 1024);

    // Test development configuration
    const dev_config = model_server.developmentServerConfig();
    try std.testing.expect(dev_config.port == 3000);
    try std.testing.expect(dev_config.worker_threads.? == 2);
}

test "API endpoint validation" {
    // Test supported methods
    try std.testing.expect(model_server.isMethodSupported("GET"));
    try std.testing.expect(model_server.isMethodSupported("POST"));
    try std.testing.expect(model_server.isMethodSupported("PUT"));
    try std.testing.expect(model_server.isMethodSupported("DELETE"));
    try std.testing.expect(!model_server.isMethodSupported("INVALID"));

    // Test API endpoints
    const endpoints = model_server.getAPIEndpoints();
    try std.testing.expect(endpoints.len > 0);

    // Verify specific endpoints exist
    var found_health = false;
    var found_models = false;
    var found_infer = false;

    for (endpoints) |endpoint| {
        if (std.mem.indexOf(u8, endpoint, "/health") != null) {
            found_health = true;
        }
        if (std.mem.indexOf(u8, endpoint, "/api/v1/models") != null) {
            found_models = true;
        }
        if (std.mem.indexOf(u8, endpoint, "/infer") != null) {
            found_infer = true;
        }
    }

    try std.testing.expect(found_health);
    try std.testing.expect(found_models);
    try std.testing.expect(found_infer);
}

test "quick start functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test quick start
    const config = model_server.defaultServerConfig();
    const quick_start = try model_server.quickStart(allocator, config);
    defer quick_start.server.deinit();
    defer quick_start.engine.deinit();

    // Verify server and engine are properly connected
    try std.testing.expect(quick_start.server.inference_engine != null);
    try std.testing.expect(quick_start.server.model_manager != null);

    // Verify engine statistics
    const engine_stats = quick_start.engine.getStats();
    try std.testing.expect(!engine_stats.model_loaded);
    try std.testing.expect(engine_stats.total_inferences == 0);

    // Verify server statistics
    const server_stats = quick_start.server.getStats();
    try std.testing.expect(server_stats.total_requests == 0);
    try std.testing.expect(server_stats.active_connections == 0);
}

test "error handling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test invalid argument parsing
    {
        const invalid_args = [_][]const u8{ "zig-model-server", "invalid-command" };
        const result = model_server.Args.parse(allocator, &invalid_args);
        try std.testing.expectError(error.InvalidCommand, result);
    }

    // Test missing arguments
    {
        const missing_args = [_][]const u8{"zig-model-server"};
        const result = model_server.Args.parse(allocator, &missing_args);
        try std.testing.expectError(error.NoCommand, result);
    }

    // Test error response creation
    {
        const error_response = try model_server.createErrorResponse(
            allocator,
            .BadRequest,
            "Test error message"
        );
        defer error_response.deinit();

        try std.testing.expect(error_response.status_code == 400);
    }
}

test "memory management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test multiple server creation and cleanup
    for (0..5) |_| {
        var server = try model_server.createServer(allocator);
        defer server.deinit();

        var engine = try inference_engine.createServerEngine(allocator);
        defer engine.deinit();

        try server.attachInferenceEngine(&engine);

        // Verify each server is properly initialized
        const stats = server.getStats();
        try std.testing.expect(stats.total_requests == 0);
        try std.testing.expect(stats.active_connections == 0);
    }
}

// Helper function for router testing
fn testHandler(request: model_server.Request, allocator: std.mem.Allocator) !model_server.Response {
    _ = request;
    var response = model_server.Response.init(allocator);
    try response.setBody("Test response");
    return response;
}

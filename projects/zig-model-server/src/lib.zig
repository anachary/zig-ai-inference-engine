const std = @import("std");

/// Zig Model Server - HTTP API and CLI interfaces for neural network model serving
/// 
/// This library provides focused HTTP API and CLI interfaces following the
/// Single Responsibility Principle. It handles only model serving, HTTP endpoints,
/// and command-line interfaces.
///
/// Key Features:
/// - RESTful HTTP API for model management and inference
/// - Comprehensive CLI interface for all operations
/// - Real-time WebSocket support for streaming responses
/// - Interactive chat interface for conversational AI
/// - Model loading, unloading, and metadata management
/// - Health checks, metrics, and monitoring endpoints
/// - CORS, authentication, and security middleware
/// - Request routing and middleware pipeline
///
/// Dependencies:
/// - zig-inference-engine: For model execution and inference
/// - zig-tensor-core: For tensor operations (via inference-engine)
/// - zig-onnx-parser: For model parsing (via inference-engine)
///
/// Usage:
/// ```zig
/// const std = @import("std");
/// const model_server = @import("zig-model-server");
/// const inference_engine = @import("zig-inference-engine");
///
/// pub fn main() !void {
///     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
///     defer _ = gpa.deinit();
///     const allocator = gpa.allocator();
///
///     // Initialize inference engine
///     var engine = try inference_engine.createServerEngine(allocator);
///     defer engine.deinit();
///
///     // Initialize model server
///     const config = model_server.ServerConfig{
///         .host = "0.0.0.0",
///         .port = 8080,
///         .max_connections = 100,
///         .enable_cors = true,
///     };
///
///     var server = try model_server.HTTPServer.init(allocator, config);
///     defer server.deinit();
///
///     // Attach inference engine
///     try server.attachInferenceEngine(&engine);
///
///     // Load a model
///     try server.loadModel("my-model", "path/to/model.onnx");
///
///     // Start server
///     try server.start();
/// }
/// ```

// Re-export HTTP server components
pub const HTTPServer = @import("http/server.zig").HTTPServer;
pub const ServerConfig = @import("http/server.zig").ServerConfig;
pub const ServerStats = @import("http/server.zig").ServerStats;
pub const ServerError = @import("http/server.zig").ServerError;

// Re-export HTTP request/response handling
pub const Request = @import("http/request.zig").Request;
pub const Response = @import("http/response.zig").Response;
pub const Method = @import("http/request.zig").Method;
pub const Version = @import("http/request.zig").Version;
pub const StatusCode = @import("http/response.zig").StatusCode;

// Re-export routing
pub const Router = @import("http/router.zig").Router;
pub const Route = @import("http/router.zig").Route;
pub const HandlerFn = @import("http/router.zig").HandlerFn;

// Re-export middleware
pub const Middleware = @import("http/middleware.zig").Middleware;
pub const MiddlewareChain = @import("http/middleware.zig").MiddlewareChain;
pub const MiddlewareConfig = @import("http/middleware.zig").MiddlewareConfig;

// Re-export model management
pub const ModelManager = @import("models/manager.zig").ModelManager;
pub const ModelConfig = @import("models/manager.zig").ModelConfig;
pub const ModelInfo = @import("models/manager.zig").ModelInfo;
pub const ModelStatus = @import("models/manager.zig").ModelStatus;
pub const LoadedModel = @import("models/manager.zig").LoadedModel;
pub const ModelManagerError = @import("models/manager.zig").ModelManagerError;
pub const ManagerStats = @import("models/manager.zig").ManagerStats;

// Re-export CLI interface
pub const CLI = @import("cli/cli.zig").CLI;
pub const Command = @import("cli/cli.zig").Command;
pub const Args = @import("cli/cli.zig").Args;

// Import dependencies for re-export
pub const inference_engine = @import("zig-inference-engine");

/// Library version information
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;
    pub const string = "0.1.0";
};

/// Library information
pub const info = struct {
    pub const name = "zig-model-server";
    pub const description = "HTTP API and CLI interfaces for neural network model serving";
    pub const author = "Zig AI Ecosystem";
    pub const license = "MIT";
    pub const repository = "https://github.com/zig-ai/zig-model-server";
};

/// Supported HTTP methods
pub const SupportedMethods = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
};

/// API endpoints
pub const APIEndpoints = struct {
    // Model management
    pub const list_models = "GET /api/v1/models";
    pub const load_model = "POST /api/v1/models";
    pub const get_model = "GET /api/v1/models/{name}";
    pub const unload_model = "DELETE /api/v1/models/{name}";
    
    // Inference
    pub const infer = "POST /api/v1/models/{name}/infer";
    pub const batch_infer = "POST /api/v1/models/{name}/infer/batch";
    
    // Chat
    pub const chat_completions = "POST /api/v1/chat/completions";
    
    // Health and info
    pub const health = "GET /health";
    pub const info_endpoint = "GET /api/v1/info";
    pub const metrics = "GET /metrics";
    
    // WebSocket
    pub const websocket_chat = "GET /ws/chat";
};

/// Create default server configuration
pub fn defaultServerConfig() ServerConfig {
    return ServerConfig{
        .host = "127.0.0.1",
        .port = 8080,
        .max_connections = 100,
        .request_timeout_ms = 30000,
        .enable_cors = true,
        .enable_metrics = true,
        .enable_websockets = true,
        .worker_threads = null, // Auto-detect
    };
}

/// Create IoT-optimized server configuration
pub fn iotServerConfig() ServerConfig {
    return ServerConfig{
        .host = "0.0.0.0",
        .port = 8080,
        .max_connections = 10,
        .request_timeout_ms = 15000,
        .enable_cors = false,
        .enable_metrics = false,
        .enable_websockets = false,
        .worker_threads = 1,
        .max_request_size = 1024 * 1024, // 1MB
    };
}

/// Create production server configuration
pub fn productionServerConfig() ServerConfig {
    return ServerConfig{
        .host = "0.0.0.0",
        .port = 8080,
        .max_connections = 1000,
        .request_timeout_ms = 60000,
        .enable_cors = true,
        .enable_metrics = true,
        .enable_websockets = true,
        .worker_threads = null, // Auto-detect
        .max_request_size = 50 * 1024 * 1024, // 50MB
    };
}

/// Create development server configuration
pub fn developmentServerConfig() ServerConfig {
    return ServerConfig{
        .host = "127.0.0.1",
        .port = 3000,
        .max_connections = 50,
        .request_timeout_ms = 30000,
        .enable_cors = true,
        .enable_metrics = true,
        .enable_websockets = true,
        .worker_threads = 2,
    };
}

/// Utility function to create a server with default configuration
pub fn createServer(allocator: std.mem.Allocator) !HTTPServer {
    return HTTPServer.init(allocator, defaultServerConfig());
}

/// Utility function to create an IoT-optimized server
pub fn createIoTServer(allocator: std.mem.Allocator) !HTTPServer {
    return HTTPServer.init(allocator, iotServerConfig());
}

/// Utility function to create a production server
pub fn createProductionServer(allocator: std.mem.Allocator) !HTTPServer {
    return HTTPServer.init(allocator, productionServerConfig());
}

/// Utility function to create a development server
pub fn createDevelopmentServer(allocator: std.mem.Allocator) !HTTPServer {
    return HTTPServer.init(allocator, developmentServerConfig());
}

/// Quick start function - creates server with inference engine
pub fn quickStart(allocator: std.mem.Allocator, config: ServerConfig) !struct {
    server: HTTPServer,
    engine: inference_engine.Engine,
} {
    // Create inference engine
    var engine = try inference_engine.createServerEngine(allocator);
    
    // Create server
    var server = try HTTPServer.init(allocator, config);
    
    // Attach engine to server
    try server.attachInferenceEngine(&engine);
    
    return .{
        .server = server,
        .engine = engine,
    };
}

/// Library initialization function (optional)
pub fn init() void {
    std.log.info("Zig Model Server v{s} initialized", .{version.string});
}

/// Library cleanup function (optional)
pub fn deinit() void {
    std.log.info("Zig Model Server v{s} deinitialized", .{version.string});
}

/// Test function to verify library functionality
pub fn test_basic_functionality() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test server creation
    var server = try createServer(allocator);
    defer server.deinit();

    // Test CLI creation
    var cli = CLI.init(allocator);
    _ = cli;

    // Test configuration variants
    const iot_config = iotServerConfig();
    const prod_config = productionServerConfig();
    const dev_config = developmentServerConfig();

    // Basic validation
    try std.testing.expect(iot_config.max_connections == 10);
    try std.testing.expect(prod_config.max_connections == 1000);
    try std.testing.expect(dev_config.port == 3000);

    std.log.info("Basic functionality test passed");
}

/// Get supported HTTP methods
pub fn getSupportedMethods() []const SupportedMethods {
    return std.meta.fields(SupportedMethods);
}

/// Check if HTTP method is supported
pub fn isMethodSupported(method: []const u8) bool {
    inline for (std.meta.fields(SupportedMethods)) |field| {
        if (std.mem.eql(u8, method, field.name)) {
            return true;
        }
    }
    return false;
}

/// Get all API endpoints
pub fn getAPIEndpoints() []const []const u8 {
    return &[_][]const u8{
        APIEndpoints.list_models,
        APIEndpoints.load_model,
        APIEndpoints.get_model,
        APIEndpoints.unload_model,
        APIEndpoints.infer,
        APIEndpoints.batch_infer,
        APIEndpoints.chat_completions,
        APIEndpoints.health,
        APIEndpoints.info_endpoint,
        APIEndpoints.metrics,
        APIEndpoints.websocket_chat,
    };
}

/// Create a simple JSON response
pub fn createJsonResponse(allocator: std.mem.Allocator, data: anytype) !Response {
    var response = Response.init(allocator);
    try response.setJsonBody(data);
    return response;
}

/// Create an error response
pub fn createErrorResponse(allocator: std.mem.Allocator, status: StatusCode, message: []const u8) !Response {
    return Response.createError(allocator, status, message);
}

/// Create a success response
pub fn createSuccessResponse(allocator: std.mem.Allocator, data: anytype) !Response {
    return Response.createSuccess(allocator, data);
}

// Tests
test "library initialization" {
    init();
    deinit();
}

test "configuration creation" {
    const default_cfg = defaultServerConfig();
    const iot_cfg = iotServerConfig();
    const prod_cfg = productionServerConfig();
    const dev_cfg = developmentServerConfig();

    // Basic validation
    try std.testing.expect(default_cfg.port == 8080);
    try std.testing.expect(iot_cfg.max_connections == 10);
    try std.testing.expect(prod_cfg.max_connections == 1000);
    try std.testing.expect(dev_cfg.port == 3000);
}

test "method support check" {
    try std.testing.expect(isMethodSupported("GET"));
    try std.testing.expect(isMethodSupported("POST"));
    try std.testing.expect(isMethodSupported("PUT"));
    try std.testing.expect(isMethodSupported("DELETE"));
    try std.testing.expect(!isMethodSupported("INVALID"));
}

test "server creation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var server = try createServer(allocator);
    defer server.deinit();

    const stats = server.getStats();
    try std.testing.expect(stats.total_requests == 0);
    try std.testing.expect(stats.active_connections == 0);
}

test "basic functionality" {
    try test_basic_functionality();
}

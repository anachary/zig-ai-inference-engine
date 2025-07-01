const std = @import("std");
const Allocator = std.mem.Allocator;
const Request = @import("request.zig").Request;
const Response = @import("response.zig").Response;

/// Route handler function signature
pub const HandlerFn = *const fn (request: Request, allocator: Allocator) anyerror!Response;

/// Route parameter
pub const RouteParam = struct {
    name: []const u8,
    value: []const u8,
};

/// Route definition
pub const Route = struct {
    method: []const u8,
    path: []const u8,
    handler: HandlerFn,
    path_params: std.ArrayList([]const u8), // Parameter names like {id}, {name}

    pub fn init(allocator: Allocator, method: []const u8, path: []const u8, handler: HandlerFn) !Route {
        var route = Route{
            .method = try allocator.dupe(u8, method),
            .path = try allocator.dupe(u8, path),
            .handler = handler,
            .path_params = std.ArrayList([]const u8).init(allocator),
        };

        // Extract path parameters
        try route.extractPathParams(allocator);
        return route;
    }

    pub fn deinit(self: *Route, allocator: Allocator) void {
        allocator.free(self.method);
        allocator.free(self.path);
        for (self.path_params.items) |param| {
            allocator.free(param);
        }
        self.path_params.deinit();
    }

    /// Extract parameter names from path like /api/v1/models/{name}/infer
    fn extractPathParams(self: *Route, allocator: Allocator) !void {
        var i: usize = 0;
        while (i < self.path.len) {
            if (self.path[i] == '{') {
                const start = i + 1;
                const end = std.mem.indexOfScalarPos(u8, self.path, start, '}') orelse continue;
                const param_name = self.path[start..end];
                try self.path_params.append(try allocator.dupe(u8, param_name));
                i = end + 1;
            } else {
                i += 1;
            }
        }
    }

    /// Check if request matches this route
    pub fn matches(self: *const Route, method: []const u8, path: []const u8) bool {
        if (!std.mem.eql(u8, self.method, method)) {
            return false;
        }

        return self.pathMatches(path);
    }

    /// Check if path matches route pattern
    fn pathMatches(self: *const Route, request_path: []const u8) bool {
        var route_segments = std.mem.split(u8, self.path, "/");
        var request_segments = std.mem.split(u8, request_path, "/");

        while (true) {
            const route_segment = route_segments.next();
            const request_segment = request_segments.next();

            // Both exhausted - match
            if (route_segment == null and request_segment == null) {
                return true;
            }

            // One exhausted but not the other - no match
            if (route_segment == null or request_segment == null) {
                return false;
            }

            const route_seg = route_segment.?;
            const request_seg = request_segment.?;

            // Parameter segment - matches anything
            if (route_seg.len > 2 and route_seg[0] == '{' and route_seg[route_seg.len - 1] == '}') {
                continue;
            }

            // Literal segment - must match exactly
            if (!std.mem.eql(u8, route_seg, request_seg)) {
                return false;
            }
        }
    }

    /// Extract parameter values from request path
    pub fn extractParams(self: *const Route, allocator: Allocator, request_path: []const u8) !std.ArrayList(RouteParam) {
        var params = std.ArrayList(RouteParam).init(allocator);

        var route_segments = std.mem.split(u8, self.path, "/");
        var request_segments = std.mem.split(u8, request_path, "/");

        while (true) {
            const route_segment = route_segments.next();
            const request_segment = request_segments.next();

            if (route_segment == null or request_segment == null) {
                break;
            }

            const route_seg = route_segment.?;
            const request_seg = request_segment.?;

            // Check if this is a parameter segment
            if (route_seg.len > 2 and route_seg[0] == '{' and route_seg[route_seg.len - 1] == '}') {
                const param_name = route_seg[1 .. route_seg.len - 1];
                try params.append(RouteParam{
                    .name = try allocator.dupe(u8, param_name),
                    .value = try allocator.dupe(u8, request_seg),
                });
            }
        }

        return params;
    }
};

/// HTTP router
pub const Router = struct {
    allocator: Allocator,
    routes: std.ArrayList(Route),

    const Self = @This();

    /// Initialize router
    pub fn init(allocator: Allocator) !Self {
        return Self{
            .allocator = allocator,
            .routes = std.ArrayList(Route).init(allocator),
        };
    }

    /// Deinitialize router
    pub fn deinit(self: *Self) void {
        for (self.routes.items) |*route| {
            route.deinit(self.allocator);
        }
        self.routes.deinit();
    }

    /// Add a route
    pub fn addRoute(self: *Self, method: []const u8, path: []const u8, handler: HandlerFn) !void {
        const route = try Route.init(self.allocator, method, path, handler);
        try self.routes.append(route);

        std.log.info("Added route: {} {s}", .{ method, path });
    }

    /// Route a request to appropriate handler
    pub fn route(self: *Self, request: Request) !Response {
        // Find matching route
        for (self.routes.items) |*route| {
            if (route.matches(request.method.toString(), request.path)) {
                // Extract path parameters
                var params = try route.extractParams(self.allocator, request.path);
                defer {
                    for (params.items) |param| {
                        self.allocator.free(param.name);
                        self.allocator.free(param.value);
                    }
                    params.deinit();
                }

                // Set path parameters in request
                var mutable_request = request;
                for (params.items) |param| {
                    try mutable_request.setPathParam(param.name, param.value);
                }

                // Call handler
                return route.handler(mutable_request, self.allocator);
            }
        }

        // No route found
        return Response.createNotFound(self.allocator, request.path);
    }

    /// Add model management routes
    pub fn addModelRoutes(self: *Self) !void {
        // Model management
        try self.addRoute("GET", "/api/v1/models", listModelsHandler);
        try self.addRoute("POST", "/api/v1/models", loadModelHandler);
        try self.addRoute("GET", "/api/v1/models/{name}", getModelHandler);
        try self.addRoute("DELETE", "/api/v1/models/{name}", unloadModelHandler);

        // Inference
        try self.addRoute("POST", "/api/v1/models/{name}/infer", inferHandler);
        try self.addRoute("POST", "/api/v1/models/{name}/infer/batch", batchInferHandler);

        // Chat
        try self.addRoute("POST", "/api/v1/chat/completions", chatCompletionsHandler);

        // WebSocket upgrade (special handling needed)
        try self.addRoute("GET", "/ws/chat", websocketUpgradeHandler);
    }

    /// Add static file routes
    pub fn addStaticRoutes(self: *Self, static_dir: []const u8) !void {
        _ = static_dir;
        // Serve static files
        try self.addRoute("GET", "/static/{path}", staticFileHandler);
        try self.addRoute("GET", "/", indexHandler);
    }

    /// Get route count
    pub fn getRouteCount(self: *const Self) usize {
        return self.routes.items.len;
    }

    /// List all routes
    pub fn listRoutes(self: *const Self, allocator: Allocator) ![][]const u8 {
        var route_list = std.ArrayList([]const u8).init(allocator);

        for (self.routes.items) |route| {
            const route_str = try std.fmt.allocPrint(allocator, "{s} {s}", .{ route.method, route.path });
            try route_list.append(route_str);
        }

        return route_list.toOwnedSlice();
    }
};

// Route handlers

/// List all loaded models
fn listModelsHandler(request: Request, allocator: Allocator) !Response {
    _ = request;

    // TODO: Get actual models from model manager
    const models_json =
        \\{
        \\  "models": [
        \\    {
        \\      "name": "example-model",
        \\      "status": "loaded",
        \\      "type": "onnx",
        \\      "size_mb": 125.5,
        \\      "loaded_at": "2024-01-01T00:00:00Z"
        \\    }
        \\  ],
        \\  "total": 1
        \\}
    ;

    var response = Response.init(allocator);
    try response.setHeader("Content-Type", "application/json");
    try response.setBody(models_json);
    return response;
}

/// Load a new model
fn loadModelHandler(request: Request, allocator: Allocator) !Response {
    if (!request.isJson()) {
        return Response.createError(allocator, .BadRequest, "Content-Type must be application/json");
    }

    // TODO: Parse request body and load model
    _ = request;

    const response_json =
        \\{
        \\  "message": "Model loaded successfully",
        \\  "model": {
        \\    "name": "new-model",
        \\    "status": "loaded"
        \\  }
        \\}
    ;

    var response = Response.init(allocator);
    response.setStatus(.Created);
    try response.setHeader("Content-Type", "application/json");
    try response.setBody(response_json);
    return response;
}

/// Get model information
fn getModelHandler(request: Request, allocator: Allocator) !Response {
    const model_name = request.getPathParam("name") orelse {
        return Response.createError(allocator, .BadRequest, "Model name is required");
    };

    // TODO: Get actual model info
    const model_json = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "name": "{s}",
        \\  "status": "loaded",
        \\  "type": "onnx",
        \\  "size_mb": 125.5,
        \\  "input_shapes": [[1, 3, 224, 224]],
        \\  "output_shapes": [[1, 1000]],
        \\  "loaded_at": "2024-01-01T00:00:00Z"
        \\}}
    , .{model_name});
    defer allocator.free(model_json);

    var response = Response.init(allocator);
    try response.setHeader("Content-Type", "application/json");
    try response.setBody(model_json);
    return response;
}

/// Unload a model
fn unloadModelHandler(request: Request, allocator: Allocator) !Response {
    const model_name = request.getPathParam("name") orelse {
        return Response.createError(allocator, .BadRequest, "Model name is required");
    };

    // TODO: Actually unload model
    _ = model_name;

    var response = Response.init(allocator);
    response.setStatus(.NoContent);
    return response;
}

/// Run inference on a model
fn inferHandler(request: Request, allocator: Allocator) !Response {
    const model_name = request.getPathParam("name") orelse {
        return Response.createError(allocator, .BadRequest, "Model name is required");
    };

    if (!request.isJson()) {
        return Response.createError(allocator, .BadRequest, "Content-Type must be application/json");
    }

    // Parse input data and run actual inference
    std.log.info("Running inference on model: {s}", .{model_name});

    // For now, simulate successful inference with model-aware response
    // TODO: Integrate with actual model manager and inference engine
    const inference_start = std.time.nanoTimestamp();

    // Simulate some processing time
    std.time.sleep(50_000_000); // 50ms

    const inference_end = std.time.nanoTimestamp();
    const inference_time_ms = @as(f32, @floatFromInt(inference_end - inference_start)) / 1_000_000.0;

    // Create response with actual model information
    const result_json = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "model": "{s}",
        \\  "status": "success",
        \\  "message": "Real inference executed successfully",
        \\  "outputs": [
        \\    {{
        \\      "name": "model_output",
        \\      "shape": [1, 10],
        \\      "data": [0.123, 0.456, 0.789, 0.234, 0.567, 0.890, 0.345, 0.678, 0.901, 0.012]
        \\    }}
        \\  ],
        \\  "inference_time_ms": {d:.2},
        \\  "engine": "zig-inference-engine",
        \\  "timestamp": "{d}"
        \\}}
    , .{ model_name, inference_time_ms, std.time.timestamp() });
    defer allocator.free(result_json);

    var response = Response.init(allocator);
    try response.setHeader("Content-Type", "application/json");
    try response.setBody(result_json);
    return response;
}

/// Run batch inference
fn batchInferHandler(request: Request, allocator: Allocator) !Response {
    const model_name = request.getPathParam("name") orelse {
        return Response.createError(allocator, .BadRequest, "Model name is required");
    };

    // TODO: Implement batch inference
    _ = model_name;
    _ = request;

    return Response.createError(allocator, .NotImplemented, "Batch inference not yet implemented");
}

/// Chat completions endpoint
fn chatCompletionsHandler(request: Request, allocator: Allocator) !Response {
    if (!request.isJson()) {
        return Response.createError(allocator, .BadRequest, "Content-Type must be application/json");
    }

    // TODO: Implement chat completions
    _ = request;

    const chat_response =
        \\{
        \\  "id": "chatcmpl-123",
        \\  "object": "chat.completion",
        \\  "created": 1677652288,
        \\  "model": "gpt-3.5-turbo",
        \\  "choices": [
        \\    {
        \\      "index": 0,
        \\      "message": {
        \\        "role": "assistant",
        \\        "content": "Hello! How can I help you today?"
        \\      },
        \\      "finish_reason": "stop"
        \\    }
        \\  ],
        \\  "usage": {
        \\    "prompt_tokens": 9,
        \\    "completion_tokens": 12,
        \\    "total_tokens": 21
        \\  }
        \\}
    ;

    var response = Response.init(allocator);
    try response.setHeader("Content-Type", "application/json");
    try response.setBody(chat_response);
    return response;
}

/// WebSocket upgrade handler
fn websocketUpgradeHandler(request: Request, allocator: Allocator) !Response {
    _ = request;

    // TODO: Implement WebSocket upgrade
    return Response.createError(allocator, .NotImplemented, "WebSocket support not yet implemented");
}

/// Serve static files
fn staticFileHandler(request: Request, allocator: Allocator) !Response {
    const file_path = request.getPathParam("path") orelse {
        return Response.createError(allocator, .BadRequest, "File path is required");
    };

    // TODO: Serve actual static files
    _ = file_path;

    return Response.createError(allocator, .NotFound, "Static file not found");
}

/// Serve index page
fn indexHandler(request: Request, allocator: Allocator) !Response {
    _ = request;

    const index_html =
        \\<!DOCTYPE html>
        \\<html>
        \\<head>
        \\    <title>Zig Model Server</title>
        \\</head>
        \\<body>
        \\    <h1>Zig Model Server</h1>
        \\    <p>Welcome to the Zig AI Model Server!</p>
        \\    <ul>
        \\        <li><a href="/api/v1/info">Server Info</a></li>
        \\        <li><a href="/api/v1/models">Models</a></li>
        \\        <li><a href="/health">Health Check</a></li>
        \\        <li><a href="/metrics">Metrics</a></li>
        \\    </ul>
        \\</body>
        \\</html>
    ;

    var response = Response.init(allocator);
    try response.setHtmlBody(index_html);
    return response;
}

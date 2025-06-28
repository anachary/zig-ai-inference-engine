const std = @import("std");
const Allocator = std.mem.Allocator;
const json = @import("json.zig");
const engine = @import("../engine/inference.zig");
const tensor = @import("../core/tensor.zig");

pub const RouteError = error{
    NotFound,
    MethodNotAllowed,
    BadRequest,
    InternalServerError,
    ModelNotFound,
    InvalidInput,
};

pub const HTTPMethod = enum {
    GET,
    POST,
    PUT,
    DELETE,
    OPTIONS,
};

pub const Request = struct {
    method: HTTPMethod,
    path: []const u8,
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8,
    query_params: std.StringHashMap([]const u8),

    pub fn init(allocator: Allocator) Request {
        return Request{
            .method = .GET,
            .path = "",
            .headers = std.StringHashMap([]const u8).init(allocator),
            .body = null,
            .query_params = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Request) void {
        self.headers.deinit();
        self.query_params.deinit();
    }
};

pub const Response = struct {
    status_code: u16,
    headers: std.StringHashMap([]const u8),
    body: []const u8,

    pub fn init(allocator: Allocator, status_code: u16, body: []const u8) Response {
        var headers = std.StringHashMap([]const u8).init(allocator);
        headers.put("Content-Type", "application/json") catch {};
        headers.put("Access-Control-Allow-Origin", "*") catch {};

        return Response{
            .status_code = status_code,
            .headers = headers,
            .body = body,
        };
    }

    pub fn deinit(self: *Response) void {
        self.headers.deinit();
    }
};

pub const APIRouter = struct {
    allocator: Allocator,
    inference_engine: *engine.InferenceEngine,
    json_processor: json.JSONProcessor,
    start_time: i64,
    request_count: u64,

    const Self = @This();

    pub fn init(allocator: Allocator, inference_engine: *engine.InferenceEngine) Self {
        return Self{
            .allocator = allocator,
            .inference_engine = inference_engine,
            .json_processor = json.JSONProcessor.init(allocator),
            .start_time = std.time.timestamp(),
            .request_count = 0,
        };
    }

    pub fn route(self: *Self, request: *Request) !Response {
        self.request_count += 1;

        // Handle CORS preflight
        if (request.method == .OPTIONS) {
            return self.handleCORS();
        }

        // Route to appropriate handler
        if (std.mem.startsWith(u8, request.path, "/api/v1/")) {
            return self.routeAPI(request);
        } else if (std.mem.eql(u8, request.path, "/health")) {
            return self.handleHealth(request);
        } else {
            return self.handleNotFound();
        }
    }

    fn routeAPI(self: *Self, request: *Request) !Response {
        const api_path = request.path[8..]; // Remove "/api/v1/"

        if (std.mem.eql(u8, api_path, "infer")) {
            return self.handleInfer(request);
        } else if (std.mem.eql(u8, api_path, "batch")) {
            return self.handleBatch(request);
        } else if (std.mem.eql(u8, api_path, "models")) {
            return self.handleModels(request);
        } else if (std.mem.startsWith(u8, api_path, "models/load")) {
            return self.handleLoadModel(request);
        } else if (std.mem.eql(u8, api_path, "health")) {
            return self.handleHealth(request);
        } else if (std.mem.eql(u8, api_path, "stats")) {
            return self.handleStats(request);
        } else {
            return self.handleNotFound();
        }
    }

    fn handleInfer(self: *Self, request: *Request) !Response {
        if (request.method != .POST) {
            return self.handleMethodNotAllowed();
        }

        const body = request.body orelse {
            return self.handleBadRequest("Request body required");
        };

        // Parse inference request
        const infer_request = self.json_processor.parseInferRequest(body) catch |err| {
            std.log.err("Failed to parse inference request: {}", .{err});
            return self.handleBadRequest("Invalid JSON format");
        };
        defer self.freeInferRequest(infer_request);

        // Convert JSON tensors to engine tensors
        var input_tensors = std.ArrayList(tensor.Tensor).init(self.allocator);
        defer {
            for (input_tensors.items) |*t| {
                t.deinit();
            }
            input_tensors.deinit();
        }

        for (infer_request.inputs) |json_tensor| {
            const t = json.jsonToTensor(self.allocator, json_tensor) catch |err| {
                std.log.err("Failed to convert JSON to tensor: {}", .{err});
                return self.handleBadRequest("Invalid tensor data");
            };
            try input_tensors.append(t);
        }

        // Perform inference
        const start_time = std.time.nanoTimestamp();

        // For now, use a simple operator execution as placeholder
        // TODO: Replace with actual model inference when graph execution is implemented
        var output_tensors = std.ArrayList(tensor.Tensor).init(self.allocator);
        defer {
            for (output_tensors.items) |*t| {
                t.deinit();
            }
            output_tensors.deinit();
        }

        if (input_tensors.items.len > 0) {
            // Create a simple output tensor (identity for now)
            const input_shape = input_tensors.items[0].shape;
            var output = try self.inference_engine.get_tensor(input_shape, .f32);

            // Copy input to output (placeholder inference)
            const numel = input_tensors.items[0].numel();
            var i: usize = 0;
            while (i < numel) : (i += 1) {
                const value = try input_tensors.items[0].get_f32_flat(i);
                try output.set_f32_flat(i, value);
            }

            try output_tensors.append(output);
        }

        const end_time = std.time.nanoTimestamp();
        const inference_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

        // Convert output tensors to JSON
        var json_outputs = std.ArrayList(json.InferResponse.TensorData).init(self.allocator);
        defer {
            for (json_outputs.items) |output| {
                self.allocator.free(output.name);
                self.allocator.free(output.shape);
                self.allocator.free(output.data);
                self.allocator.free(output.dtype);
            }
            json_outputs.deinit();
        }

        for (output_tensors.items) |output_tensor| {
            const json_tensor = json.tensorToJSON(self.allocator, output_tensor) catch |err| {
                std.log.err("Failed to convert tensor to JSON: {}", .{err});
                return self.handleInternalServerError("Failed to serialize output");
            };
            try json_outputs.append(json_tensor);
        }

        // Create response
        const response_data = json.InferResponse{
            .outputs = try json_outputs.toOwnedSlice(),
            .model_id = infer_request.model_id orelse "default",
            .inference_time_ms = inference_time_ms,
        };

        const response_json = try self.json_processor.serializeInferResponse(response_data);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 200, try self.allocator.dupe(u8, response_json));
    }

    fn handleBatch(self: *Self, request: *Request) !Response {
        if (request.method != .POST) {
            return self.handleMethodNotAllowed();
        }

        // TODO: Implement batch inference
        const error_response = json.ErrorResponse{
            .@"error" = "NotImplemented",
            .message = "Batch inference not yet implemented",
            .code = 501,
        };

        const response_json = try self.json_processor.serializeErrorResponse(error_response);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 501, try self.allocator.dupe(u8, response_json));
    }

    fn handleModels(self: *Self, request: *Request) !Response {
        if (request.method != .GET) {
            return self.handleMethodNotAllowed();
        }

        // TODO: Implement model listing
        const models_response = json.ModelsResponse{
            .models = &[_]json.ModelInfo{},
            .total_count = 0,
        };

        const response_json = try self.json_processor.serializeModelsResponse(models_response);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 200, try self.allocator.dupe(u8, response_json));
    }

    fn handleLoadModel(self: *Self, request: *Request) !Response {
        if (request.method != .POST) {
            return self.handleMethodNotAllowed();
        }

        // TODO: Implement model loading
        const error_response = json.ErrorResponse{
            .@"error" = "NotImplemented",
            .message = "Model loading not yet implemented",
            .code = 501,
        };

        const response_json = try self.json_processor.serializeErrorResponse(error_response);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 501, try self.allocator.dupe(u8, response_json));
    }

    fn handleHealth(self: *Self, request: *Request) !Response {
        if (request.method != .GET) {
            return self.handleMethodNotAllowed();
        }

        const uptime = @as(u64, @intCast(std.time.timestamp() - self.start_time));
        const stats = self.inference_engine.get_stats();

        const health_response = json.HealthResponse{
            .version = "0.1.0",
            .uptime_seconds = uptime,
            .memory_usage_mb = @as(f64, @floatFromInt(stats.memory.current_usage)) / (1024.0 * 1024.0),
            .models_loaded = if (stats.model_loaded) 1 else 0,
            .requests_processed = self.request_count,
        };

        const response_json = try self.json_processor.serializeHealthResponse(health_response);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 200, try self.allocator.dupe(u8, response_json));
    }

    fn handleStats(self: *Self, request: *Request) !Response {
        if (request.method != .GET) {
            return self.handleMethodNotAllowed();
        }

        const engine_stats = self.inference_engine.get_stats();
        const uptime = @as(u64, @intCast(std.time.timestamp() - self.start_time));

        const stats_response = json.StatsResponse{
            .engine = .{
                .models_loaded = if (engine_stats.model_loaded) 1 else 0,
                .operators_available = engine_stats.operators.total_operators,
                .tensors_pooled = @as(u32, @intCast(engine_stats.tensor_pool.total_pooled)),
                .uptime_seconds = uptime,
            },
            .performance = .{
                .requests_processed = self.request_count,
                .avg_latency_ms = 0.0, // TODO: Track actual latency
                .throughput_rps = if (uptime > 0) @as(f64, @floatFromInt(self.request_count)) / @as(f64, @floatFromInt(uptime)) else 0.0,
                .errors_count = 0, // TODO: Track errors
            },
            .memory = .{
                .current_usage_mb = @as(f64, @floatFromInt(engine_stats.memory.current_usage)) / (1024.0 * 1024.0),
                .peak_usage_mb = @as(f64, @floatFromInt(engine_stats.memory.peak_usage)) / (1024.0 * 1024.0),
                .pool_hit_rate = 0.0, // TODO: Calculate hit rate from pool stats
                .gpu_usage_mb = 0.0, // TODO: Add GPU memory tracking
            },
        };

        const response_json = try self.json_processor.serializeStatsResponse(stats_response);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 200, try self.allocator.dupe(u8, response_json));
    }

    fn handleCORS(self: *Self) Response {
        var headers = std.StringHashMap([]const u8).init(self.allocator);
        headers.put("Access-Control-Allow-Origin", "*") catch {};
        headers.put("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS") catch {};
        headers.put("Access-Control-Allow-Headers", "Content-Type, Authorization") catch {};

        return Response{
            .status_code = 200,
            .headers = headers,
            .body = "",
        };
    }

    fn handleNotFound(self: *Self) !Response {
        const error_response = json.ErrorResponse{
            .@"error" = "NotFound",
            .message = "Endpoint not found",
            .code = 404,
        };

        const response_json = try self.json_processor.serializeErrorResponse(error_response);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 404, try self.allocator.dupe(u8, response_json));
    }

    fn handleMethodNotAllowed(self: *Self) !Response {
        const error_response = json.ErrorResponse{
            .@"error" = "MethodNotAllowed",
            .message = "HTTP method not allowed for this endpoint",
            .code = 405,
        };

        const response_json = try self.json_processor.serializeErrorResponse(error_response);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 405, try self.allocator.dupe(u8, response_json));
    }

    fn handleBadRequest(self: *Self, message: []const u8) !Response {
        const error_response = json.ErrorResponse{
            .@"error" = "BadRequest",
            .message = message,
            .code = 400,
        };

        const response_json = try self.json_processor.serializeErrorResponse(error_response);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 400, try self.allocator.dupe(u8, response_json));
    }

    fn handleInternalServerError(self: *Self, message: []const u8) !Response {
        const error_response = json.ErrorResponse{
            .@"error" = "InternalServerError",
            .message = message,
            .code = 500,
        };

        const response_json = try self.json_processor.serializeErrorResponse(error_response);
        defer self.allocator.free(response_json);

        return Response.init(self.allocator, 500, try self.allocator.dupe(u8, response_json));
    }

    fn freeInferRequest(self: *Self, request: json.InferRequest) void {
        for (request.inputs) |input| {
            self.allocator.free(input.name);
            self.allocator.free(input.shape);
            self.allocator.free(input.data);
            self.allocator.free(input.dtype);
        }
        self.allocator.free(request.inputs);
        if (request.model_id) |model_id| {
            self.allocator.free(model_id);
        }
    }
};

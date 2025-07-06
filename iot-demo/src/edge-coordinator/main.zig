const std = @import("std");
const http = std.http;
const json = std.json;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Allocator = std.mem.Allocator;

/// IoT Edge Coordinator for Zig AI Platform
/// Manages multiple Raspberry Pi devices and coordinates inference requests
pub const EdgeCoordinator = struct {
    allocator: Allocator,
    devices: ArrayList(IoTDevice),
    device_map: HashMap([]const u8, *IoTDevice),
    port: u16,
    server: ?std.http.Server,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator, port: u16) !Self {
        return Self{
            .allocator = allocator,
            .devices = ArrayList(IoTDevice).init(allocator),
            .device_map = HashMap([]const u8, *IoTDevice).init(allocator),
            .port = port,
            .server = null,
        };
    }
    
    pub fn deinit(self: *Self) void {
        for (self.devices.items) |*device| {
            device.deinit(self.allocator);
        }
        self.devices.deinit();
        self.device_map.deinit();
        
        if (self.server) |*server| {
            server.deinit();
        }
    }
    
    /// Register a new IoT device
    pub fn registerDevice(self: *Self, device_info: DeviceInfo) !void {
        var device = IoTDevice.init(self.allocator, device_info);
        try self.devices.append(device);
        
        const device_ptr = &self.devices.items[self.devices.items.len - 1];
        try self.device_map.put(device_info.name, device_ptr);
        
        std.log.info("Registered device: {s} at {s}:{d}", .{ device_info.name, device_info.host, device_info.port });
    }
    
    /// Route inference request to optimal device
    pub fn routeInference(self: *Self, request: InferenceRequest) !InferenceResponse {
        // Find the best device for this request
        const optimal_device = try self.selectOptimalDevice(request);
        
        // Forward request to the selected device
        return try self.forwardToDevice(optimal_device, request);
    }
    
    /// Select optimal device based on load balancing strategy
    fn selectOptimalDevice(self: *Self, request: InferenceRequest) !*IoTDevice {
        if (self.devices.items.len == 0) {
            return error.NoDevicesAvailable;
        }
        
        // Simple least-loaded strategy for now
        var best_device: *IoTDevice = &self.devices.items[0];
        var lowest_load: f32 = std.math.floatMax(f32);
        
        for (self.devices.items) |*device| {
            if (device.status == .online) {
                const current_load = device.getCurrentLoad();
                if (current_load < lowest_load) {
                    lowest_load = current_load;
                    best_device = device;
                }
            }
        }
        
        if (best_device.status != .online) {
            return error.NoHealthyDevices;
        }
        
        return best_device;
    }
    
    /// Forward request to specific device
    fn forwardToDevice(self: *Self, device: *IoTDevice, request: InferenceRequest) !InferenceResponse {
        // In a real implementation, this would make HTTP request to the device
        // For now, simulate the response
        
        const start_time = std.time.milliTimestamp();
        
        // Simulate network latency and processing
        std.time.sleep(std.time.ns_per_ms * 100); // 100ms simulated latency
        
        const end_time = std.time.milliTimestamp();
        const processing_time = @as(f32, @floatFromInt(end_time - start_time));
        
        // Update device statistics
        device.requests_processed += 1;
        device.last_request_time = start_time;
        
        return InferenceResponse{
            .result = try std.fmt.allocPrint(self.allocator, "Response from {s}: Processed '{s}'", .{ device.info.name, request.query }),
            .processing_time_ms = processing_time,
            .device_name = device.info.name,
            .timestamp = start_time,
            .model_used = "edge-optimized-llm",
        };
    }
    
    /// Start the HTTP server
    pub fn start(self: *Self) !void {
        std.log.info("Starting Edge Coordinator on port {d}", .{self.port});
        
        // Initialize server
        var server = std.http.Server.init(self.allocator, .{ .reuse_address = true });
        self.server = server;
        
        const address = std.net.Address.parseIp("0.0.0.0", self.port) catch unreachable;
        try server.listen(address);
        
        std.log.info("Edge Coordinator listening on http://0.0.0.0:{d}", .{self.port});
        
        // Handle requests
        while (true) {
            var response = try server.accept(.{ .allocator = self.allocator });
            defer response.deinit();
            
            try self.handleRequest(&response);
        }
    }
    
    /// Handle incoming HTTP request
    fn handleRequest(self: *Self, response: *std.http.Server.Response) !void {
        try response.wait();
        
        const method = response.request.method;
        const target = response.request.target;
        
        std.log.info("Request: {s} {s}", .{ @tagName(method), target });
        
        if (std.mem.eql(u8, target, "/health")) {
            try self.handleHealthCheck(response);
        } else if (std.mem.eql(u8, target, "/api/status")) {
            try self.handleStatusRequest(response);
        } else if (std.mem.eql(u8, target, "/api/devices")) {
            try self.handleDevicesRequest(response);
        } else if (std.mem.eql(u8, target, "/api/inference")) {
            try self.handleInferenceRequest(response);
        } else {
            try self.handleNotFound(response);
        }
    }
    
    fn handleHealthCheck(self: *Self, response: *std.http.Server.Response) !void {
        _ = self;
        
        const health_response = 
            \\{"status": "healthy", "service": "edge-coordinator", "timestamp": "2024-01-01T00:00:00Z"}
        ;
        
        try response.headers.append("content-type", "application/json");
        try response.do();
        try response.writeAll(health_response);
        try response.finish();
    }
    
    fn handleStatusRequest(self: *Self, response: *std.http.Server.Response) !void {
        var status_json = ArrayList(u8).init(self.allocator);
        defer status_json.deinit();
        
        try status_json.appendSlice("{\"coordinator_status\":\"online\",\"devices\":[");
        
        for (self.devices.items, 0..) |*device, i| {
            if (i > 0) try status_json.appendSlice(",");
            
            const device_status = try std.fmt.allocPrint(self.allocator, 
                "{{\"name\":\"{s}\",\"status\":\"{s}\",\"load\":{d:.2},\"requests\":{d}}}", 
                .{ device.info.name, @tagName(device.status), device.getCurrentLoad(), device.requests_processed }
            );
            defer self.allocator.free(device_status);
            
            try status_json.appendSlice(device_status);
        }
        
        try status_json.appendSlice("]}");
        
        try response.headers.append("content-type", "application/json");
        try response.do();
        try response.writeAll(status_json.items);
        try response.finish();
    }
    
    fn handleDevicesRequest(self: *Self, response: *std.http.Server.Response) !void {
        var devices_json = ArrayList(u8).init(self.allocator);
        defer devices_json.deinit();
        
        try devices_json.appendSlice("{\"devices\":[");
        
        for (self.devices.items, 0..) |*device, i| {
            if (i > 0) try devices_json.appendSlice(",");
            
            const device_info = try std.fmt.allocPrint(self.allocator,
                "{{\"name\":\"{s}\",\"host\":\"{s}\",\"port\":{d},\"type\":\"{s}\",\"status\":\"{s}\"}}",
                .{ device.info.name, device.info.host, device.info.port, device.info.device_type, @tagName(device.status) }
            );
            defer self.allocator.free(device_info);
            
            try devices_json.appendSlice(device_info);
        }
        
        try devices_json.appendSlice("]}");
        
        try response.headers.append("content-type", "application/json");
        try response.do();
        try response.writeAll(devices_json.items);
        try response.finish();
    }
    
    fn handleInferenceRequest(self: *Self, response: *std.http.Server.Response) !void {
        // Read request body
        const body = try response.reader().readAllAlloc(self.allocator, 8192);
        defer self.allocator.free(body);
        
        // Parse JSON request
        var parsed = json.parseFromSlice(json.Value, self.allocator, body, .{}) catch {
            try self.sendErrorResponse(response, 400, "Invalid JSON");
            return;
        };
        defer parsed.deinit();
        
        const query = parsed.value.object.get("query") orelse {
            try self.sendErrorResponse(response, 400, "Missing 'query' field");
            return;
        };
        
        // Create inference request
        const inference_request = InferenceRequest{
            .query = query.string,
            .timestamp = std.time.milliTimestamp(),
            .device_id = null,
            .priority = .normal,
        };
        
        // Route to optimal device
        const inference_response = self.routeInference(inference_request) catch |err| {
            const error_msg = switch (err) {
                error.NoDevicesAvailable => "No devices available",
                error.NoHealthyDevices => "No healthy devices available",
                else => "Internal server error",
            };
            try self.sendErrorResponse(response, 500, error_msg);
            return;
        };
        
        // Send response
        const response_json = try std.fmt.allocPrint(self.allocator,
            "{{\"result\":\"{s}\",\"processing_time_ms\":{d:.2},\"device_name\":\"{s}\",\"model_used\":\"{s}\"}}",
            .{ inference_response.result, inference_response.processing_time_ms, inference_response.device_name, inference_response.model_used }
        );
        defer self.allocator.free(response_json);
        
        try response.headers.append("content-type", "application/json");
        try response.do();
        try response.writeAll(response_json);
        try response.finish();
    }
    
    fn handleNotFound(self: *Self, response: *std.http.Server.Response) !void {
        _ = self;
        
        const not_found_response = 
            \\{"error": "Not Found", "message": "The requested endpoint was not found"}
        ;
        
        response.status = .not_found;
        try response.headers.append("content-type", "application/json");
        try response.do();
        try response.writeAll(not_found_response);
        try response.finish();
    }
    
    fn sendErrorResponse(self: *Self, response: *std.http.Server.Response, status_code: u16, message: []const u8) !void {
        _ = self;
        
        const error_json = try std.fmt.allocPrint(self.allocator, 
            "{{\"error\":\"Error\",\"message\":\"{s}\"}}", .{message}
        );
        defer self.allocator.free(error_json);
        
        response.status = switch (status_code) {
            400 => .bad_request,
            500 => .internal_server_error,
            else => .internal_server_error,
        };
        
        try response.headers.append("content-type", "application/json");
        try response.do();
        try response.writeAll(error_json);
        try response.finish();
    }
};

/// IoT Device representation
pub const IoTDevice = struct {
    info: DeviceInfo,
    status: DeviceStatus,
    requests_processed: u64,
    last_request_time: i64,
    current_load: f32,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator, info: DeviceInfo) Self {
        _ = allocator;
        return Self{
            .info = info,
            .status = .online,
            .requests_processed = 0,
            .last_request_time = 0,
            .current_load = 0.0,
        };
    }
    
    pub fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.info.name);
        allocator.free(self.info.host);
        allocator.free(self.info.device_type);
    }
    
    pub fn getCurrentLoad(self: *Self) f32 {
        // Simulate load calculation based on recent requests
        const current_time = std.time.milliTimestamp();
        const time_since_last = current_time - self.last_request_time;
        
        // Load decreases over time
        if (time_since_last > 10000) { // 10 seconds
            self.current_load = 0.0;
        } else {
            self.current_load = @max(0.0, self.current_load - 0.1);
        }
        
        return self.current_load;
    }
};

/// Device information structure
pub const DeviceInfo = struct {
    name: []const u8,
    host: []const u8,
    port: u16,
    device_type: []const u8,
    capabilities: []const u8,
};

/// Device status enumeration
pub const DeviceStatus = enum {
    online,
    offline,
    maintenance,
    error,
};

/// Inference request structure
pub const InferenceRequest = struct {
    query: []const u8,
    timestamp: i64,
    device_id: ?[]const u8,
    priority: Priority,
    
    pub const Priority = enum {
        low,
        normal,
        high,
        urgent,
    };
};

/// Inference response structure
pub const InferenceResponse = struct {
    result: []const u8,
    processing_time_ms: f32,
    device_name: []const u8,
    timestamp: i64,
    model_used: []const u8,
};

/// Main function
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Initialize coordinator
    var coordinator = try EdgeCoordinator.init(allocator, 8080);
    defer coordinator.deinit();
    
    // Register simulated devices
    try coordinator.registerDevice(DeviceInfo{
        .name = try allocator.dupe(u8, "smart-home-pi"),
        .host = try allocator.dupe(u8, "localhost"),
        .port = 8081,
        .device_type = try allocator.dupe(u8, "raspberry-pi-4"),
        .capabilities = try allocator.dupe(u8, "smart-home,voice-assistant"),
    });
    
    try coordinator.registerDevice(DeviceInfo{
        .name = try allocator.dupe(u8, "industrial-pi"),
        .host = try allocator.dupe(u8, "localhost"),
        .port = 8082,
        .device_type = try allocator.dupe(u8, "raspberry-pi-4"),
        .capabilities = try allocator.dupe(u8, "industrial-iot,sensor-analysis"),
    });
    
    try coordinator.registerDevice(DeviceInfo{
        .name = try allocator.dupe(u8, "retail-pi"),
        .host = try allocator.dupe(u8, "localhost"),
        .port = 8083,
        .device_type = try allocator.dupe(u8, "raspberry-pi-4"),
        .capabilities = try allocator.dupe(u8, "retail-edge,customer-service"),
    });
    
    // Start the coordinator
    try coordinator.start();
}

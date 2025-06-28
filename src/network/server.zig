const std = @import("std");
const Allocator = std.mem.Allocator;
const routes = @import("routes.zig");
const engine = @import("../engine/inference.zig");

pub const ServerError = error{
    BindFailed,
    ListenFailed,
    AcceptFailed,
    InvalidRequest,
    ParseError,
    OutOfMemory,
};

pub const HTTPServer = struct {
    allocator: Allocator,
    port: u16,
    running: bool,
    server: ?std.net.StreamServer,
    router: ?routes.APIRouter,
    inference_engine: ?*engine.InferenceEngine,

    const Self = @This();

    pub fn init(allocator: Allocator, port: u16) !Self {
        return Self{
            .allocator = allocator,
            .port = port,
            .running = false,
            .server = null,
            .router = null,
            .inference_engine = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.stop();
        if (self.server) |*server| {
            server.deinit();
        }
    }

    pub fn setInferenceEngine(self: *Self, inference_engine: *engine.InferenceEngine) void {
        self.inference_engine = inference_engine;
        self.router = routes.APIRouter.init(self.allocator, inference_engine);
    }

    pub fn start(self: *Self) !void {
        if (self.inference_engine == null) {
            return ServerError.InvalidRequest;
        }

        self.running = true;

        // Create and bind server
        var server = std.net.StreamServer.init(.{});
        const address = std.net.Address.parseIp("0.0.0.0", self.port) catch |err| {
            std.log.err("Failed to parse address: {}", .{err});
            return ServerError.BindFailed;
        };

        server.listen(address) catch |err| {
            std.log.err("Failed to bind to port {d}: {}", .{ self.port, err });
            return ServerError.BindFailed;
        };

        self.server = server;

        std.log.info("HTTP server started on port {d}", .{self.port});

        // Start accepting connections
        try self.acceptLoop();
    }

    fn acceptLoop(self: *Self) !void {
        while (self.running) {
            if (self.server) |*server| {
                const connection = server.accept() catch |err| {
                    if (err == error.WouldBlock) {
                        std.time.sleep(1000000); // 1ms
                        continue;
                    }
                    std.log.err("Failed to accept connection: {}", .{err});
                    continue;
                };

                // Handle connection in a separate thread for concurrency
                const thread = std.Thread.spawn(.{}, handleConnection, .{ self, connection }) catch |err| {
                    std.log.err("Failed to spawn connection handler: {}", .{err});
                    connection.stream.close();
                    continue;
                };
                thread.detach();
            }
        }
    }

    fn handleConnection(self: *Self, connection: std.net.StreamServer.Connection) void {
        defer connection.stream.close();

        var buffer: [8192]u8 = undefined;
        const bytes_read = connection.stream.readAll(&buffer) catch |err| {
            std.log.err("Failed to read from connection: {}", .{err});
            return;
        };

        if (bytes_read == 0) return;

        const request_data = buffer[0..bytes_read];

        // Parse HTTP request
        var request = self.parseHTTPRequest(request_data) catch |err| {
            std.log.err("Failed to parse HTTP request: {}", .{err});
            self.sendErrorResponse(connection.stream, 400, "Bad Request") catch {};
            return;
        };
        defer request.deinit();

        // Route request
        if (self.router) |*router| {
            var response = router.route(&request) catch |err| {
                std.log.err("Failed to route request: {}", .{err});
                self.sendErrorResponse(connection.stream, 500, "Internal Server Error") catch {};
                return;
            };
            defer response.deinit();

            // Send response
            self.sendHTTPResponse(connection.stream, response) catch |err| {
                std.log.err("Failed to send response: {}", .{err});
            };
        }
    }

    fn parseHTTPRequest(self: *Self, data: []const u8) !routes.Request {
        var request = routes.Request.init(self.allocator);

        // Split request into lines
        var lines = std.mem.split(u8, data, "\r\n");

        // Parse request line
        const request_line = lines.next() orelse return ServerError.ParseError;
        var request_parts = std.mem.split(u8, request_line, " ");

        // Parse method
        const method_str = request_parts.next() orelse return ServerError.ParseError;
        request.method = std.meta.stringToEnum(routes.HTTPMethod, method_str) orelse return ServerError.ParseError;

        // Parse path
        const path_str = request_parts.next() orelse return ServerError.ParseError;
        request.path = try self.allocator.dupe(u8, path_str);

        // Parse headers
        while (lines.next()) |line| {
            if (line.len == 0) break; // Empty line indicates end of headers

            if (std.mem.indexOf(u8, line, ": ")) |colon_pos| {
                const header_name = line[0..colon_pos];
                const header_value = line[colon_pos + 2 ..];
                try request.headers.put(try self.allocator.dupe(u8, header_name), try self.allocator.dupe(u8, header_value));
            }
        }

        // Parse body (remaining data after headers)
        const headers_end = std.mem.indexOf(u8, data, "\r\n\r\n");
        if (headers_end) |end_pos| {
            const body_start = end_pos + 4;
            if (body_start < data.len) {
                request.body = try self.allocator.dupe(u8, data[body_start..]);
            }
        }

        return request;
    }

    fn sendHTTPResponse(self: *Self, stream: std.net.Stream, response: routes.Response) !void {

        // Send status line
        const status_line = try std.fmt.allocPrint(self.allocator, "HTTP/1.1 {d} {s}\r\n", .{ response.status_code, getStatusText(response.status_code) });
        defer self.allocator.free(status_line);
        _ = try stream.writeAll(status_line);

        // Send headers
        var header_iter = response.headers.iterator();
        while (header_iter.next()) |entry| {
            const header_line = try std.fmt.allocPrint(self.allocator, "{s}: {s}\r\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            defer self.allocator.free(header_line);
            _ = try stream.writeAll(header_line);
        }

        // Send content length
        const content_length = try std.fmt.allocPrint(self.allocator, "Content-Length: {d}\r\n", .{response.body.len});
        defer self.allocator.free(content_length);
        _ = try stream.writeAll(content_length);

        // End headers
        _ = try stream.writeAll("\r\n");

        // Send body
        _ = try stream.writeAll(response.body);
    }

    fn sendErrorResponse(self: *Self, stream: std.net.Stream, status_code: u16, message: []const u8) !void {
        const error_body = try std.fmt.allocPrint(self.allocator, "{{\"error\": \"{s}\", \"code\": {d}}}", .{ message, status_code });
        defer self.allocator.free(error_body);

        const status_line = try std.fmt.allocPrint(self.allocator, "HTTP/1.1 {d} {s}\r\n", .{ status_code, getStatusText(status_code) });
        defer self.allocator.free(status_line);

        _ = try stream.writeAll(status_line);
        _ = try stream.writeAll("Content-Type: application/json\r\n");

        const content_length = try std.fmt.allocPrint(self.allocator, "Content-Length: {d}\r\n", .{error_body.len});
        defer self.allocator.free(content_length);
        _ = try stream.writeAll(content_length);

        _ = try stream.writeAll("\r\n");
        _ = try stream.writeAll(error_body);
    }

    pub fn stop(self: *Self) void {
        self.running = false;
        if (self.server) |*server| {
            server.deinit();
            self.server = null;
        }
        std.log.info("HTTP server stopped", .{});
    }

    pub fn is_running(self: *const Self) bool {
        return self.running;
    }
};

fn getStatusText(status_code: u16) []const u8 {
    return switch (status_code) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        else => "Unknown",
    };
}

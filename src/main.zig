const std = @import("std");
const lib = @import("lib.zig");

const log = std.log;
const print = std.debug.print;

const Config = struct {
    port: u16 = 8080,
    model_path: ?[]const u8 = null,
    threads: u32 = 0, // 0 = auto-detect
    log_level: std.log.Level = .info,
    memory_limit_mb: u32 = 1024,
};

fn parseArgs(allocator: std.mem.Allocator) !Config {
    var config = Config{};

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--port") or std.mem.eql(u8, arg, "-p")) {
            i += 1;
            if (i >= args.len) {
                log.err("Missing value for --port", .{});
                return error.InvalidArgs;
            }
            config.port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            i += 1;
            if (i >= args.len) {
                log.err("Missing value for --model", .{});
                return error.InvalidArgs;
            }
            config.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--threads") or std.mem.eql(u8, arg, "-t")) {
            i += 1;
            if (i >= args.len) {
                log.err("Missing value for --threads", .{});
                return error.InvalidArgs;
            }
            config.threads = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--memory-limit")) {
            i += 1;
            if (i >= args.len) {
                log.err("Missing value for --memory-limit", .{});
                return error.InvalidArgs;
            }
            config.memory_limit_mb = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        } else {
            log.err("Unknown argument: {s}", .{arg});
            return error.InvalidArgs;
        }
    }

    return config;
}

fn printHelp() void {
    print(
        \\Zig AI Interface Engine
        \\
        \\Usage: ai-engine [OPTIONS]
        \\
        \\Options:
        \\  -p, --port <PORT>           Server port (default: 8080)
        \\  -m, --model <PATH>          Path to model file
        \\  -t, --threads <COUNT>       Number of worker threads (default: auto)
        \\      --memory-limit <MB>     Memory limit in MB (default: 1024)
        \\  -h, --help                  Show this help message
        \\
        \\Examples:
        \\  ai-engine --model model.onnx --port 3000
        \\  ai-engine --threads 4 --memory-limit 2048
        \\
    , .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = parseArgs(allocator) catch |err| switch (err) {
        error.InvalidArgs => {
            printHelp();
            std.process.exit(1);
        },
        else => return err,
    };

    log.info("Starting Zig AI Interface Engine", .{});
    log.info("Configuration:", .{});
    log.info("  Port: {d}", .{config.port});
    log.info("  Model: {s}", .{config.model_path orelse "none"});
    log.info("  Threads: {d}", .{if (config.threads == 0) std.Thread.getCpuCount() catch 1 else config.threads});
    log.info("  Memory limit: {d}MB", .{config.memory_limit_mb});

    // Initialize the AI engine
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = config.memory_limit_mb,
        .num_threads = if (config.threads == 0) null else config.threads,
    });
    defer engine.deinit();

    // Load model if specified
    if (config.model_path) |model_path| {
        log.info("Loading model from: {s}", .{model_path});
        try engine.loadModel(model_path);
        log.info("Model loaded successfully", .{});
    }

    // Start HTTP server
    log.info("Starting HTTP server on port {d}", .{config.port});
    try engine.startServer(config.port);

    // Keep the main thread alive
    log.info("Server started. Press Ctrl+C to stop.", .{});
    while (true) {
        std.time.sleep(std.time.ns_per_s);
    }
}

test "main tests" {
    _ = @import("test.zig");
}

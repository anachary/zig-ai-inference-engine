const std = @import("std");
const model_server = @import("zig-model-server");

/// Main entry point for the Zig Model Server CLI
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Initialize CLI
    var cli = model_server.CLI.init(allocator);

    // Handle graceful shutdown
    const original_handler = std.os.sigaction(std.os.SIG.INT, &std.os.Sigaction{
        .handler = .{ .handler = handleSignal },
        .mask = std.os.empty_sigset,
        .flags = 0,
    }, null);
    _ = original_handler;

    // Run CLI
    cli.run(args) catch |err| {
        switch (err) {
            error.NoCommand => {
                std.debug.print("❌ Error: No command specified\n\n");
                cli.printHelp();
                std.process.exit(1);
            },
            error.InvalidCommand => {
                std.debug.print("❌ Error: Invalid command\n\n");
                cli.printHelp();
                std.process.exit(1);
            },
            error.MissingValue => {
                std.debug.print("❌ Error: Missing required argument value\n");
                std.process.exit(1);
            },
            error.InvalidPort => {
                std.debug.print("❌ Error: Invalid port number\n");
                std.process.exit(1);
            },
            error.UnknownArgument => {
                std.debug.print("❌ Error: Unknown argument\n");
                std.process.exit(1);
            },
            else => {
                std.debug.print("❌ Error: {}\n", .{err});
                std.process.exit(1);
            },
        }
    };
}

/// Handle interrupt signals for graceful shutdown
fn handleSignal(sig: c_int) callconv(.C) void {
    switch (sig) {
        std.os.SIG.INT => {
            std.debug.print("\n🛑 Received interrupt signal, shutting down gracefully...\n");
            std.process.exit(0);
        },
        std.os.SIG.TERM => {
            std.debug.print("\n🛑 Received termination signal, shutting down gracefully...\n");
            std.process.exit(0);
        },
        else => {},
    }
}

/// Test the main CLI functionality
test "main CLI functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test CLI initialization
    var cli = model_server.CLI.init(allocator);
    _ = cli;

    // Test argument parsing
    const test_args = [_][]const u8{ "zig-model-server", "help" };
    const parsed = try model_server.Args.parse(allocator, &test_args);
    try std.testing.expect(parsed.command == .help);
}

/// Integration test for CLI commands
test "CLI command integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = model_server.CLI.init(allocator);

    // Test help command
    const help_args = [_][]const u8{ "zig-model-server", "help" };
    try cli.run(&help_args);

    // Test version command
    const version_args = [_][]const u8{ "zig-model-server", "version" };
    try cli.run(&version_args);
}

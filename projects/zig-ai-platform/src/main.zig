const std = @import("std");
const ai_platform = @import("lib.zig");

/// Main entry point for the Zig AI Platform CLI
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize platform library
    ai_platform.init();
    defer ai_platform.deinit();

    // Get command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Initialize CLI
    var cli = ai_platform.CLI.init(allocator);

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
            error.InvalidEnvironment => {
                std.debug.print("❌ Error: Invalid environment. Use: development, testing, staging, production\n");
                std.process.exit(1);
            },
            error.InvalidTarget => {
                std.debug.print("❌ Error: Invalid deployment target. Use: iot, desktop, server, cloud, kubernetes\n");
                std.process.exit(1);
            },
            error.EnvironmentRequired => {
                std.debug.print("❌ Error: Environment is required for deployment (--env)\n");
                std.process.exit(1);
            },
            error.TargetRequired => {
                std.debug.print("❌ Error: Deployment target is required (--target)\n");
                std.process.exit(1);
            },
            error.KeyRequired => {
                std.debug.print("❌ Error: Configuration key is required (--key)\n");
                std.process.exit(1);
            },
            error.ValueRequired => {
                std.debug.print("❌ Error: Configuration value is required (--value)\n");
                std.process.exit(1);
            },
            error.UnknownArgument => {
                std.debug.print("❌ Error: Unknown argument\n");
                std.process.exit(1);
            },
            error.ModelNotLoaded => {
                std.debug.print("❌ Error: Model not loaded\n");
                std.debug.print("💡 Load a model first before running inference\n");
                std.debug.print("📚 Use 'zig-ai-platform help' for more information\n");
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

test "main CLI functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test CLI initialization
    var cli = ai_platform.CLI.init(allocator);
    _ = cli;

    // Test argument parsing
    const test_args = [_][]const u8{ "zig-ai-platform", "help" };
    const parsed = try ai_platform.Args.parse(allocator, &test_args);
    try std.testing.expect(parsed.command == .help);
}

test "CLI command integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = ai_platform.CLI.init(allocator);

    // Test help command
    const help_args = [_][]const u8{ "zig-ai-platform", "help" };
    try cli.run(&help_args);

    // Test version command
    const version_args = [_][]const u8{ "zig-ai-platform", "version" };
    try cli.run(&version_args);
}

const std = @import("std");

/// Zig AI Ecosystem - Unified CLI Build File
///
/// This build file creates the unified "zig-ai" CLI that provides a single,
/// marketable interface for clients to chat with their local AI models.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the unified CLI executable with ONNX parser integration
    const cli_exe = b.addExecutable(.{
        .name = "zig-ai",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add ONNX parser as a module (Zig 0.11 syntax)
    const onnx_parser_module = b.createModule(.{
        .source_file = .{ .path = "projects/zig-onnx-parser/src/lib.zig" },
    });
    cli_exe.addModule("zig-onnx-parser", onnx_parser_module);

    // Add inference engine as a module
    const inference_engine_module = b.createModule(.{
        .source_file = .{ .path = "projects/zig-inference-engine/src/lib.zig" },
    });
    cli_exe.addModule("zig-inference-engine", inference_engine_module);

    // Add tensor core as a module
    const tensor_core_module = b.createModule(.{
        .source_file = .{ .path = "projects/zig-tensor-core/src/lib.zig" },
    });
    cli_exe.addModule("zig-tensor-core", tensor_core_module);

    // Install the CLI
    b.installArtifact(cli_exe);

    // Create run step for the CLI
    const cli_run = b.addRunArtifact(cli_exe);
    cli_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        cli_run.addArgs(args);
    }

    // Main CLI command
    const cli_step = b.step("cli", "Run the unified Zig AI CLI");
    cli_step.dependOn(&cli_run.step);

    // Debug ONNX parsing
    const debug_onnx_step = b.step("debug-onnx", "Debug ONNX parser issues");
    const debug_onnx_exe = b.addExecutable(.{
        .name = "debug-onnx",
        .root_source_file = .{ .path = "debug_onnx.zig" },
        .target = target,
        .optimize = optimize,
    });
    debug_onnx_exe.addModule("zig-onnx-parser", onnx_parser_module);
    const debug_onnx_run = b.addRunArtifact(debug_onnx_exe);
    debug_onnx_step.dependOn(&debug_onnx_run.step);

    // Print ecosystem information
    const info_step = b.step("info", "Show Zig AI Ecosystem information");
    const info_exe = b.addExecutable(.{
        .name = "info",
        .root_source_file = .{ .path = "src/info.zig" },
        .target = target,
        .optimize = optimize,
    });
    const info_run = b.addRunArtifact(info_exe);
    info_step.dependOn(&info_run.step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", "zig-cache", "zig-out" });
    clean_step.dependOn(&clean_cmd.step);

    // Development commands for ecosystem projects
    const dev_step = b.step("dev", "Development commands for ecosystem projects");
    const dev_exe = b.addExecutable(.{
        .name = "dev",
        .root_source_file = .{ .path = "src/dev.zig" },
        .target = target,
        .optimize = optimize,
    });
    const dev_run = b.addRunArtifact(dev_exe);
    dev_step.dependOn(&dev_run.step);

    // Test step
    const test_step = b.step("test", "Run CLI tests");
    const cli_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(cli_tests);
    test_step.dependOn(&run_tests.step);

    // Default step builds and installs the CLI
    b.default_step = b.getInstallStep();
}

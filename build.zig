const std = @import("std");

/// Zig AI Ecosystem - Orchestrator Build File
///
/// This build file orchestrates the modular Zig AI ecosystem.
/// Each project has its own build.zig for focused development.
/// Use this for ecosystem-wide operations and quick access.
pub fn build(b: *std.Build) void {
    _ = b.standardTargetOptions(.{});
    _ = b.standardOptimizeOption(.{});

    // Print ecosystem information
    const info_step = b.step("info", "Show Zig AI Ecosystem information");
    const info_cmd = b.addSystemCommand(&[_][]const u8{ "echo", "🚀 Zig AI Ecosystem - Modular Architecture\n" ++
        "==========================================\n" ++
        "📦 Projects:\n" ++
        "  • zig-tensor-core      - Tensor operations & memory\n" ++
        "  • zig-onnx-parser      - ONNX model parsing\n" ++
        "  • zig-inference-engine - Model execution\n" ++
        "  • zig-model-server     - HTTP API & CLI\n" ++
        "  • zig-ai-platform      - Unified orchestrator\n\n" ++
        "🔧 Quick Commands:\n" ++
        "  zig build info          - Show this information\n" ++
        "  zig build build-all     - Build all projects\n" ++
        "  zig build test-all      - Test all projects\n" ++
        "  zig build clean-all     - Clean all projects\n\n" ++
        "📁 Individual Projects:\n" ++
        "  cd projects/[project-name] && zig build\n" });
    info_step.dependOn(&info_cmd.step);

    // Build all projects
    const build_all_step = b.step("build-all", "Build all ecosystem projects");

    const projects = [_][]const u8{
        "zig-tensor-core",
        "zig-onnx-parser",
        "zig-inference-engine",
        "zig-model-server",
        "zig-ai-platform",
    };

    for (projects) |project| {
        const build_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "build", "-p", b.fmt("projects/{s}", .{project}) });
        build_cmd.cwd = b.fmt("projects/{s}", .{project});
        build_all_step.dependOn(&build_cmd.step);
    }

    // Test all projects
    const test_all_step = b.step("test-all", "Test all ecosystem projects");

    for (projects) |project| {
        const test_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "build", "test", "-p", b.fmt("projects/{s}", .{project}) });
        test_cmd.cwd = b.fmt("projects/{s}", .{project});
        test_all_step.dependOn(&test_cmd.step);
    }

    // Clean all projects
    const clean_all_step = b.step("clean-all", "Clean all ecosystem projects");

    for (projects) |project| {
        const clean_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "build", "clean", "-p", b.fmt("projects/{s}", .{project}) });
        clean_cmd.cwd = b.fmt("projects/{s}", .{project});
        clean_all_step.dependOn(&clean_cmd.step);
    }

    // Quick access to main CLI (from zig-model-server)
    const cli_step = b.step("cli", "Run the main Zig AI CLI");
    const cli_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "build", "run", "-p", "projects/zig-model-server" });
    cli_cmd.cwd = "projects/zig-model-server";
    cli_step.dependOn(&cli_cmd.step);

    // Quick access to platform orchestrator
    const platform_step = b.step("platform", "Run the Zig AI Platform");
    const platform_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "build", "run", "-p", "projects/zig-ai-platform" });
    platform_cmd.cwd = "projects/zig-ai-platform";
    platform_step.dependOn(&platform_cmd.step);

    // Default step shows info
    b.default_step = info_step;
}

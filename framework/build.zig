const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Framework library
    const framework_lib = b.addStaticLibrary(.{
        .name = "zig-ai-framework",
        .root_source_file = b.path("lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Install the framework library
    b.installArtifact(framework_lib);

    // Create framework module for external use
    const framework_module = b.addModule("zig-ai-framework", .{
        .root_source_file = b.path("lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Framework tests
    const framework_tests = b.addTest(.{
        .root_source_file = b.path("lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_step = b.step("test", "Run framework tests");
    test_step.dependOn(&b.addRunArtifact(framework_tests).step);

    // Examples
    const examples_step = b.step("examples", "Build framework examples");

    // Basic framework usage example
    const basic_example = b.addExecutable(.{
        .name = "basic-framework-example",
        .root_source_file = b.path("examples/basic_usage.zig"),
        .target = target,
        .optimize = optimize,
    });
    basic_example.root_module.addImport("zig-ai-framework", framework_module);

    const install_basic_example = b.addInstallArtifact(basic_example, .{});
    examples_step.dependOn(&install_basic_example.step);

    // Operator registration example
    const operator_example = b.addExecutable(.{
        .name = "operator-registration-example",
        .root_source_file = b.path("examples/operator_registration.zig"),
        .target = target,
        .optimize = optimize,
    });
    operator_example.root_module.addImport("zig-ai-framework", framework_module);

    const install_operator_example = b.addInstallArtifact(operator_example, .{});
    examples_step.dependOn(&install_operator_example.step);

    // Benchmarks
    const benchmark_step = b.step("benchmark", "Run framework benchmarks");

    const framework_benchmark = b.addExecutable(.{
        .name = "framework-benchmark",
        .root_source_file = b.path("benchmarks/framework_benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    framework_benchmark.root_module.addImport("zig-ai-framework", framework_module);

    const run_benchmark = b.addRunArtifact(framework_benchmark);
    benchmark_step.dependOn(&run_benchmark.step);

    // Documentation generation
    const docs_step = b.step("docs", "Generate framework documentation");
    const docs = framework_lib.getEmittedDocs();
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs,
        .install_dir = .prefix,
        .install_subdir = "docs/framework",
    });
    docs_step.dependOn(&install_docs.step);
}

// Helper function to create module with dependencies
pub fn createFrameworkModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.addModule("zig-ai-framework", .{
        .root_source_file = b.path("lib.zig"),
        .target = target,
        .optimize = optimize,
    });
}

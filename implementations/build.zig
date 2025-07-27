const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Add framework dependency
    const framework_dep = b.dependency("zig-ai-framework", .{
        .target = target,
        .optimize = optimize,
    });
    const framework_module = framework_dep.module("zig-ai-framework");

    // Implementations library
    const implementations_lib = b.addStaticLibrary(.{
        .name = "zig-ai-implementations",
        .root_source_file = b.path("lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    implementations_lib.root_module.addImport("framework", framework_module);

    // Install the implementations library
    b.installArtifact(implementations_lib);

    // Create implementations module for external use
    const implementations_module = b.addModule("zig-ai-implementations", .{
        .root_source_file = b.path("lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    implementations_module.addImport("framework", framework_module);

    // Operator tests
    const operator_tests = b.addTest(.{
        .root_source_file = b.path("operators/arithmetic/add.zig"),
        .target = target,
        .optimize = optimize,
    });
    operator_tests.root_module.addImport("framework", framework_module);

    const test_step = b.step("test", "Run operator tests");
    test_step.dependOn(&b.addRunArtifact(operator_tests).step);

    // Model-specific tests
    const transformer_tests = b.addTest(.{
        .root_source_file = b.path("models/transformers/common.zig"),
        .target = target,
        .optimize = optimize,
    });
    transformer_tests.root_module.addImport("framework", framework_module);

    const transformer_test_step = b.step("test-transformers", "Run transformer tests");
    transformer_test_step.dependOn(&b.addRunArtifact(transformer_tests).step);

    // Examples
    const examples_step = b.step("examples", "Build implementation examples");

    // Custom operator example
    const custom_operator_example = b.addExecutable(.{
        .name = "custom-operator-example",
        .root_source_file = b.path("examples/custom_operator.zig"),
        .target = target,
        .optimize = optimize,
    });
    custom_operator_example.root_module.addImport("framework", framework_module);
    custom_operator_example.root_module.addImport("implementations", implementations_module);

    const install_custom_example = b.addInstallArtifact(custom_operator_example, .{});
    examples_step.dependOn(&install_custom_example.step);

    // Model architecture example
    const model_arch_example = b.addExecutable(.{
        .name = "model-architecture-example",
        .root_source_file = b.path("examples/model_architecture.zig"),
        .target = target,
        .optimize = optimize,
    });
    model_arch_example.root_module.addImport("framework", framework_module);
    model_arch_example.root_module.addImport("implementations", implementations_module);

    const install_model_example = b.addInstallArtifact(model_arch_example, .{});
    examples_step.dependOn(&install_model_example.step);

    // Benchmarks
    const benchmark_step = b.step("benchmark", "Run implementation benchmarks");

    const operator_benchmark = b.addExecutable(.{
        .name = "operator-benchmark",
        .root_source_file = b.path("benchmarks/operator_benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    operator_benchmark.root_module.addImport("framework", framework_module);
    operator_benchmark.root_module.addImport("implementations", implementations_module);

    const run_operator_benchmark = b.addRunArtifact(operator_benchmark);
    benchmark_step.dependOn(&run_operator_benchmark.step);

    // Model benchmark
    const model_benchmark = b.addExecutable(.{
        .name = "model-benchmark",
        .root_source_file = b.path("benchmarks/model_benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    model_benchmark.root_module.addImport("framework", framework_module);
    model_benchmark.root_module.addImport("implementations", implementations_module);

    const run_model_benchmark = b.addRunArtifact(model_benchmark);
    benchmark_step.dependOn(&run_model_benchmark.step);

    // Documentation generation
    const docs_step = b.step("docs", "Generate implementation documentation");
    const docs = implementations_lib.getEmittedDocs();
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs,
        .install_dir = .prefix,
        .install_subdir = "docs/implementations",
    });
    docs_step.dependOn(&install_docs.step);

    // Cross-compilation targets for different architectures
    const cross_targets = [_]std.Target.Query{
        .{ .cpu_arch = .x86_64, .os_tag = .linux },
        .{ .cpu_arch = .aarch64, .os_tag = .linux },
        .{ .cpu_arch = .x86_64, .os_tag = .windows },
        .{ .cpu_arch = .x86_64, .os_tag = .macos },
        .{ .cpu_arch = .aarch64, .os_tag = .macos },
    };

    const cross_step = b.step("cross", "Cross-compile for multiple targets");

    for (cross_targets, 0..) |cross_target, i| {
        const cross_lib = b.addStaticLibrary(.{
            .name = b.fmt("zig-ai-implementations-{s}-{s}", .{ @tagName(cross_target.cpu_arch.?), @tagName(cross_target.os_tag.?) }),
            .root_source_file = b.path("lib.zig"),
            .target = b.resolveTargetQuery(cross_target),
            .optimize = .ReleaseSmall, // Optimize for size on cross-compiled targets
        });
        cross_lib.root_module.addImport("framework", framework_module);

        const cross_install = b.addInstallArtifact(cross_lib, .{});
        cross_step.dependOn(&cross_install.step);

        const cross_target_step = b.step(b.fmt("cross-{d}", .{i}), b.fmt("Cross-compile for {s}-{s}", .{ @tagName(cross_target.cpu_arch.?), @tagName(cross_target.os_tag.?) }));
        cross_target_step.dependOn(&cross_install.step);
    }
}

// Helper function to create implementations module with framework dependency
pub fn createImplementationsModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    framework_module: *std.Build.Module,
) *std.Build.Module {
    const implementations_module = b.addModule("zig-ai-implementations", .{
        .root_source_file = b.path("lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    implementations_module.addImport("framework", framework_module);
    return implementations_module;
}

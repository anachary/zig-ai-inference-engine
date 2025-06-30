const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the main library
    const lib = b.addStaticLibrary(.{
        .name = "zig-inference-engine",
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add common interfaces as a module
    const common_interfaces = b.addModule("common-interfaces", .{
        .root_source_file = b.path("../../common/interfaces/tensor.zig"),
    });
    lib.root_module.addImport("common-interfaces", common_interfaces);

    // Install the library
    b.installArtifact(lib);

    // Create a module for external use
    const inference_engine_module = b.addModule("zig-inference-engine", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    inference_engine_module.addImport("common-interfaces", common_interfaces);

    // Unit tests
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_unit_tests.root_module.addImport("common-interfaces", common_interfaces);

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    // Engine tests
    const engine_tests = b.addTest(.{
        .root_source_file = b.path("src/engine/engine.zig"),
        .target = target,
        .optimize = optimize,
    });
    engine_tests.root_module.addImport("common-interfaces", common_interfaces);

    const run_engine_tests = b.addRunArtifact(engine_tests);
    const engine_test_step = b.step("test-engine", "Run engine tests");
    engine_test_step.dependOn(&run_engine_tests.step);

    // Operator tests
    const operator_tests = b.addTest(.{
        .root_source_file = b.path("src/operators/registry.zig"),
        .target = target,
        .optimize = optimize,
    });
    operator_tests.root_module.addImport("common-interfaces", common_interfaces);

    const run_operator_tests = b.addRunArtifact(operator_tests);
    const operator_test_step = b.step("test-operators", "Run operator tests");
    operator_test_step.dependOn(&run_operator_tests.step);

    // Scheduler tests
    const scheduler_tests = b.addTest(.{
        .root_source_file = b.path("src/scheduler/scheduler.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_scheduler_tests = b.addRunArtifact(scheduler_tests);
    const scheduler_test_step = b.step("test-scheduler", "Run scheduler tests");
    scheduler_test_step.dependOn(&run_scheduler_tests.step);

    // GPU backend tests
    const gpu_tests = b.addTest(.{
        .root_source_file = b.path("src/gpu/backend.zig"),
        .target = target,
        .optimize = optimize,
    });
    gpu_tests.root_module.addImport("common-interfaces", common_interfaces);

    const run_gpu_tests = b.addRunArtifact(gpu_tests);
    const gpu_test_step = b.step("test-gpu", "Run GPU backend tests");
    gpu_test_step.dependOn(&run_gpu_tests.step);

    // All tests
    const all_tests_step = b.step("test-all", "Run all tests");
    all_tests_step.dependOn(&run_lib_unit_tests.step);
    all_tests_step.dependOn(&run_engine_tests.step);
    all_tests_step.dependOn(&run_operator_tests.step);
    all_tests_step.dependOn(&run_scheduler_tests.step);
    all_tests_step.dependOn(&run_gpu_tests.step);

    // Examples
    const examples_step = b.step("examples", "Build examples");

    // Basic inference example
    const basic_example = b.addExecutable(.{
        .name = "basic-inference",
        .root_source_file = b.path("examples/basic_inference.zig"),
        .target = target,
        .optimize = optimize,
    });
    basic_example.root_module.addImport("zig-inference-engine", inference_engine_module);
    basic_example.root_module.addImport("common-interfaces", common_interfaces);

    const install_basic_example = b.addInstallArtifact(basic_example, .{});
    examples_step.dependOn(&install_basic_example.step);

    // Operator benchmark example
    const benchmark_example = b.addExecutable(.{
        .name = "operator-benchmark",
        .root_source_file = b.path("examples/operator_benchmark.zig"),
        .target = target,
        .optimize = optimize,
    });
    benchmark_example.root_module.addImport("zig-inference-engine", inference_engine_module);
    benchmark_example.root_module.addImport("common-interfaces", common_interfaces);

    const install_benchmark_example = b.addInstallArtifact(benchmark_example, .{});
    examples_step.dependOn(&install_benchmark_example.step);

    // GPU example
    const gpu_example = b.addExecutable(.{
        .name = "gpu-inference",
        .root_source_file = b.path("examples/gpu_inference.zig"),
        .target = target,
        .optimize = optimize,
    });
    gpu_example.root_module.addImport("zig-inference-engine", inference_engine_module);
    gpu_example.root_module.addImport("common-interfaces", common_interfaces);

    const install_gpu_example = b.addInstallArtifact(gpu_example, .{});
    examples_step.dependOn(&install_gpu_example.step);

    // Benchmarks
    const benchmark_step = b.step("benchmark", "Run benchmarks");

    const operator_benchmark = b.addExecutable(.{
        .name = "operator-benchmark-runner",
        .root_source_file = b.path("benchmarks/operator_benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    operator_benchmark.root_module.addImport("zig-inference-engine", inference_engine_module);
    operator_benchmark.root_module.addImport("common-interfaces", common_interfaces);

    const run_operator_benchmark = b.addRunArtifact(operator_benchmark);
    benchmark_step.dependOn(&run_operator_benchmark.step);

    const scheduler_benchmark = b.addExecutable(.{
        .name = "scheduler-benchmark-runner",
        .root_source_file = b.path("benchmarks/scheduler_benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    scheduler_benchmark.root_module.addImport("zig-inference-engine", inference_engine_module);

    const run_scheduler_benchmark = b.addRunArtifact(scheduler_benchmark);
    benchmark_step.dependOn(&run_scheduler_benchmark.step);

    // Documentation
    const docs_step = b.step("docs", "Generate documentation");
    const docs = lib_unit_tests;
    docs.emit_docs = .emit;
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&install_docs.step);

    // Integration tests (requires other modules)
    const integration_test_step = b.step("test-integration", "Run integration tests");
    
    // Note: These would require zig-tensor-core and zig-onnx-parser to be available
    // For now, we'll create placeholder integration tests
    const integration_tests = b.addTest(.{
        .root_source_file = b.path("tests/integration_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_tests.root_module.addImport("zig-inference-engine", inference_engine_module);
    integration_tests.root_module.addImport("common-interfaces", common_interfaces);

    const run_integration_tests = b.addRunArtifact(integration_tests);
    integration_test_step.dependOn(&run_integration_tests.step);

    // Performance profiling
    const profile_step = b.step("profile", "Run performance profiling");
    
    const profile_exe = b.addExecutable(.{
        .name = "inference-profiler",
        .root_source_file = b.path("tools/profiler.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    profile_exe.root_module.addImport("zig-inference-engine", inference_engine_module);
    profile_exe.root_module.addImport("common-interfaces", common_interfaces);

    const run_profiler = b.addRunArtifact(profile_exe);
    profile_step.dependOn(&run_profiler.step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", "zig-out", "zig-cache" });
    clean_step.dependOn(&clean_cmd.step);

    // Install step
    const install_step = b.step("install", "Install the library");
    install_step.dependOn(&b.default_step.step);

    // Development helpers
    const dev_step = b.step("dev", "Development build with all features");
    dev_step.dependOn(&test_step.step);
    dev_step.dependOn(&examples_step.step);
    dev_step.dependOn(&docs_step.step);

    // CI step
    const ci_step = b.step("ci", "Continuous integration build");
    ci_step.dependOn(&all_tests_step.step);
    ci_step.dependOn(&examples_step.step);
    ci_step.dependOn(&benchmark_step.step);

    // Release step
    const release_step = b.step("release", "Release build");
    release_step.dependOn(&all_tests_step.step);
    release_step.dependOn(&docs_step.step);
    release_step.dependOn(&install_step.step);
}

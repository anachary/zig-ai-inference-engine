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

    // Add common interfaces module first
    const common_interfaces_module = b.createModule(.{
        .source_file = .{ .path = "common/interfaces/tensor.zig" },
    });

    // Add tensor core as a module
    const tensor_core_module = b.createModule(.{
        .source_file = .{ .path = "projects/zig-tensor-core/src/lib.zig" },
    });

    // Add ONNX parser as a module (Zig 0.11 syntax)
    const onnx_parser_module = b.createModule(.{
        .source_file = .{ .path = "projects/zig-onnx-parser/src/lib.zig" },
    });

    // Add inference engine as a module with dependencies
    const inference_engine_module = b.createModule(.{
        .source_file = .{ .path = "projects/zig-inference-engine/src/lib.zig" },
        .dependencies = &.{
            .{ .name = "common-interfaces", .module = common_interfaces_module },
        },
    });

    // Add modules to CLI executable
    cli_exe.addModule("zig-onnx-parser", onnx_parser_module);
    cli_exe.addModule("zig-inference-engine", inference_engine_module);
    cli_exe.addModule("zig-tensor-core", tensor_core_module);
    cli_exe.addModule("common-interfaces", common_interfaces_module);

    // Install the CLI
    b.installArtifact(cli_exe);

    // Create test executable
    const test_exe = b.addTest(.{
        .name = "test-operators",
        .root_source_file = .{ .path = "tests/test_operators.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add modules to test executable
    test_exe.addModule("zig-inference-engine", inference_engine_module);
    test_exe.addModule("zig-tensor-core", tensor_core_module);
    test_exe.addModule("common-interfaces", common_interfaces_module);

    // Create operator test step
    const operator_test_step = b.step("test-operators", "Run operator tests");
    operator_test_step.dependOn(&b.addRunArtifact(test_exe).step);

    // Create real inference test executable
    const inference_test_exe = b.addTest(.{
        .name = "test-inference",
        .root_source_file = .{ .path = "tests/test_real_inference.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add modules to inference test executable
    inference_test_exe.addModule("zig-inference-engine", inference_engine_module);
    inference_test_exe.addModule("zig-tensor-core", tensor_core_module);
    inference_test_exe.addModule("common-interfaces", common_interfaces_module);

    // Create inference test step
    const inference_test_step = b.step("test-inference", "Run real inference tests");
    inference_test_step.dependOn(&b.addRunArtifact(inference_test_exe).step);

    // Create real model test executable
    const real_model_test_exe = b.addExecutable(.{
        .name = "test-real-model",
        .root_source_file = .{ .path = "test_real_model.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add modules to real model test executable
    real_model_test_exe.addModule("zig-onnx-parser", onnx_parser_module);
    real_model_test_exe.addModule("zig-inference-engine", inference_engine_module);
    real_model_test_exe.addModule("zig-tensor-core", tensor_core_module);
    real_model_test_exe.addModule("common-interfaces", common_interfaces_module);

    // Create real model test step
    const real_model_test_step = b.step("test-real-model", "Test real ONNX model loading");
    real_model_test_step.dependOn(&b.addRunArtifact(real_model_test_exe).step);

    // Create complete inference test executable
    const complete_test_exe = b.addExecutable(.{
        .name = "test-complete-inference",
        .root_source_file = .{ .path = "test_complete_inference.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add modules to complete inference test executable
    complete_test_exe.addModule("zig-inference-engine", inference_engine_module);
    complete_test_exe.addModule("zig-tensor-core", tensor_core_module);
    complete_test_exe.addModule("common-interfaces", common_interfaces_module);

    // Create complete inference test step
    const complete_test_step = b.step("test-complete", "Test complete inference pipeline");
    complete_test_step.dependOn(&b.addRunArtifact(complete_test_exe).step);

    // Create end-to-end test executable
    const e2e_test_exe = b.addExecutable(.{
        .name = "test-e2e-inference",
        .root_source_file = .{ .path = "test_e2e_inference.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add modules to e2e test executable
    e2e_test_exe.addModule("zig-onnx-parser", onnx_parser_module);
    e2e_test_exe.addModule("zig-inference-engine", inference_engine_module);
    e2e_test_exe.addModule("zig-tensor-core", tensor_core_module);
    e2e_test_exe.addModule("common-interfaces", common_interfaces_module);

    // Create end-to-end test step
    const e2e_test_step = b.step("test-e2e", "Test end-to-end inference with real models");
    e2e_test_step.dependOn(&b.addRunArtifact(e2e_test_exe).step);

    // Create performance benchmark executable
    const benchmark_exe = b.addExecutable(.{
        .name = "benchmark-performance",
        .root_source_file = .{ .path = "benchmark_performance.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add modules to benchmark executable
    benchmark_exe.addModule("zig-onnx-parser", onnx_parser_module);
    benchmark_exe.addModule("zig-inference-engine", inference_engine_module);
    benchmark_exe.addModule("zig-tensor-core", tensor_core_module);
    benchmark_exe.addModule("common-interfaces", common_interfaces_module);

    // Create benchmark step
    const benchmark_step = b.step("benchmark", "Run comprehensive performance benchmarks");
    benchmark_step.dependOn(&b.addRunArtifact(benchmark_exe).step);

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

const std = @import("std");

/// Zig AI Ecosystem - Unified CLI Build File
///
/// This build file creates the unified "zig-ai" CLI that provides a single,
/// marketable interface for clients to chat with their local AI models.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // === SEGFAULT PREVENTION CONFIGURATION ===
    // Memory safety and debugging options
    const enable_safety_checks = b.option(bool, "safety", "Enable runtime safety checks") orelse true;
    const enable_stack_protection = b.option(bool, "stack-protection", "Enable stack overflow protection") orelse true;
    const enable_memory_debugging = b.option(bool, "memory-debug", "Enable memory debugging features") orelse (optimize == .Debug);
    const enable_bounds_checking = b.option(bool, "bounds-check", "Enable array bounds checking") orelse true;
    const enable_overflow_checks = b.option(bool, "overflow-check", "Enable integer overflow checks") orelse true;

    // ONNX Runtime debugging options for fixing remaining issues
    const enable_graph_validation = b.option(bool, "graph-validation", "Enable graph validation (may cause segfault)") orelse false;
    const enable_topological_sort = b.option(bool, "topological-sort", "Enable topological sort optimization") orelse false;
    const enable_cleanup_exit = b.option(bool, "cleanup-exit", "Enable normal cleanup instead of early exit") orelse false;

    // Production mode - safe memory management without cleanup testing
    const enable_production_mode = b.option(bool, "production", "Enable production mode (safe memory management, no cleanup testing)") orelse false;

    // Helper function to configure memory safety for executables (Zig 0.11 compatible)
    const configureMemorySafety = struct {
        fn apply(exe: *std.Build.Step.Compile, safety_checks: bool, stack_protection: bool, memory_debugging: bool, bounds_checking: bool, overflow_checks: bool) void {
            // Configure available safety options for Zig 0.11
            if (memory_debugging) {
                exe.strip = false; // Keep debug symbols for debugging
            }
            if (bounds_checking) {
                exe.single_threaded = false; // Enable multi-threading for runtime checks
            }
            // Note: Many safety features in Zig 0.11 are controlled by build mode and compile-time options
            // The build options will be passed to the code via the options module
            _ = safety_checks; // Passed via build options
            _ = stack_protection; // Passed via build options
            _ = overflow_checks; // Passed via build options
        }
    }.apply;

    // Create the unified CLI executable with ONNX parser integration
    const cli_exe = b.addExecutable(.{
        .name = "zig-ai",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Apply memory safety configuration to CLI
    configureMemorySafety(cli_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);

    // Add build options as compile-time constants
    const options = b.addOptions();
    // Memory safety options
    options.addOption(bool, "enable_safety_checks", enable_safety_checks);
    options.addOption(bool, "enable_stack_protection", enable_stack_protection);
    options.addOption(bool, "enable_memory_debugging", enable_memory_debugging);
    options.addOption(bool, "enable_bounds_checking", enable_bounds_checking);
    options.addOption(bool, "enable_overflow_checks", enable_overflow_checks);
    // ONNX Runtime options
    options.addOption(bool, "enable_graph_validation", enable_graph_validation);
    options.addOption(bool, "enable_topological_sort", enable_topological_sort);
    options.addOption(bool, "enable_cleanup_exit", enable_cleanup_exit);
    options.addOption(bool, "enable_production_mode", enable_production_mode);

    const options_module = options.createModule();

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

    // Add ONNX Runtime as a module with dependencies
    const onnx_runtime_module = b.createModule(.{
        .source_file = .{ .path = "projects/zig-onnx-runtime/src/lib.zig" },
        .dependencies = &.{
            .{ .name = "build_options", .module = options_module },
        },
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
    cli_exe.addModule("zig-onnx-runtime", onnx_runtime_module);
    cli_exe.addModule("zig-inference-engine", inference_engine_module);
    cli_exe.addModule("zig-tensor-core", tensor_core_module);
    cli_exe.addModule("common-interfaces", common_interfaces_module);
    cli_exe.addModule("build_options", options_module);

    // Install the CLI
    b.installArtifact(cli_exe);

    // Create test executable
    const test_exe = b.addTest(.{
        .name = "test-operators",
        .root_source_file = .{ .path = "tests/test_operators.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Apply memory safety configuration to test executable
    configureMemorySafety(test_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);

    // Add modules to test executable
    test_exe.addModule("zig-onnx-runtime", onnx_runtime_module);
    test_exe.addModule("zig-inference-engine", inference_engine_module);
    test_exe.addModule("zig-tensor-core", tensor_core_module);
    test_exe.addModule("common-interfaces", common_interfaces_module);
    test_exe.addModule("build_options", options_module);

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

    // Apply memory safety configuration to inference test executable
    configureMemorySafety(inference_test_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);

    // Add modules to inference test executable
    inference_test_exe.addModule("zig-onnx-runtime", onnx_runtime_module);
    inference_test_exe.addModule("zig-inference-engine", inference_engine_module);
    inference_test_exe.addModule("zig-tensor-core", tensor_core_module);
    inference_test_exe.addModule("common-interfaces", common_interfaces_module);
    inference_test_exe.addModule("build_options", options_module);

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

    // Apply memory safety configuration to real model test executable
    configureMemorySafety(real_model_test_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);

    // Add modules to real model test executable
    real_model_test_exe.addModule("zig-onnx-parser", onnx_parser_module);
    real_model_test_exe.addModule("zig-inference-engine", inference_engine_module);
    real_model_test_exe.addModule("zig-tensor-core", tensor_core_module);
    real_model_test_exe.addModule("common-interfaces", common_interfaces_module);
    real_model_test_exe.addModule("build_options", options_module);

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

    // Apply memory safety configuration to complete inference test executable
    configureMemorySafety(complete_test_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);

    // Add modules to complete inference test executable
    complete_test_exe.addModule("zig-inference-engine", inference_engine_module);
    complete_test_exe.addModule("zig-tensor-core", tensor_core_module);
    complete_test_exe.addModule("common-interfaces", common_interfaces_module);
    complete_test_exe.addModule("build_options", options_module);

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

    // Apply memory safety configuration to e2e test executable
    configureMemorySafety(e2e_test_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);

    // Add modules to e2e test executable
    e2e_test_exe.addModule("zig-onnx-parser", onnx_parser_module);
    e2e_test_exe.addModule("zig-inference-engine", inference_engine_module);
    e2e_test_exe.addModule("zig-tensor-core", tensor_core_module);
    e2e_test_exe.addModule("common-interfaces", common_interfaces_module);
    e2e_test_exe.addModule("build_options", options_module);

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

    // Apply memory safety configuration to benchmark executable
    configureMemorySafety(benchmark_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);

    // Add modules to benchmark executable
    benchmark_exe.addModule("zig-onnx-parser", onnx_parser_module);
    benchmark_exe.addModule("zig-inference-engine", inference_engine_module);
    benchmark_exe.addModule("zig-tensor-core", tensor_core_module);
    benchmark_exe.addModule("common-interfaces", common_interfaces_module);
    benchmark_exe.addModule("build_options", options_module);

    // Create benchmark step
    const benchmark_step = b.step("benchmark", "Run comprehensive performance benchmarks");
    benchmark_step.dependOn(&b.addRunArtifact(benchmark_exe).step);

    // Create LLM chat executable
    const llm_chat_exe = b.addExecutable(.{
        .name = "llm-chat",
        .root_source_file = .{ .path = "llm_chat.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Apply memory safety configuration to LLM chat executable
    configureMemorySafety(llm_chat_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);

    // Add modules to LLM chat executable
    llm_chat_exe.addModule("zig-onnx-parser", onnx_parser_module);
    llm_chat_exe.addModule("zig-inference-engine", inference_engine_module);
    llm_chat_exe.addModule("zig-tensor-core", tensor_core_module);
    llm_chat_exe.addModule("common-interfaces", common_interfaces_module);
    llm_chat_exe.addModule("build_options", options_module);

    // Create LLM chat step
    const llm_chat_step = b.step("llm-chat", "Run LLM chat interface");
    llm_chat_step.dependOn(&b.addRunArtifact(llm_chat_exe).step);

    // Create simple LLM demo executable
    const simple_llm_exe = b.addExecutable(.{
        .name = "simple-llm-demo",
        .root_source_file = .{ .path = "simple_llm_demo.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Create simple LLM demo step
    const simple_llm_step = b.step("llm-demo", "Run simple LLM demo");
    simple_llm_step.dependOn(&b.addRunArtifact(simple_llm_exe).step);

    // Create tiny model downloader executable
    const tiny_model_exe = b.addExecutable(.{
        .name = "download-tiny-model",
        .root_source_file = .{ .path = "download_tiny_model.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Create tiny model downloader step
    const tiny_model_step = b.step("download-model", "Download and test tiny LLM models");
    tiny_model_step.dependOn(&b.addRunArtifact(tiny_model_exe).step);

    // Create Qwen interactive chat executable
    const qwen_chat_exe = b.addExecutable(.{
        .name = "qwen-chat",
        .root_source_file = .{ .path = "qwen_chat.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Apply memory safety configuration to Qwen chat executable
    configureMemorySafety(qwen_chat_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);
    qwen_chat_exe.addModule("build_options", options_module);

    // Create Qwen chat step
    const qwen_chat_step = b.step("qwen-chat", "Start interactive chat with Qwen 0.5B");
    qwen_chat_step.dependOn(&b.addRunArtifact(qwen_chat_exe).step);

    // Create simple Qwen chat executable
    const qwen_simple_exe = b.addExecutable(.{
        .name = "qwen-simple-chat",
        .root_source_file = .{ .path = "qwen_simple_chat.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Apply memory safety configuration to simple Qwen chat executable
    configureMemorySafety(qwen_simple_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);
    qwen_simple_exe.addModule("build_options", options_module);

    // Create simple Qwen chat step
    const qwen_simple_step = b.step("qwen", "Start simple interactive chat with Qwen 0.5B");
    qwen_simple_step.dependOn(&b.addRunArtifact(qwen_simple_exe).step);

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

    // Apply memory safety configuration to debug ONNX executable
    configureMemorySafety(debug_onnx_exe, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);
    debug_onnx_exe.addModule("zig-onnx-parser", onnx_parser_module);
    debug_onnx_exe.addModule("build_options", options_module);

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

    // Apply memory safety configuration to CLI tests
    configureMemorySafety(cli_tests, enable_safety_checks, enable_stack_protection, enable_memory_debugging, enable_bounds_checking, enable_overflow_checks);
    cli_tests.addModule("build_options", options_module);

    const run_tests = b.addRunArtifact(cli_tests);
    test_step.dependOn(&run_tests.step);

    // Default step builds and installs the CLI
    b.default_step = b.getInstallStep();
}

const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the main library
    const lib = b.addStaticLibrary(.{
        .name = "zig-model-server",
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add dependencies
    const inference_engine = b.addModule("zig-inference-engine", .{
        .root_source_file = .{ .path = "../zig-inference-engine/src/lib.zig" },
    });
    lib.root_module.addImport("zig-inference-engine", inference_engine);

    // Install the library
    b.installArtifact(lib);

    // Create a module for external use
    const model_server_module = b.addModule("zig-model-server", .{
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });
    model_server_module.addImport("zig-inference-engine", inference_engine);

    // CLI executable
    const cli_exe = b.addExecutable(.{
        .name = "zig-model-server",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    cli_exe.root_module.addImport("zig-model-server", model_server_module);
    cli_exe.root_module.addImport("zig-inference-engine", inference_engine);

    const install_cli = b.addInstallArtifact(cli_exe, .{});
    const cli_step = b.step("cli", "Build CLI executable");
    cli_step.dependOn(&install_cli.step);

    // Unit tests
    const lib_unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });
    lib_unit_tests.root_module.addImport("zig-inference-engine", inference_engine);

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    // HTTP server tests
    const http_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/http/server.zig" },
        .target = target,
        .optimize = optimize,
    });
    http_tests.root_module.addImport("zig-inference-engine", inference_engine);

    const run_http_tests = b.addRunArtifact(http_tests);
    const http_test_step = b.step("test-http", "Run HTTP server tests");
    http_test_step.dependOn(&run_http_tests.step);

    // CLI tests
    const cli_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/cli/cli.zig" },
        .target = target,
        .optimize = optimize,
    });
    cli_tests.root_module.addImport("zig-inference-engine", inference_engine);

    const run_cli_tests = b.addRunArtifact(cli_tests);
    const cli_test_step = b.step("test-cli", "Run CLI tests");
    cli_test_step.dependOn(&run_cli_tests.step);

    // Model manager tests
    const model_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/models/manager.zig" },
        .target = target,
        .optimize = optimize,
    });
    model_tests.root_module.addImport("zig-inference-engine", inference_engine);

    const run_model_tests = b.addRunArtifact(model_tests);
    const model_test_step = b.step("test-models", "Run model manager tests");
    model_test_step.dependOn(&run_model_tests.step);

    // All tests
    const all_tests_step = b.step("test-all", "Run all tests");
    all_tests_step.dependOn(&run_lib_unit_tests.step);
    all_tests_step.dependOn(&run_http_tests.step);
    all_tests_step.dependOn(&run_cli_tests.step);
    all_tests_step.dependOn(&run_model_tests.step);

    // Examples
    const examples_step = b.step("examples", "Build examples");

    // Basic HTTP server example
    const basic_server_example = b.addExecutable(.{
        .name = "basic-server",
        .root_source_file = .{ .path = "examples/basic_server.zig" },
        .target = target,
        .optimize = optimize,
    });
    basic_server_example.root_module.addImport("zig-model-server", model_server_module);
    basic_server_example.root_module.addImport("zig-inference-engine", inference_engine);

    const install_basic_server = b.addInstallArtifact(basic_server_example, .{});
    examples_step.dependOn(&install_basic_server.step);

    // CLI usage example
    const cli_example = b.addExecutable(.{
        .name = "cli-example",
        .root_source_file = .{ .path = "examples/cli_usage.zig" },
        .target = target,
        .optimize = optimize,
    });
    cli_example.root_module.addImport("zig-model-server", model_server_module);
    cli_example.root_module.addImport("zig-inference-engine", inference_engine);

    const install_cli_example = b.addInstallArtifact(cli_example, .{});
    examples_step.dependOn(&install_cli_example.step);

    // Model management example
    const model_example = b.addExecutable(.{
        .name = "model-management",
        .root_source_file = .{ .path = "examples/model_management.zig" },
        .target = target,
        .optimize = optimize,
    });
    model_example.root_module.addImport("zig-model-server", model_server_module);
    model_example.root_module.addImport("zig-inference-engine", inference_engine);

    const install_model_example = b.addInstallArtifact(model_example, .{});
    examples_step.dependOn(&install_model_example.step);

    // Chat interface example
    const chat_example = b.addExecutable(.{
        .name = "chat-interface",
        .root_source_file = .{ .path = "examples/chat_interface.zig" },
        .target = target,
        .optimize = optimize,
    });
    chat_example.root_module.addImport("zig-model-server", model_server_module);
    chat_example.root_module.addImport("zig-inference-engine", inference_engine);

    const install_chat_example = b.addInstallArtifact(chat_example, .{});
    examples_step.dependOn(&install_chat_example.step);

    // Benchmarks
    const benchmark_step = b.step("benchmark", "Run benchmarks");

    const http_benchmark = b.addExecutable(.{
        .name = "http-benchmark",
        .root_source_file = .{ .path = "benchmarks/http_benchmark.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    http_benchmark.root_module.addImport("zig-model-server", model_server_module);
    http_benchmark.root_module.addImport("zig-inference-engine", inference_engine);

    const run_http_benchmark = b.addRunArtifact(http_benchmark);
    benchmark_step.dependOn(&run_http_benchmark.step);

    const model_benchmark = b.addExecutable(.{
        .name = "model-benchmark",
        .root_source_file = .{ .path = "benchmarks/model_benchmark.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    model_benchmark.root_module.addImport("zig-model-server", model_server_module);
    model_benchmark.root_module.addImport("zig-inference-engine", inference_engine);

    const run_model_benchmark = b.addRunArtifact(model_benchmark);
    benchmark_step.dependOn(&run_model_benchmark.step);

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

    // Integration tests
    const integration_test_step = b.step("test-integration", "Run integration tests");

    const integration_tests = b.addTest(.{
        .root_source_file = .{ .path = "tests/integration_test.zig" },
        .target = target,
        .optimize = optimize,
    });
    integration_tests.root_module.addImport("zig-model-server", model_server_module);
    integration_tests.root_module.addImport("zig-inference-engine", inference_engine);

    const run_integration_tests = b.addRunArtifact(integration_tests);
    integration_test_step.dependOn(&run_integration_tests.step);

    // Load tests
    const load_test_step = b.step("test-load", "Run load tests");

    const load_tests = b.addExecutable(.{
        .name = "load-test",
        .root_source_file = .{ .path = "tests/load_test.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    load_tests.root_module.addImport("zig-model-server", model_server_module);
    load_tests.root_module.addImport("zig-inference-engine", inference_engine);

    const run_load_tests = b.addRunArtifact(load_tests);
    load_test_step.dependOn(&run_load_tests.step);

    // Server runner for development
    const run_server = b.addRunArtifact(cli_exe);
    run_server.addArgs(&[_][]const u8{ "serve", "--host", "127.0.0.1", "--port", "8080", "--verbose" });
    const serve_step = b.step("serve", "Run development server");
    serve_step.dependOn(&run_server.step);

    // Quick start server
    const quick_start = b.addRunArtifact(cli_exe);
    quick_start.addArgs(&[_][]const u8{ "serve", "--port", "3000" });
    const quick_step = b.step("quick", "Quick start server on port 3000");
    quick_step.dependOn(&quick_start.step);

    // Production server
    const prod_server = b.addRunArtifact(cli_exe);
    prod_server.addArgs(&[_][]const u8{ "serve", "--host", "0.0.0.0", "--port", "8080", "--workers", "8", "--max-connections", "1000" });
    const prod_step = b.step("production", "Run production server");
    prod_step.dependOn(&prod_server.step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", "zig-out", "zig-cache" });
    clean_step.dependOn(&clean_cmd.step);

    // Install step
    const install_step = b.step("install", "Install the library and CLI");
    install_step.dependOn(&b.default_step.step);
    install_step.dependOn(&install_cli.step);

    // Development helpers
    const dev_step = b.step("dev", "Development build with all features");
    dev_step.dependOn(&test_step.step);
    dev_step.dependOn(&examples_step.step);
    dev_step.dependOn(&docs_step.step);
    dev_step.dependOn(&cli_step.step);

    // CI step
    const ci_step = b.step("ci", "Continuous integration build");
    ci_step.dependOn(&all_tests_step.step);
    ci_step.dependOn(&examples_step.step);
    ci_step.dependOn(&benchmark_step.step);
    ci_step.dependOn(&integration_test_step.step);

    // Release step
    const release_step = b.step("release", "Release build");
    release_step.dependOn(&all_tests_step.step);
    release_step.dependOn(&docs_step.step);
    release_step.dependOn(&install_step.step);
    release_step.dependOn(&integration_test_step.step);

    // Help step
    const help_step = b.step("help", "Show available build commands");
    const help_cmd = b.addSystemCommand(&[_][]const u8{
        "echo",
        \\Available commands:
        \\  zig build                 - Build library
        \\  zig build cli             - Build CLI executable
        \\  zig build test            - Run unit tests
        \\  zig build test-all        - Run all tests
        \\  zig build examples        - Build examples
        \\  zig build benchmark       - Run benchmarks
        \\  zig build docs            - Generate documentation
        \\  zig build serve           - Run development server
        \\  zig build quick           - Quick start server
        \\  zig build production      - Run production server
        \\  zig build dev             - Development build
        \\  zig build ci              - CI build
        \\  zig build release         - Release build
        \\  zig build clean           - Clean artifacts
        \\  zig build help            - Show this help
    });
    help_step.dependOn(&help_cmd.step);
}

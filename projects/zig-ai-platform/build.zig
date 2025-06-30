const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the main library
    const lib = b.addStaticLibrary(.{
        .name = "zig-ai-platform",
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add ecosystem dependencies
    const tensor_core = b.addModule("zig-tensor-core", .{
        .root_source_file = b.path("../zig-tensor-core/src/lib.zig"),
    });
    const onnx_parser = b.addModule("zig-onnx-parser", .{
        .root_source_file = b.path("../zig-onnx-parser/src/lib.zig"),
    });
    const inference_engine = b.addModule("zig-inference-engine", .{
        .root_source_file = b.path("../zig-inference-engine/src/lib.zig"),
    });
    const model_server = b.addModule("zig-model-server", .{
        .root_source_file = b.path("../zig-model-server/src/lib.zig"),
    });

    // Add dependencies to library
    lib.root_module.addImport("zig-tensor-core", tensor_core);
    lib.root_module.addImport("zig-onnx-parser", onnx_parser);
    lib.root_module.addImport("zig-inference-engine", inference_engine);
    lib.root_module.addImport("zig-model-server", model_server);

    // Install the library
    b.installArtifact(lib);

    // Create a module for external use
    const ai_platform_module = b.addModule("zig-ai-platform", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    ai_platform_module.addImport("zig-tensor-core", tensor_core);
    ai_platform_module.addImport("zig-onnx-parser", onnx_parser);
    ai_platform_module.addImport("zig-inference-engine", inference_engine);
    ai_platform_module.addImport("zig-model-server", model_server);

    // CLI executable
    const cli_exe = b.addExecutable(.{
        .name = "zig-ai-platform",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_exe.root_module.addImport("zig-tensor-core", tensor_core);
    cli_exe.root_module.addImport("zig-onnx-parser", onnx_parser);
    cli_exe.root_module.addImport("zig-inference-engine", inference_engine);
    cli_exe.root_module.addImport("zig-model-server", model_server);

    const install_cli = b.addInstallArtifact(cli_exe, .{});
    const cli_step = b.step("cli", "Build CLI executable");
    cli_step.dependOn(&install_cli.step);

    // Unit tests
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_unit_tests.root_module.addImport("zig-tensor-core", tensor_core);
    lib_unit_tests.root_module.addImport("zig-onnx-parser", onnx_parser);
    lib_unit_tests.root_module.addImport("zig-inference-engine", inference_engine);
    lib_unit_tests.root_module.addImport("zig-model-server", model_server);

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    // Platform core tests
    const platform_tests = b.addTest(.{
        .root_source_file = b.path("src/platform/core.zig"),
        .target = target,
        .optimize = optimize,
    });
    platform_tests.root_module.addImport("zig-tensor-core", tensor_core);
    platform_tests.root_module.addImport("zig-onnx-parser", onnx_parser);
    platform_tests.root_module.addImport("zig-inference-engine", inference_engine);
    platform_tests.root_module.addImport("zig-model-server", model_server);

    const run_platform_tests = b.addRunArtifact(platform_tests);
    const platform_test_step = b.step("test-platform", "Run platform tests");
    platform_test_step.dependOn(&run_platform_tests.step);

    // Configuration tests
    const config_tests = b.addTest(.{
        .root_source_file = b.path("src/config/manager.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_config_tests = b.addRunArtifact(config_tests);
    const config_test_step = b.step("test-config", "Run configuration tests");
    config_test_step.dependOn(&run_config_tests.step);

    // Services tests
    const services_tests = b.addTest(.{
        .root_source_file = b.path("src/services/health.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_services_tests = b.addRunArtifact(services_tests);
    const services_test_step = b.step("test-services", "Run services tests");
    services_test_step.dependOn(&run_services_tests.step);

    // CLI tests
    const cli_tests = b.addTest(.{
        .root_source_file = b.path("src/cli/cli.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_cli_tests = b.addRunArtifact(cli_tests);
    const cli_test_step = b.step("test-cli", "Run CLI tests");
    cli_test_step.dependOn(&run_cli_tests.step);

    // All tests
    const all_tests_step = b.step("test-all", "Run all tests");
    all_tests_step.dependOn(&run_lib_unit_tests.step);
    all_tests_step.dependOn(&run_platform_tests.step);
    all_tests_step.dependOn(&run_config_tests.step);
    all_tests_step.dependOn(&run_services_tests.step);
    all_tests_step.dependOn(&run_cli_tests.step);

    // Examples
    const examples_step = b.step("examples", "Build examples");

    // Basic platform example
    const basic_platform_example = b.addExecutable(.{
        .name = "basic-platform",
        .root_source_file = b.path("examples/basic_platform.zig"),
        .target = target,
        .optimize = optimize,
    });
    basic_platform_example.root_module.addImport("zig-ai-platform", ai_platform_module);

    const install_basic_platform = b.addInstallArtifact(basic_platform_example, .{});
    examples_step.dependOn(&install_basic_platform.step);

    // IoT deployment example
    const iot_example = b.addExecutable(.{
        .name = "iot-deployment",
        .root_source_file = b.path("examples/iot_deployment.zig"),
        .target = target,
        .optimize = optimize,
    });
    iot_example.root_module.addImport("zig-ai-platform", ai_platform_module);

    const install_iot_example = b.addInstallArtifact(iot_example, .{});
    examples_step.dependOn(&install_iot_example.step);

    // Production deployment example
    const production_example = b.addExecutable(.{
        .name = "production-deployment",
        .root_source_file = b.path("examples/production_deployment.zig"),
        .target = target,
        .optimize = optimize,
    });
    production_example.root_module.addImport("zig-ai-platform", ai_platform_module);

    const install_production_example = b.addInstallArtifact(production_example, .{});
    examples_step.dependOn(&install_production_example.step);

    // Configuration management example
    const config_example = b.addExecutable(.{
        .name = "config-management",
        .root_source_file = b.path("examples/config_management.zig"),
        .target = target,
        .optimize = optimize,
    });
    config_example.root_module.addImport("zig-ai-platform", ai_platform_module);

    const install_config_example = b.addInstallArtifact(config_example, .{});
    examples_step.dependOn(&install_config_example.step);

    // Integration tests
    const integration_test_step = b.step("test-integration", "Run integration tests");
    
    const integration_tests = b.addTest(.{
        .root_source_file = b.path("tests/integration_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_tests.root_module.addImport("zig-ai-platform", ai_platform_module);

    const run_integration_tests = b.addRunArtifact(integration_tests);
    integration_test_step.dependOn(&run_integration_tests.step);

    // End-to-end tests
    const e2e_test_step = b.step("test-e2e", "Run end-to-end tests");
    
    const e2e_tests = b.addTest(.{
        .root_source_file = b.path("tests/e2e_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    e2e_tests.root_module.addImport("zig-ai-platform", ai_platform_module);

    const run_e2e_tests = b.addRunArtifact(e2e_tests);
    e2e_test_step.dependOn(&run_e2e_tests.step);

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

    // Platform commands
    const init_cmd = b.addRunArtifact(cli_exe);
    init_cmd.addArgs(&[_][]const u8{ "init" });
    const init_step = b.step("init", "Initialize platform");
    init_step.dependOn(&init_cmd.step);

    const start_dev_cmd = b.addRunArtifact(cli_exe);
    start_dev_cmd.addArgs(&[_][]const u8{ "start", "--env", "development" });
    const start_dev_step = b.step("start-dev", "Start development platform");
    start_dev_step.dependOn(&start_dev_cmd.step);

    const start_prod_cmd = b.addRunArtifact(cli_exe);
    start_prod_cmd.addArgs(&[_][]const u8{ "start", "--env", "production" });
    const start_prod_step = b.step("start-prod", "Start production platform");
    start_prod_step.dependOn(&start_prod_cmd.step);

    const status_cmd = b.addRunArtifact(cli_exe);
    status_cmd.addArgs(&[_][]const u8{ "status" });
    const status_step = b.step("status", "Show platform status");
    status_step.dependOn(&status_cmd.step);

    const health_cmd = b.addRunArtifact(cli_exe);
    health_cmd.addArgs(&[_][]const u8{ "health" });
    const health_step = b.step("health", "Check platform health");
    health_step.dependOn(&health_cmd.step);

    // Deployment commands
    const deploy_iot_cmd = b.addRunArtifact(cli_exe);
    deploy_iot_cmd.addArgs(&[_][]const u8{ "deploy", "--env", "production", "--target", "iot" });
    const deploy_iot_step = b.step("deploy-iot", "Deploy to IoT");
    deploy_iot_step.dependOn(&deploy_iot_cmd.step);

    const deploy_server_cmd = b.addRunArtifact(cli_exe);
    deploy_server_cmd.addArgs(&[_][]const u8{ "deploy", "--env", "production", "--target", "server" });
    const deploy_server_step = b.step("deploy-server", "Deploy to server");
    deploy_server_step.dependOn(&deploy_server_cmd.step);

    // Configuration commands
    const config_gen_cmd = b.addRunArtifact(cli_exe);
    config_gen_cmd.addArgs(&[_][]const u8{ "config", "generate", "--env", "production" });
    const config_gen_step = b.step("config-gen", "Generate configuration");
    config_gen_step.dependOn(&config_gen_cmd.step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", "zig-out", "zig-cache" });
    clean_step.dependOn(&clean_cmd.step);

    // Install step
    const install_step = b.step("install", "Install the platform and CLI");
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
    ci_step.dependOn(&integration_test_step.step);
    ci_step.dependOn(&e2e_test_step.step);

    // Release step
    const release_step = b.step("release", "Release build");
    release_step.dependOn(&all_tests_step.step);
    release_step.dependOn(&docs_step.step);
    release_step.dependOn(&install_step.step);
    release_step.dependOn(&integration_test_step.step);
    release_step.dependOn(&e2e_test_step.step);

    // Help step
    const help_step = b.step("help", "Show available build commands");
    const help_cmd = b.addSystemCommand(&[_][]const u8{ "echo", 
        \\Available commands:
        \\  zig build                 - Build platform library
        \\  zig build cli             - Build CLI executable
        \\  zig build test            - Run unit tests
        \\  zig build test-all        - Run all tests
        \\  zig build examples        - Build examples
        \\  zig build docs            - Generate documentation
        \\  zig build init            - Initialize platform
        \\  zig build start-dev       - Start development platform
        \\  zig build start-prod      - Start production platform
        \\  zig build status          - Show platform status
        \\  zig build health          - Check platform health
        \\  zig build deploy-iot      - Deploy to IoT
        \\  zig build deploy-server   - Deploy to server
        \\  zig build config-gen      - Generate configuration
        \\  zig build dev             - Development build
        \\  zig build ci              - CI build
        \\  zig build release         - Release build
        \\  zig build clean           - Clean artifacts
        \\  zig build help            - Show this help
    });
    help_step.dependOn(&help_cmd.step);
}

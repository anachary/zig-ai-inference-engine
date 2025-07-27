const std = @import("std");

/// Zig AI Platform - Unified Build System with New Framework Architecture
///
/// This build file integrates the new modular framework with existing projects
/// while maintaining backward compatibility.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // === FRAMEWORK INTEGRATION ===
    
    // Framework library
    const framework_lib = b.addStaticLibrary(.{
        .name = "zig-ai-framework",
        .root_source_file = b.path("framework/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(framework_lib);

    // Framework module
    const framework_module = b.addModule("zig-ai-framework", .{
        .root_source_file = b.path("framework/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Implementations library
    const implementations_lib = b.addStaticLibrary(.{
        .name = "zig-ai-implementations",
        .root_source_file = b.path("implementations/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    implementations_lib.root_module.addImport("framework", framework_module);
    b.installArtifact(implementations_lib);

    // Implementations module
    const implementations_module = b.addModule("zig-ai-implementations", .{
        .root_source_file = b.path("implementations/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    implementations_module.addImport("framework", framework_module);

    // === EXISTING PROJECTS INTEGRATION ===
    
    // Add existing projects as dependencies (backward compatibility)
    const projects = [_][]const u8{
        "zig-tensor-core",
        "zig-onnx-parser", 
        "zig-inference-engine",
        "zig-model-server",
    };

    var project_modules = std.ArrayList(*std.Build.Module).init(b.allocator);
    defer project_modules.deinit();

    for (projects) |project_name| {
        // Create module for each project
        const project_module = b.addModule(project_name, .{
            .root_source_file = b.path(b.fmt("projects/{s}/src/main.zig", .{project_name})),
            .target = target,
            .optimize = optimize,
        });
        
        // Add framework dependencies to existing projects
        project_module.addImport("framework", framework_module);
        project_module.addImport("implementations", implementations_module);
        
        project_modules.append(project_module) catch unreachable;
    }

    // === MAIN PLATFORM MODULE ===
    
    // Create the main zig-ai-platform module
    const zig_ai_platform = b.addModule("zig-ai-platform", .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add all dependencies to main platform
    zig_ai_platform.addImport("framework", framework_module);
    zig_ai_platform.addImport("implementations", implementations_module);
    
    for (projects, project_modules.items) |project_name, project_module| {
        zig_ai_platform.addImport(project_name, project_module);
    }

    // === EXECUTABLES ===

    // Main CLI executable (preserving existing functionality)
    const cli_exe = b.addExecutable(.{
        .name = "zig-ai",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add all modules to CLI
    cli_exe.root_module.addImport("zig-ai-platform", zig_ai_platform);
    cli_exe.root_module.addImport("framework", framework_module);
    cli_exe.root_module.addImport("implementations", implementations_module);
    
    for (projects, project_modules.items) |project_name, project_module| {
        cli_exe.root_module.addImport(project_name, project_module);
    }
    
    b.installArtifact(cli_exe);

    // Framework demo executable
    const framework_demo = b.addExecutable(.{
        .name = "framework-demo",
        .root_source_file = b.path("examples/framework_demo.zig"),
        .target = target,
        .optimize = optimize,
    });
    framework_demo.root_module.addImport("framework", framework_module);
    framework_demo.root_module.addImport("implementations", implementations_module);
    
    const install_demo = b.addInstallArtifact(framework_demo, .{});

    // === TESTING ===

    // Framework tests
    const framework_tests = b.addTest(.{
        .root_source_file = b.path("framework/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const framework_test_step = b.step("test-framework", "Run framework tests");
    framework_test_step.dependOn(&b.addRunArtifact(framework_tests).step);

    // Implementation tests
    const impl_tests = b.addTest(.{
        .root_source_file = b.path("implementations/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    impl_tests.root_module.addImport("framework", framework_module);

    const impl_test_step = b.step("test-implementations", "Run implementation tests");
    impl_test_step.dependOn(&b.addRunArtifact(impl_tests).step);

    // Main platform tests
    const main_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_tests.root_module.addImport("zig-ai-platform", zig_ai_platform);
    main_tests.root_module.addImport("framework", framework_module);
    main_tests.root_module.addImport("implementations", implementations_module);

    const main_test_step = b.step("test-main", "Run main platform tests");
    main_test_step.dependOn(&b.addRunArtifact(main_tests).step);

    // Combined test step
    const test_all_step = b.step("test-all", "Run all tests");
    test_all_step.dependOn(framework_test_step);
    test_all_step.dependOn(impl_test_step);
    test_all_step.dependOn(main_test_step);

    // Default test step (backward compatibility)
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(test_all_step);

    // === EXAMPLES AND DEMOS ===

    const examples_step = b.step("examples", "Build all examples");
    examples_step.dependOn(&install_demo.step);

    // Operator example
    const operator_example = b.addExecutable(.{
        .name = "operator-example",
        .root_source_file = b.path("examples/operator_example.zig"),
        .target = target,
        .optimize = optimize,
    });
    operator_example.root_module.addImport("framework", framework_module);
    operator_example.root_module.addImport("implementations", implementations_module);
    
    const install_operator_example = b.addInstallArtifact(operator_example, .{});
    examples_step.dependOn(&install_operator_example.step);

    // === BENCHMARKS ===

    const benchmark_step = b.step("benchmark", "Run benchmarks");

    const framework_benchmark = b.addExecutable(.{
        .name = "framework-benchmark",
        .root_source_file = b.path("benchmarks/framework_benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    framework_benchmark.root_module.addImport("framework", framework_module);
    framework_benchmark.root_module.addImport("implementations", implementations_module);

    const run_framework_benchmark = b.addRunArtifact(framework_benchmark);
    benchmark_step.dependOn(&run_framework_benchmark.step);

    // === RUN COMMANDS ===

    // Run main CLI (backward compatibility)
    const run_cmd = b.addRunArtifact(cli_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the main CLI");
    run_step.dependOn(&run_cmd.step);

    // Run framework demo
    const run_demo_cmd = b.addRunArtifact(framework_demo);
    const run_demo_step = b.step("run-demo", "Run framework demo");
    run_demo_step.dependOn(&run_demo_cmd.step);

    // === DOCUMENTATION ===

    const docs_step = b.step("docs", "Generate documentation");
    
    // Framework docs
    const framework_docs = framework_lib.getEmittedDocs();
    const install_framework_docs = b.addInstallDirectory(.{
        .source_dir = framework_docs,
        .install_dir = .prefix,
        .install_subdir = "docs/framework",
    });
    docs_step.dependOn(&install_framework_docs.step);

    // Implementation docs
    const impl_docs = implementations_lib.getEmittedDocs();
    const install_impl_docs = b.addInstallDirectory(.{
        .source_dir = impl_docs,
        .install_dir = .prefix,
        .install_subdir = "docs/implementations",
    });
    docs_step.dependOn(&install_impl_docs.step);

    // === INSTALLATION TARGETS ===

    const install_all_step = b.step("install-all", "Install all artifacts");
    install_all_step.dependOn(b.getInstallStep());
    install_all_step.dependOn(&install_demo.step);
    install_all_step.dependOn(&install_operator_example.step);

    // === CLEAN STEP ===

    const clean_step = b.step("clean", "Clean build artifacts");
    // Note: Zig 0.11 doesn't have a built-in clean step, but we can document it
    _ = clean_step;

    // === MIGRATION HELPERS ===

    // Step to help migrate existing code to new framework
    const migration_step = b.step("migrate", "Show migration guide");
    const migration_cmd = b.addSystemCommand(&[_][]const u8{
        "echo", 
        "Migration Guide:\n" ++
        "1. Update imports: const framework = @import(\"framework\");\n" ++
        "2. Update imports: const implementations = @import(\"implementations\");\n" ++
        "3. Use implementations.AIPlatform for complete functionality\n" ++
        "4. See examples/ directory for usage patterns\n" ++
        "5. Run 'zig build test-all' to verify migration"
    });
    migration_step.dependOn(&migration_cmd.step);

    // === DEVELOPMENT HELPERS ===

    // Format code
    const fmt_step = b.step("fmt", "Format source code");
    const fmt_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "fmt", "." });
    fmt_step.dependOn(&fmt_cmd.step);

    // Check code
    const check_step = b.step("check", "Check code for errors");
    const check_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "build", "--summary", "none" });
    check_step.dependOn(&check_cmd.step);
}

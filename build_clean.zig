const std = @import("std");

/// Clean, Simple Build System for Zig AI Platform
/// 
/// This build file implements the clean 4-directory structure:
/// 1. framework/      - Core framework and interfaces
/// 2. implementations/ - Concrete implementations  
/// 3. docs/           - All documentation
/// 4. examples/       - Real-world examples (iot, aks)
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // === 1. FRAMEWORK ===
    
    // Framework library
    const framework_lib = b.addStaticLibrary(.{
        .name = "zig-ai-framework",
        .root_source_file = b.path("framework/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(framework_lib);

    // Framework module
    const framework_module = b.addModule("framework", .{
        .root_source_file = b.path("framework/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === 2. IMPLEMENTATIONS ===
    
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
    const implementations_module = b.addModule("implementations", .{
        .root_source_file = b.path("implementations/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    implementations_module.addImport("framework", framework_module);

    // === 3. MAIN PLATFORM ===
    
    // Main CLI executable (preserving existing functionality)
    const main_exe = b.addExecutable(.{
        .name = "zig-ai",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_exe.root_module.addImport("framework", framework_module);
    main_exe.root_module.addImport("implementations", implementations_module);
    
    // Add existing project dependencies for backward compatibility
    const projects = [_][]const u8{
        "zig-tensor-core",
        "zig-onnx-parser", 
        "zig-inference-engine",
        "zig-model-server",
    };
    
    for (projects) |project_name| {
        const project_module = b.addModule(project_name, .{
            .root_source_file = b.path(b.fmt("projects/{s}/src/main.zig", .{project_name})),
            .target = target,
            .optimize = optimize,
        });
        main_exe.root_module.addImport(project_name, project_module);
    }
    
    b.installArtifact(main_exe);

    // === 4. EXAMPLES ===

    const examples_step = b.step("examples", "Build all examples");

    // Framework demo
    const framework_demo = b.addExecutable(.{
        .name = "framework-demo",
        .root_source_file = b.path("examples/framework_demo.zig"),
        .target = target,
        .optimize = optimize,
    });
    framework_demo.root_module.addImport("framework", framework_module);
    framework_demo.root_module.addImport("implementations", implementations_module);
    
    const install_demo = b.addInstallArtifact(framework_demo, .{});
    examples_step.dependOn(&install_demo.step);

    // IoT Examples
    const iot_examples_step = b.step("examples-iot", "Build IoT examples");
    
    // Raspberry Pi Tiny LLM
    const pi_tiny_llm = b.addExecutable(.{
        .name = "pi-tiny-llm",
        .root_source_file = b.path("examples/iot/raspberry-pi/tiny-llm/src/main.zig"),
        .target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux }),
        .optimize = .ReleaseSmall, // Optimize for size on edge devices
    });
    pi_tiny_llm.root_module.addImport("framework", framework_module);
    pi_tiny_llm.root_module.addImport("implementations", implementations_module);
    
    const install_pi_llm = b.addInstallArtifact(pi_tiny_llm, .{});
    iot_examples_step.dependOn(&install_pi_llm.step);
    examples_step.dependOn(&install_pi_llm.step);

    // Edge inference example
    const edge_inference = b.addExecutable(.{
        .name = "edge-inference",
        .root_source_file = b.path("examples/iot/edge-inference/src/main.zig"),
        .target = target,
        .optimize = .ReleaseSmall,
    });
    edge_inference.root_module.addImport("framework", framework_module);
    edge_inference.root_module.addImport("implementations", implementations_module);
    
    const install_edge = b.addInstallArtifact(edge_inference, .{});
    iot_examples_step.dependOn(&install_edge.step);
    examples_step.dependOn(&install_edge.step);

    // AKS Examples
    const aks_examples_step = b.step("examples-aks", "Build AKS examples");
    
    // Distributed inference coordinator
    const coordinator = b.addExecutable(.{
        .name = "coordinator",
        .root_source_file = b.path("examples/aks/distributed-inference/model-sharding/src/coordinator/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    coordinator.root_module.addImport("framework", framework_module);
    coordinator.root_module.addImport("implementations", implementations_module);
    
    const install_coordinator = b.addInstallArtifact(coordinator, .{});
    aks_examples_step.dependOn(&install_coordinator.step);
    examples_step.dependOn(&install_coordinator.step);

    // Distributed inference worker
    const worker = b.addExecutable(.{
        .name = "worker",
        .root_source_file = b.path("examples/aks/distributed-inference/model-sharding/src/worker/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    worker.root_module.addImport("framework", framework_module);
    worker.root_module.addImport("implementations", implementations_module);
    
    const install_worker = b.addInstallArtifact(worker, .{});
    aks_examples_step.dependOn(&install_worker.step);
    examples_step.dependOn(&install_worker.step);

    // === 5. TESTING ===

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

    // === 6. BENCHMARKS ===

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

    // === 7. RUN COMMANDS ===

    // Run main CLI (backward compatibility)
    const run_cmd = b.addRunArtifact(main_exe);
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

    // === 8. DOCUMENTATION ===

    const docs_step = b.step("docs", "Generate documentation");
    
    // Framework docs
    const framework_docs = framework_lib.getEmittedDocs();
    const install_framework_docs = b.addInstallDirectory(.{
        .source_dir = framework_docs,
        .install_dir = .prefix,
        .install_subdir = "docs/api/framework",
    });
    docs_step.dependOn(&install_framework_docs.step);

    // Implementation docs
    const impl_docs = implementations_lib.getEmittedDocs();
    const install_impl_docs = b.addInstallDirectory(.{
        .source_dir = impl_docs,
        .install_dir = .prefix,
        .install_subdir = "docs/api/implementations",
    });
    docs_step.dependOn(&install_impl_docs.step);

    // === 9. CROSS-COMPILATION ===

    const cross_step = b.step("cross", "Cross-compile for multiple targets");

    const cross_targets = [_]std.Target.Query{
        .{ .cpu_arch = .x86_64, .os_tag = .linux },
        .{ .cpu_arch = .aarch64, .os_tag = .linux },
        .{ .cpu_arch = .x86_64, .os_tag = .windows },
        .{ .cpu_arch = .x86_64, .os_tag = .macos },
        .{ .cpu_arch = .aarch64, .os_tag = .macos },
    };

    for (cross_targets, 0..) |cross_target, i| {
        const cross_exe = b.addExecutable(.{
            .name = b.fmt("zig-ai-{s}-{s}", .{ @tagName(cross_target.cpu_arch.?), @tagName(cross_target.os_tag.?) }),
            .root_source_file = b.path("src/main.zig"),
            .target = b.resolveTargetQuery(cross_target),
            .optimize = .ReleaseSmall,
        });
        cross_exe.root_module.addImport("framework", framework_module);
        cross_exe.root_module.addImport("implementations", implementations_module);

        const cross_install = b.addInstallArtifact(cross_exe, .{});
        cross_step.dependOn(&cross_install.step);

        const cross_target_step = b.step(b.fmt("cross-{d}", .{i}), b.fmt("Cross-compile for {s}-{s}", .{ @tagName(cross_target.cpu_arch.?), @tagName(cross_target.os_tag.?) }));
        cross_target_step.dependOn(&cross_install.step);
    }

    // === 10. DEVELOPMENT HELPERS ===

    // Format code
    const fmt_step = b.step("fmt", "Format source code");
    const fmt_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "fmt", "." });
    fmt_step.dependOn(&fmt_cmd.step);

    // Check code
    const check_step = b.step("check", "Check code for errors");
    const check_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "build", "--summary", "none" });
    check_step.dependOn(&check_cmd.step);

    // Clean build artifacts
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", "zig-out", "zig-cache" });
    clean_step.dependOn(&clean_cmd.step);

    // === 11. INSTALLATION ===

    const install_all_step = b.step("install-all", "Install all artifacts");
    install_all_step.dependOn(b.getInstallStep());
    install_all_step.dependOn(examples_step);

    // === 12. HELP ===

    const help_step = b.step("help", "Show available build commands");
    const help_cmd = b.addSystemCommand(&[_][]const u8{
        "echo", 
        "Zig AI Platform Build Commands:\n" ++
        "\n" ++
        "🏗️  Building:\n" ++
        "  zig build                    # Build main platform\n" ++
        "  zig build examples           # Build all examples\n" ++
        "  zig build examples-iot       # Build IoT examples\n" ++
        "  zig build examples-aks       # Build AKS examples\n" ++
        "\n" ++
        "🧪 Testing:\n" ++
        "  zig build test               # Run all tests\n" ++
        "  zig build test-framework     # Test framework only\n" ++
        "  zig build test-implementations # Test implementations only\n" ++
        "\n" ++
        "🚀 Running:\n" ++
        "  zig build run                # Run main CLI\n" ++
        "  zig build run-demo           # Run framework demo\n" ++
        "\n" ++
        "📚 Documentation:\n" ++
        "  zig build docs               # Generate API docs\n" ++
        "\n" ++
        "⚡ Performance:\n" ++
        "  zig build benchmark          # Run benchmarks\n" ++
        "\n" ++
        "🌍 Cross-compilation:\n" ++
        "  zig build cross              # All targets\n" ++
        "  zig build cross-0            # Linux x86_64\n" ++
        "  zig build cross-1            # Linux ARM64\n" ++
        "\n" ++
        "🔧 Development:\n" ++
        "  zig build fmt                # Format code\n" ++
        "  zig build check              # Check for errors\n" ++
        "  zig build clean              # Clean artifacts\n" ++
        "\n" ++
        "📁 Structure:\n" ++
        "  framework/      - Core framework\n" ++
        "  implementations/ - Concrete implementations\n" ++
        "  docs/           - Documentation\n" ++
        "  examples/       - Real-world examples\n"
    });
    help_step.dependOn(&help_cmd.step);
}

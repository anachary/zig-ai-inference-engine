const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main library
    const lib = b.addStaticLibrary(.{
        .name = "zig-ai-inference",
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });

    // TODO: Add SIMD support when C files are implemented
    // if (target.getCpuArch() == .x86_64) {
    //     lib.addCSourceFile(.{
    //         .file = .{ .path = "src/core/simd_x86.c" },
    //         .flags = &[_][]const u8{ "-mavx2", "-mfma" },
    //     });
    // }

    b.installArtifact(lib);

    // Main executable
    const exe = b.addExecutable(.{
        .name = "ai-engine",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    exe.linkLibrary(lib);
    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/test.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Integration Tests
    const integration_tests = b.addTest(.{
        .root_source_file = .{ .path = "tests/integration_test.zig" },
        .target = target,
        .optimize = optimize,
    });
    integration_tests.linkLibrary(lib);
    integration_tests.addModule("zig-ai-inference", b.createModule(.{
        .source_file = .{ .path = "src/lib.zig" },
    }));

    const run_integration_tests = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("test-integration", "Run comprehensive integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // Benchmarks
    const benchmarks = b.addExecutable(.{
        .name = "benchmarks",
        .root_source_file = .{ .path = "benchmarks/main.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });

    benchmarks.linkLibrary(lib);
    benchmarks.addModule("zig-ai-inference", b.createModule(.{
        .source_file = .{ .path = "src/lib.zig" },
    }));
    const run_benchmarks = b.addRunArtifact(benchmarks);
    const bench_step = b.step("bench", "Run performance benchmarks");
    bench_step.dependOn(&run_benchmarks.step);

    // Main CLI
    const main_cli = b.addExecutable(.{
        .name = "zig-ai",
        .root_source_file = .{ .path = "examples/zig_ai_cli.zig" },
        .target = target,
        .optimize = optimize,
    });
    main_cli.linkLibrary(lib);
    main_cli.addModule("zig-ai-inference", b.createModule(.{
        .source_file = .{ .path = "src/lib.zig" },
    }));
    b.installArtifact(main_cli);

    const main_cli_run = b.addRunArtifact(main_cli);
    if (b.args) |args| {
        main_cli_run.addArgs(args);
    }
    const main_cli_step = b.step("cli", "Run the main Zig AI CLI");
    main_cli_step.dependOn(&main_cli_run.step);

    // Examples
    const examples = [_][]const u8{
        "simple_inference",
        "model_loading",
        "computation_graph",
        "enhanced_operators",
        "gpu_demo",
        "zig_ai_cli", // Main unified CLI
    };

    for (examples) |example| {
        const example_exe = b.addExecutable(.{
            .name = example,
            .root_source_file = .{ .path = b.fmt("examples/{s}.zig", .{example}) },
            .target = target,
            .optimize = optimize,
        });

        example_exe.linkLibrary(lib);
        example_exe.addModule("zig-ai-inference", b.createModule(.{
            .source_file = .{ .path = "src/lib.zig" },
        }));
        b.installArtifact(example_exe);

        const example_run = b.addRunArtifact(example_exe);
        const example_step = b.step(b.fmt("run-{s}", .{example}), b.fmt("Run {s} example", .{example}));
        example_step.dependOn(&example_run.step);
    }
}

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main library
    const lib = b.addStaticLibrary(.{
        .name = "zig-ai-engine",
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

    // Benchmarks
    const benchmarks = b.addExecutable(.{
        .name = "benchmarks",
        .root_source_file = .{ .path = "benchmarks/main.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });

    benchmarks.linkLibrary(lib);
    const run_benchmarks = b.addRunArtifact(benchmarks);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_benchmarks.step);

    // Examples
    const examples = [_][]const u8{
        "simple_inference",
        "model_loading",
        "custom_operator",
        "phase1_demo",
    };

    for (examples) |example| {
        const example_exe = b.addExecutable(.{
            .name = example,
            .root_source_file = .{ .path = b.fmt("examples/{s}.zig", .{example}) },
            .target = target,
            .optimize = optimize,
        });

        example_exe.linkLibrary(lib);
        example_exe.addModule("zig-ai-engine", b.createModule(.{
            .source_file = .{ .path = "src/lib.zig" },
        }));
        b.installArtifact(example_exe);

        const example_run = b.addRunArtifact(example_exe);
        const example_step = b.step(b.fmt("run-{s}", .{example}), b.fmt("Run {s} example", .{example}));
        example_step.dependOn(&example_run.step);
    }
}

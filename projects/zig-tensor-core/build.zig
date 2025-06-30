const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module
    const tensor_core = b.addModule("zig-tensor-core", .{
        .source_file = .{ .path = "src/lib.zig" },
        .dependencies = &.{},
    });

    // Static library
    const lib = b.addStaticLibrary(.{
        .name = "zig-tensor-core",
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Enable SIMD optimizations for release builds
    if (optimize != .Debug) {
        switch (target.getCpuArch()) {
            .x86_64 => {
                lib.addCSourceFile(.{
                    .file = .{ .path = "src/core/simd_x86.c" },
                    .flags = &.{ "-mavx2", "-mfma" },
                });
            },
            .aarch64 => {
                lib.addCSourceFile(.{
                    .file = .{ .path = "src/core/simd_arm.c" },
                    .flags = &.{"-march=armv8-a+simd"},
                });
            },
            else => {},
        }
    }

    b.installArtifact(lib);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    // Test step
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Benchmark executable
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = .{ .path = "examples/benchmark.zig" },
        .target = target,
        .optimize = optimize,
    });
    benchmark.addModule("zig-tensor-core", tensor_core);

    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);

    // Examples
    const examples = [_][]const u8{
        "basic_usage",
        "memory_management", 
        "simd_operations",
        "tensor_operations",
    };

    for (examples) |example_name| {
        const example = b.addExecutable(.{
            .name = example_name,
            .root_source_file = .{ .path = b.fmt("examples/{s}.zig", .{example_name}) },
            .target = target,
            .optimize = optimize,
        });
        example.addModule("zig-tensor-core", tensor_core);

        const run_example = b.addRunArtifact(example);
        const example_step = b.step(b.fmt("run-{s}", .{example_name}), b.fmt("Run {s} example", .{example_name}));
        example_step.dependOn(&run_example.step);

        b.installArtifact(example);
    }

    // Documentation generation
    const docs = b.addTest(.{
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = .Debug,
    });

    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&docs.step);

    // Performance tests
    const perf_tests = b.addExecutable(.{
        .name = "perf-tests",
        .root_source_file = .{ .path = "tests/performance.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    perf_tests.addModule("zig-tensor-core", tensor_core);

    const run_perf_tests = b.addRunArtifact(perf_tests);
    const perf_step = b.step("perf", "Run performance tests");
    perf_step.dependOn(&run_perf_tests.step);

    // Memory leak detection (debug builds only)
    if (optimize == .Debug) {
        const leak_tests = b.addTest(.{
            .root_source_file = .{ .path = "tests/memory_leaks.zig" },
            .target = target,
            .optimize = optimize,
        });
        leak_tests.addModule("zig-tensor-core", tensor_core);

        const run_leak_tests = b.addRunArtifact(leak_tests);
        const leak_step = b.step("test-leaks", "Run memory leak tests");
        leak_step.dependOn(&run_leak_tests.step);
    }

    // Cross-compilation targets for IoT/embedded
    const cross_targets = [_]std.zig.CrossTarget{
        .{ .cpu_arch = .aarch64, .os_tag = .linux },
        .{ .cpu_arch = .arm, .os_tag = .linux },
        .{ .cpu_arch = .riscv64, .os_tag = .linux },
    };

    for (cross_targets, 0..) |cross_target, i| {
        const cross_lib = b.addStaticLibrary(.{
            .name = b.fmt("zig-tensor-core-{s}-{s}", .{ @tagName(cross_target.cpu_arch.?), @tagName(cross_target.os_tag.?) }),
            .root_source_file = .{ .path = "src/lib.zig" },
            .target = cross_target,
            .optimize = .ReleaseSmall, // Optimize for size on embedded targets
        });

        const cross_step = b.step(b.fmt("cross-{d}", .{i}), b.fmt("Cross-compile for {s}-{s}", .{ @tagName(cross_target.cpu_arch.?), @tagName(cross_target.os_tag.?) }));
        cross_step.dependOn(&cross_lib.step);

        b.installArtifact(cross_lib);
    }

    // Integration with common interfaces
    const common_interfaces = b.createModule(.{
        .source_file = .{ .path = "../../common/interfaces/tensor.zig" },
    });

    // Add common interfaces to the library
    tensor_core.dependencies.put("common-interfaces", common_interfaces) catch unreachable;

    // Fuzzing tests (if available)
    if (b.option(bool, "fuzz", "Enable fuzzing tests") orelse false) {
        const fuzz_tests = b.addExecutable(.{
            .name = "fuzz-tests",
            .root_source_file = .{ .path = "tests/fuzz.zig" },
            .target = target,
            .optimize = optimize,
        });
        fuzz_tests.addModule("zig-tensor-core", tensor_core);

        const run_fuzz_tests = b.addRunArtifact(fuzz_tests);
        const fuzz_step = b.step("fuzz", "Run fuzzing tests");
        fuzz_step.dependOn(&run_fuzz_tests.step);
    }

    // Code coverage (debug builds only)
    if (optimize == .Debug and b.option(bool, "coverage", "Enable code coverage") orelse false) {
        unit_tests.test_runner = .{ .path = "tests/coverage_runner.zig" };
        
        const coverage_step = b.step("coverage", "Generate code coverage report");
        coverage_step.dependOn(&run_unit_tests.step);
    }

    // Static analysis
    const analyze = b.addSystemCommand(&.{ "zig", "fmt", "--check", "src/" });
    const analyze_step = b.step("analyze", "Run static analysis");
    analyze_step.dependOn(&analyze.step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    clean_step.dependOn(&b.addRemoveDirTree(b.install_path).step);

    // All tests step
    const all_tests_step = b.step("test-all", "Run all tests");
    all_tests_step.dependOn(test_step);
    all_tests_step.dependOn(perf_step);
    if (optimize == .Debug) {
        // Add leak tests only for debug builds
        const leak_tests = b.addTest(.{
            .root_source_file = .{ .path = "tests/memory_leaks.zig" },
            .target = target,
            .optimize = optimize,
        });
        leak_tests.addModule("zig-tensor-core", tensor_core);
        all_tests_step.dependOn(&b.addRunArtifact(leak_tests).step);
    }

    // Default step
    b.default_step.dependOn(&lib.step);
    b.default_step.dependOn(test_step);
}

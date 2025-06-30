const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module
    const onnx_parser = b.addModule("zig-onnx-parser", .{
        .source_file = .{ .path = "src/lib.zig" },
        .dependencies = &.{},
    });

    // Add dependency on tensor-core for data type compatibility
    const tensor_core = b.dependency("zig-tensor-core", .{
        .target = target,
        .optimize = optimize,
    });
    onnx_parser.dependencies.put("zig-tensor-core", tensor_core.module("zig-tensor-core")) catch unreachable;

    // Static library
    const lib = b.addStaticLibrary(.{
        .name = "zig-onnx-parser",
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });

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

    // Examples
    const examples = [_][]const u8{
        "basic_parsing",
        "model_validation",
        "model_optimization",
        "format_conversion",
    };

    for (examples) |example_name| {
        const example = b.addExecutable(.{
            .name = example_name,
            .root_source_file = .{ .path = b.fmt("examples/{s}.zig", .{example_name}) },
            .target = target,
            .optimize = optimize,
        });
        example.addModule("zig-onnx-parser", onnx_parser);

        const run_example = b.addRunArtifact(example);
        const example_step = b.step(b.fmt("run-{s}", .{example_name}), b.fmt("Run {s} example", .{example_name}));
        example_step.dependOn(&run_example.step);

        b.installArtifact(example);
    }

    // Model testing with real ONNX files
    const model_tests = b.addExecutable(.{
        .name = "model-tests",
        .root_source_file = .{ .path = "tests/model_tests.zig" },
        .target = target,
        .optimize = optimize,
    });
    model_tests.addModule("zig-onnx-parser", onnx_parser);

    const run_model_tests = b.addRunArtifact(model_tests);
    const model_test_step = b.step("test-models", "Test with real ONNX models");
    model_test_step.dependOn(&run_model_tests.step);

    // Protobuf tests
    const protobuf_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/formats/onnx/protobuf.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_protobuf_tests = b.addRunArtifact(protobuf_tests);
    const protobuf_test_step = b.step("test-protobuf", "Run protobuf parser tests");
    protobuf_test_step.dependOn(&run_protobuf_tests.step);

    // Benchmark executable
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = .{ .path = "examples/benchmark.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    benchmark.addModule("zig-onnx-parser", onnx_parser);

    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run parsing benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);

    // Validation tool
    const validator = b.addExecutable(.{
        .name = "onnx-validator",
        .root_source_file = .{ .path = "tools/validator.zig" },
        .target = target,
        .optimize = optimize,
    });
    validator.addModule("zig-onnx-parser", onnx_parser);

    const validator_step = b.step("validator", "Build ONNX validation tool");
    validator_step.dependOn(&validator.step);
    b.installArtifact(validator);

    // Model inspector tool
    const inspector = b.addExecutable(.{
        .name = "onnx-inspector",
        .root_source_file = .{ .path = "tools/inspector.zig" },
        .target = target,
        .optimize = optimize,
    });
    inspector.addModule("zig-onnx-parser", onnx_parser);

    const inspector_step = b.step("inspector", "Build ONNX model inspector tool");
    inspector_step.dependOn(&inspector.step);
    b.installArtifact(inspector);

    // Documentation generation
    const docs = b.addTest(.{
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = .Debug,
    });

    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&docs.step);

    // Integration with common interfaces
    const common_interfaces = b.createModule(.{
        .source_file = .{ .path = "../../common/interfaces/model.zig" },
    });

    // Add common interfaces to the library
    onnx_parser.dependencies.put("common-interfaces", common_interfaces) catch unreachable;

    // Cross-compilation for different targets
    const cross_targets = [_]std.zig.CrossTarget{
        .{ .cpu_arch = .aarch64, .os_tag = .linux },
        .{ .cpu_arch = .arm, .os_tag = .linux },
        .{ .cpu_arch = .x86_64, .os_tag = .windows },
    };

    for (cross_targets, 0..) |cross_target, i| {
        const cross_lib = b.addStaticLibrary(.{
            .name = b.fmt("zig-onnx-parser-{s}-{s}", .{ @tagName(cross_target.cpu_arch.?), @tagName(cross_target.os_tag.?) }),
            .root_source_file = .{ .path = "src/lib.zig" },
            .target = cross_target,
            .optimize = optimize,
        });

        const cross_step = b.step(b.fmt("cross-{d}", .{i}), b.fmt("Cross-compile for {s}-{s}", .{ @tagName(cross_target.cpu_arch.?), @tagName(cross_target.os_tag.?) }));
        cross_step.dependOn(&cross_lib.step);

        b.installArtifact(cross_lib);
    }

    // Fuzzing tests (if enabled)
    if (b.option(bool, "fuzz", "Enable fuzzing tests") orelse false) {
        const fuzz_tests = b.addExecutable(.{
            .name = "fuzz-tests",
            .root_source_file = .{ .path = "tests/fuzz.zig" },
            .target = target,
            .optimize = optimize,
        });
        fuzz_tests.addModule("zig-onnx-parser", onnx_parser);

        const run_fuzz_tests = b.addRunArtifact(fuzz_tests);
        const fuzz_step = b.step("fuzz", "Run fuzzing tests");
        fuzz_step.dependOn(&run_fuzz_tests.step);
    }

    // Memory leak detection
    if (optimize == .Debug) {
        const leak_tests = b.addTest(.{
            .root_source_file = .{ .path = "tests/memory_leaks.zig" },
            .target = target,
            .optimize = optimize,
        });
        leak_tests.addModule("zig-onnx-parser", onnx_parser);

        const run_leak_tests = b.addRunArtifact(leak_tests);
        const leak_step = b.step("test-leaks", "Run memory leak tests");
        leak_step.dependOn(&run_leak_tests.step);
    }

    // Performance profiling
    const profile_tests = b.addExecutable(.{
        .name = "profile-tests",
        .root_source_file = .{ .path = "tests/profiling.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    profile_tests.addModule("zig-onnx-parser", onnx_parser);

    const run_profile_tests = b.addRunArtifact(profile_tests);
    const profile_step = b.step("profile", "Run performance profiling");
    profile_step.dependOn(&run_profile_tests.step);

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
    all_tests_step.dependOn(protobuf_test_step);
    all_tests_step.dependOn(model_test_step);
    if (optimize == .Debug) {
        const leak_tests = b.addTest(.{
            .root_source_file = .{ .path = "tests/memory_leaks.zig" },
            .target = target,
            .optimize = optimize,
        });
        leak_tests.addModule("zig-onnx-parser", onnx_parser);
        all_tests_step.dependOn(&b.addRunArtifact(leak_tests).step);
    }

    // Default step
    b.default_step.dependOn(&lib.step);
    b.default_step.dependOn(test_step);
}

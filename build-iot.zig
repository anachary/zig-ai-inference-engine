const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create IoT-specific library
    const iot_lib = b.addSharedLibrary(.{
        .name = "zig-ai-iot",
        .root_source_file = b.path("src/iot/c_api.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add dependencies
    const tensor_core = b.addModule("zig-tensor-core", .{
        .root_source_file = b.path("../zig-tensor-core/src/lib.zig"),
    });

    const onnx_parser = b.addModule("zig-onnx-parser", .{
        .root_source_file = b.path("../zig-onnx-parser/src/lib.zig"),
    });

    const inference_engine = b.addModule("zig-inference-engine", .{
        .root_source_file = b.path("../zig-inference-engine/src/lib.zig"),
    });

    // Add dependencies to IoT library
    iot_lib.root_module.addImport("zig-tensor-core", tensor_core);
    iot_lib.root_module.addImport("zig-onnx-parser", onnx_parser);
    iot_lib.root_module.addImport("zig-inference-engine", inference_engine);

    // IoT-specific optimizations
    iot_lib.root_module.strip = optimize != .Debug;
    iot_lib.root_module.single_threaded = false; // Allow multi-threading for IoT
    
    // Link system libraries for IoT devices
    if (target.result.os.tag == .linux) {
        iot_lib.linkSystemLibrary("c");
        iot_lib.linkSystemLibrary("m");
        iot_lib.linkSystemLibrary("pthread");
    }

    // Install the IoT library
    b.installArtifact(iot_lib);

    // Create IoT-specific executable for testing
    const iot_test = b.addExecutable(.{
        .name = "zig-ai-iot-test",
        .root_source_file = b.path("src/iot/test_runner.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add dependencies to test executable
    iot_test.root_module.addImport("zig-tensor-core", tensor_core);
    iot_test.root_module.addImport("zig-onnx-parser", onnx_parser);
    iot_test.root_module.addImport("zig-inference-engine", inference_engine);

    // Link libraries for test
    if (target.result.os.tag == .linux) {
        iot_test.linkSystemLibrary("c");
        iot_test.linkSystemLibrary("m");
        iot_test.linkSystemLibrary("pthread");
    }

    b.installArtifact(iot_test);

    // Create run step for IoT test
    const run_iot_test = b.addRunArtifact(iot_test);
    run_iot_test.step.dependOn(b.getInstallStep());

    const run_iot_step = b.step("run-iot-test", "Run IoT inference test");
    run_iot_step.dependOn(&run_iot_test.step);

    // Create Python wheel build step
    const python_wheel_step = b.step("python-wheel", "Build Python wheel for IoT bindings");
    
    const wheel_cmd = b.addSystemCommand(&[_][]const u8{
        "python3", "setup.py", "bdist_wheel"
    });
    wheel_cmd.cwd = b.path("src/iot");
    wheel_cmd.step.dependOn(b.getInstallStep());
    
    python_wheel_step.dependOn(&wheel_cmd.step);

    // Create Docker build step for IoT
    const docker_iot_step = b.step("docker-iot", "Build IoT Docker image");
    
    const docker_cmd = b.addSystemCommand(&[_][]const u8{
        "docker", "build", "-f", "docker/Dockerfile.iot-arm64", "-t", "zig-ai-iot:latest", "."
    });
    docker_cmd.step.dependOn(b.getInstallStep());
    
    docker_iot_step.dependOn(&docker_cmd.step);

    // Unit tests for IoT components
    const iot_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/iot/c_api.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add dependencies to unit tests
    iot_unit_tests.root_module.addImport("zig-tensor-core", tensor_core);
    iot_unit_tests.root_module.addImport("zig-onnx-parser", onnx_parser);
    iot_unit_tests.root_module.addImport("zig-inference-engine", inference_engine);

    const run_iot_unit_tests = b.addRunArtifact(iot_unit_tests);
    const iot_test_step = b.step("test-iot", "Run IoT unit tests");
    iot_test_step.dependOn(&run_iot_unit_tests.step);

    // Benchmark step for IoT performance
    const iot_benchmark = b.addExecutable(.{
        .name = "zig-ai-iot-benchmark",
        .root_source_file = b.path("src/iot/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast, // Always optimize benchmarks
    });

    // Add dependencies to benchmark
    iot_benchmark.root_module.addImport("zig-tensor-core", tensor_core);
    iot_benchmark.root_module.addImport("zig-onnx-parser", onnx_parser);
    iot_benchmark.root_module.addImport("zig-inference-engine", inference_engine);

    if (target.result.os.tag == .linux) {
        iot_benchmark.linkSystemLibrary("c");
        iot_benchmark.linkSystemLibrary("m");
        iot_benchmark.linkSystemLibrary("pthread");
    }

    b.installArtifact(iot_benchmark);

    const run_iot_benchmark = b.addRunArtifact(iot_benchmark);
    const benchmark_step = b.step("benchmark-iot", "Run IoT performance benchmarks");
    benchmark_step.dependOn(&run_iot_benchmark.step);

    // Cross-compilation targets for common IoT devices
    const arm_targets = [_]std.Target.Query{
        .{ .cpu_arch = .aarch64, .os_tag = .linux },  // ARM64 (Raspberry Pi 4, Jetson)
        .{ .cpu_arch = .arm, .os_tag = .linux },      // ARM32 (Raspberry Pi 3)
    };

    for (arm_targets) |arm_target| {
        const arm_lib = b.addSharedLibrary(.{
            .name = "zig-ai-iot",
            .root_source_file = b.path("src/iot/c_api.zig"),
            .target = b.resolveTargetQuery(arm_target),
            .optimize = .ReleaseFast,
        });

        // Add dependencies
        arm_lib.root_module.addImport("zig-tensor-core", tensor_core);
        arm_lib.root_module.addImport("zig-onnx-parser", onnx_parser);
        arm_lib.root_module.addImport("zig-inference-engine", inference_engine);

        // ARM-specific optimizations
        arm_lib.root_module.strip = true;
        arm_lib.linkSystemLibrary("c");
        arm_lib.linkSystemLibrary("m");
        arm_lib.linkSystemLibrary("pthread");

        const target_name = switch (arm_target.cpu_arch.?) {
            .aarch64 => "arm64",
            .arm => "arm32",
            else => "unknown",
        };

        const install_arm = b.addInstallArtifact(arm_lib, .{
            .dest_dir = .{ .override = .{ .custom = b.fmt("lib/{s}", .{target_name}) } },
        });

        const arm_step = b.step(
            b.fmt("build-{s}", .{target_name}),
            b.fmt("Build IoT library for {s}", .{target_name})
        );
        arm_step.dependOn(&install_arm.step);
    }

    // Documentation generation
    const docs_step = b.step("docs-iot", "Generate IoT documentation");
    
    const docs_cmd = b.addSystemCommand(&[_][]const u8{
        "zig", "build-exe", "--show-builtin", "src/iot/c_api.zig"
    });
    docs_cmd.step.dependOn(b.getInstallStep());
    
    docs_step.dependOn(&docs_cmd.step);

    // Clean step for IoT artifacts
    const clean_step = b.step("clean-iot", "Clean IoT build artifacts");
    
    const clean_cmd = b.addSystemCommand(&[_][]const u8{
        "rm", "-rf", "zig-out/lib/zig-ai-iot*", "src/iot/__pycache__", "src/iot/build", "src/iot/dist"
    });
    
    clean_step.dependOn(&clean_cmd.step);
}

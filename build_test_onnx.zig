const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the test executable
    const test_exe = b.addExecutable(.{
        .name = "test_onnx_loading",
        .root_source_file = b.path("test_onnx_loading.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add the ONNX parser module
    const onnx_parser_module = b.addModule("onnx_parser", .{
        .root_source_file = b.path("projects/zig-onnx-parser/src/parser.zig"),
    });
    
    test_exe.root_module.addImport("onnx_parser", onnx_parser_module);

    b.installArtifact(test_exe);

    // Create run step
    const run_cmd = b.addRunArtifact(test_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("test-onnx", "Test ONNX loading with real model");
    run_step.dependOn(&run_cmd.step);
}

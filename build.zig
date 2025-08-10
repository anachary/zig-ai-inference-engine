const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const optimize = b.standardOptimizeOption(.{});

    // Create the main library as a dynamic library (DLL)
    const lib = b.addSharedLibrary(.{
        .name = "zig-ai-platform",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
    });

    // Install the library
    b.installArtifact(lib);

    // Also create a static library for easier linking in some cases
    const static_lib = b.addStaticLibrary(.{
        .name = "zig-ai-platform-static",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Install the static library
    b.installArtifact(static_lib);

    // Create the CLI executable that uses the library
    const cli_exe = b.addExecutable(.{
        .name = "zig-ai-cli",
        .root_source_file = .{ .path = "examples/cli.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Link the CLI with the dynamic library
    cli_exe.linkLibrary(lib);

    // Install the CLI executable
    b.installArtifact(cli_exe);

    // Create a run step for the CLI
    const run_cmd = b.addRunArtifact(cli_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    // Allow passing arguments to the CLI
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the CLI");
    run_step.dependOn(&run_cmd.step);

    // Create a chat-specific run step
    const chat_cmd = b.addRunArtifact(cli_exe);
    chat_cmd.step.dependOn(b.getInstallStep());
    chat_cmd.addArg("chat");
    if (b.args) |args| {
        chat_cmd.addArgs(args);
    }

    const chat_step = b.step("chat", "Run interactive chat mode");
    chat_step.dependOn(&chat_cmd.step);

    // Create unit tests
    const lib_unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    // Create examples
    const examples = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "demo", .path = "examples/demo.zig" },
        .{ .name = "load-gguf", .path = "examples/load_gguf.zig" },
        .{ .name = "tokenize", .path = "examples/tokenize.zig" },
        .{ .name = "inference", .path = "examples/inference.zig" },
        .{ .name = "v2-inference", .path = "examples/v2_inference.zig" },
        .{ .name = "v2-tokenizer-test", .path = "tests_v2/tokenizer_test.zig" },
        .{ .name = "v2-tokenizer-unicode-test", .path = "tests_v2/tokenizer_unicode_test.zig" },
        .{ .name = "v2-smoke-forward-test", .path = "tests_v2/smoke_forward_test.zig" },
        .{ .name = "v2-tokenizer-stress-test", .path = "tests_v2/tokenizer_stress_test.zig" },
        .{ .name = "v2-chat", .path = "examples/v2_chat_cli.zig" },
    };

    for (examples) |example| {
        const exe = b.addExecutable(.{
            .name = example.name,
            .root_source_file = .{ .path = example.path },
            .target = target,
            .optimize = optimize,
        });

        // Add the primary library module
        exe.addModule("zig-ai-platform", b.createModule(.{
            .source_file = .{ .path = "src/main.zig" },
        }));
        // Add src_v2 API as a module for v2 examples
        exe.addModule("src_v2", b.createModule(.{
            .source_file = .{ .path = "src_v2/root.zig" },
        }));

        const install_exe = b.addInstallArtifact(exe, .{});
        const example_step = b.step(example.name, b.fmt("Build and install {s} example", .{example.name}));
        example_step.dependOn(&install_exe.step);
    }
}

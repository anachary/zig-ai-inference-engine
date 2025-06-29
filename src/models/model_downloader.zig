const std = @import("std");
const Allocator = std.mem.Allocator;

pub const DownloadError = error{
    NetworkError,
    FileError,
    InvalidUrl,
    OutOfMemory,
    DownloadFailed,
    GitOperationFailed,
    SubmoduleNotFound,
};

pub const GitOperationError = error{
    GitNotFound,
    SubmoduleInitFailed,
    SubmoduleUpdateFailed,
    InvalidRepository,
    OutOfMemory,
};

pub const DownloadProgress = struct {
    bytes_downloaded: u64,
    total_bytes: u64,
    percentage: f32,

    pub fn update(self: *DownloadProgress, downloaded: u64, total: u64) void {
        self.bytes_downloaded = downloaded;
        self.total_bytes = total;
        self.percentage = if (total > 0) @as(f32, @floatFromInt(downloaded)) / @as(f32, @floatFromInt(total)) * 100.0 else 0.0;
    }
};

pub const GitOperations = struct {
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
        };
    }

    pub fn isGitRepository(self: *Self) bool {
        _ = self;
        // Check if .git directory exists
        std.fs.cwd().access(".git", .{}) catch return false;
        return true;
    }

    pub fn hasSubmodule(self: *Self, submodule_path: []const u8) bool {
        // Check if submodule directory exists and has .git file
        const git_file_path = std.fmt.allocPrint(self.allocator, "{s}/.git", .{submodule_path}) catch return false;
        defer self.allocator.free(git_file_path);

        std.fs.cwd().access(git_file_path, .{}) catch return false;
        return true;
    }

    pub fn initSubmodule(self: *Self, submodule_path: []const u8) !void {
        std.log.info("🔧 Initializing Git submodule: {s}", .{submodule_path});

        var process = std.ChildProcess.init(&[_][]const u8{ "git", "submodule", "init", submodule_path }, self.allocator);
        process.stdout_behavior = .Pipe;
        process.stderr_behavior = .Pipe;

        try process.spawn();
        const result = try process.wait();

        switch (result) {
            .Exited => |code| {
                if (code == 0) {
                    std.log.info("✅ Submodule initialized successfully", .{});
                } else {
                    std.log.err("❌ Submodule init failed with exit code: {d}", .{code});
                    return GitOperationError.SubmoduleInitFailed;
                }
            },
            else => {
                std.log.err("❌ Submodule init process failed", .{});
                return GitOperationError.SubmoduleInitFailed;
            },
        }
    }

    pub fn updateSubmodule(self: *Self, submodule_path: []const u8) !void {
        std.log.info("📥 Updating Git submodule: {s}", .{submodule_path});

        var process = std.ChildProcess.init(&[_][]const u8{ "git", "submodule", "update", "--remote", submodule_path }, self.allocator);
        process.stdout_behavior = .Pipe;
        process.stderr_behavior = .Pipe;

        try process.spawn();
        const result = try process.wait();

        switch (result) {
            .Exited => |code| {
                if (code == 0) {
                    std.log.info("✅ Submodule updated successfully", .{});
                } else {
                    std.log.err("❌ Submodule update failed with exit code: {d}", .{code});
                    return GitOperationError.SubmoduleUpdateFailed;
                }
            },
            else => {
                std.log.err("❌ Submodule update process failed", .{});
                return GitOperationError.SubmoduleUpdateFailed;
            },
        }
    }

    pub fn initAndUpdateSubmodule(self: *Self, submodule_path: []const u8) !void {
        std.log.info("🚀 Initializing and updating submodule: {s}", .{submodule_path});

        var process = std.ChildProcess.init(&[_][]const u8{ "git", "submodule", "update", "--init", "--remote", submodule_path }, self.allocator);
        process.stdout_behavior = .Pipe;
        process.stderr_behavior = .Pipe;

        try process.spawn();
        const result = try process.wait();

        switch (result) {
            .Exited => |code| {
                if (code == 0) {
                    std.log.info("✅ Submodule initialized and updated successfully", .{});
                } else {
                    std.log.err("❌ Submodule init/update failed with exit code: {d}", .{code});
                    return GitOperationError.SubmoduleUpdateFailed;
                }
            },
            else => {
                std.log.err("❌ Submodule init/update process failed", .{});
                return GitOperationError.SubmoduleUpdateFailed;
            },
        }
    }
};

pub const ModelDownloader = struct {
    allocator: Allocator,
    git_ops: GitOperations,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .git_ops = GitOperations.init(allocator),
        };
    }

    pub fn downloadModel(self: *Self, url: []const u8, output_path: []const u8) !void {
        std.log.info("🌐 Starting download from: {s}", .{url});
        std.log.info("📁 Saving to: {s}", .{output_path});

        // For now, we'll use curl or wget as external tools
        // In a full implementation, you'd use HTTP client libraries

        if (self.isWindowsSystem()) {
            try self.downloadWithPowershell(url, output_path);
        } else {
            try self.downloadWithCurl(url, output_path);
        }
    }

    pub fn downloadModelWithSubmodule(self: *Self, model_name: []const u8, submodule_path: []const u8, fallback_url: ?[]const u8, output_path: []const u8) ![]u8 {
        // Try submodule approach first
        if (self.trySubmoduleDownload(model_name, submodule_path)) |submodule_model_path| {
            std.log.info("✅ Using model from submodule: {s}", .{submodule_model_path});
            return submodule_model_path;
        } else |err| {
            std.log.info("ℹ️ Submodule not available ({any}), trying HTTP download...", .{err});

            // Fallback to HTTP download if submodule fails
            if (fallback_url) |url| {
                try self.downloadModel(url, output_path);
                return self.allocator.dupe(u8, output_path);
            } else {
                std.log.err("❌ No fallback URL provided for HTTP download", .{});
                return DownloadError.SubmoduleNotFound;
            }
        }
    }

    fn trySubmoduleDownload(self: *Self, model_name: []const u8, submodule_path: []const u8) ![]u8 {
        // Check if we're in a Git repository
        if (!self.git_ops.isGitRepository()) {
            std.log.info("ℹ️ Not in a Git repository, skipping submodule approach", .{});
            return DownloadError.SubmoduleNotFound;
        }

        // Check if submodule exists and is initialized
        if (!self.git_ops.hasSubmodule(submodule_path)) {
            std.log.info("🔧 Submodule not found, attempting to initialize: {s}", .{submodule_path});
            self.git_ops.initAndUpdateSubmodule(submodule_path) catch |err| {
                std.log.err("❌ Failed to initialize submodule: {any}", .{err});
                return DownloadError.GitOperationFailed;
            };
        } else {
            std.log.info("🔄 Updating existing submodule: {s}", .{submodule_path});
            self.git_ops.updateSubmodule(submodule_path) catch |err| {
                std.log.warn("⚠️ Failed to update submodule (continuing with existing): {any}", .{err});
                // Continue with existing submodule content
            };
        }

        // Look for the model file in the submodule
        const model_file_path = std.fmt.allocPrint(self.allocator, "{s}/{s}.onnx", .{ submodule_path, model_name }) catch return DownloadError.OutOfMemory;

        // Check if model file exists
        std.fs.cwd().access(model_file_path, .{}) catch |err| {
            self.allocator.free(model_file_path);
            std.log.err("❌ Model file not found in submodule: {s} (error: {any})", .{ model_file_path, err });
            return DownloadError.SubmoduleNotFound;
        };

        std.log.info("✅ Found model in submodule: {s}", .{model_file_path});
        return model_file_path;
    }

    fn isWindowsSystem(self: *Self) bool {
        _ = self;
        return @import("builtin").os.tag == .windows;
    }

    fn downloadWithPowershell(self: *Self, url: []const u8, output_path: []const u8) !void {
        std.log.info("📥 Using PowerShell to download...", .{});

        // Create PowerShell command
        var command_buffer: [2048]u8 = undefined;
        const command = try std.fmt.bufPrint(command_buffer[0..], "powershell -Command \"" ++
            "$ProgressPreference = 'SilentlyContinue'; " ++
            "Invoke-WebRequest -Uri '{s}' -OutFile '{s}' -UseBasicParsing\"", .{ url, output_path });

        std.log.info("🔧 Command: {s}", .{command});

        // Create PowerShell command string
        var ps_command_buffer: [2048]u8 = undefined;
        const ps_command = try std.fmt.bufPrint(ps_command_buffer[0..], "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '{s}' -OutFile '{s}' -UseBasicParsing", .{ url, output_path });

        // Execute the command directly with PowerShell
        var process = std.ChildProcess.init(&[_][]const u8{ "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", "-Command", ps_command }, self.allocator);
        process.stdout_behavior = .Pipe;
        process.stderr_behavior = .Pipe;

        try process.spawn();

        const result = try process.wait();

        // Read stdout and stderr for debugging
        if (process.stdout) |stdout| {
            const stdout_data = try stdout.readToEndAlloc(self.allocator, 8192);
            defer self.allocator.free(stdout_data);
            if (stdout_data.len > 0) {
                std.log.info("PowerShell stdout: {s}", .{stdout_data});
            }
        }

        if (process.stderr) |stderr| {
            const stderr_data = try stderr.readToEndAlloc(self.allocator, 8192);
            defer self.allocator.free(stderr_data);
            if (stderr_data.len > 0) {
                std.log.err("PowerShell stderr: {s}", .{stderr_data});
            }
        }

        switch (result) {
            .Exited => |code| {
                if (code == 0) {
                    std.log.info("✅ Download completed successfully", .{});
                } else {
                    std.log.err("❌ Download failed with exit code: {d}", .{code});
                    return DownloadError.DownloadFailed;
                }
            },
            else => {
                std.log.err("❌ Download process failed", .{});
                return DownloadError.DownloadFailed;
            },
        }
    }

    fn downloadWithCurl(self: *Self, url: []const u8, output_path: []const u8) !void {
        std.log.info("📥 Using curl to download...", .{});

        // Create curl command
        var command_args = [_][]const u8{
            "curl",
            "-L", // Follow redirects
            "-o", output_path, // Output file
            "--progress-bar", // Show progress
            url,
        };

        // Execute curl
        var process = std.ChildProcess.init(&command_args, self.allocator);
        process.stdout_behavior = .Pipe;
        process.stderr_behavior = .Pipe;

        try process.spawn();

        const result = try process.wait();

        switch (result) {
            .Exited => |code| {
                if (code == 0) {
                    std.log.info("✅ Download completed successfully", .{});
                } else {
                    std.log.err("❌ Download failed with exit code: {d}", .{code});
                    return DownloadError.DownloadFailed;
                }
            },
            else => {
                std.log.err("❌ Download process failed", .{});
                return DownloadError.DownloadFailed;
            },
        }
    }

    pub fn downloadOfficialModel(self: *Self, model_name: []const u8, output_dir: []const u8) ![]u8 {
        const model_info = self.getOfficialModelInfo(model_name) orelse {
            std.log.err("❌ Unknown official model: {s}", .{model_name});
            return DownloadError.InvalidUrl;
        };

        // Create output directory
        std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        // Create full output path
        var path_buffer: [512]u8 = undefined;
        const output_path = try std.fmt.bufPrint(path_buffer[0..], "{s}/{s}", .{ output_dir, model_info.filename });

        // Check if already exists
        if (std.fs.cwd().access(output_path, .{})) {
            std.log.info("✅ Model already exists: {s}", .{output_path});
            return self.allocator.dupe(u8, output_path);
        } else |_| {
            std.log.info("📥 Downloading official model: {s}", .{model_info.name});
            std.log.info("📊 Size: {s} ({d}MB)", .{ model_info.parameters, model_info.size_mb });

            try self.downloadModel(model_info.download_url, output_path);

            // Verify the download
            const file = std.fs.cwd().openFile(output_path, .{}) catch |err| {
                std.log.err("❌ Failed to verify downloaded file", .{});
                return err;
            };
            defer file.close();

            const file_size = try file.getEndPos();
            std.log.info("✅ Downloaded: {s} ({d} bytes)", .{ output_path, file_size });

            return self.allocator.dupe(u8, output_path);
        }
    }

    const OfficialModelInfo = struct {
        name: []const u8,
        parameters: []const u8,
        size_mb: u32,
        download_url: []const u8,
        filename: []const u8,
        description: []const u8,
    };

    fn getOfficialModelInfo(self: *Self, model_name: []const u8) ?OfficialModelInfo {
        _ = self;

        // Official model registry with real download URLs
        if (std.mem.eql(u8, model_name, "distilgpt2")) {
            return OfficialModelInfo{
                .name = "DistilGPT-2",
                .parameters = "82M",
                .size_mb = 330,
                .download_url = "https://huggingface.co/distilgpt2/resolve/main/pytorch_model.bin",
                .filename = "distilgpt2.onnx",
                .description = "Lightweight GPT-2 variant",
            };
        } else if (std.mem.eql(u8, model_name, "gpt2-small")) {
            return OfficialModelInfo{
                .name = "GPT-2 Small",
                .parameters = "124M",
                .size_mb = 500,
                .download_url = "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin",
                .filename = "gpt2-small.onnx",
                .description = "Original GPT-2 small model",
            };
        } else if (std.mem.eql(u8, model_name, "tinyllama")) {
            return OfficialModelInfo{
                .name = "TinyLlama-1.1B",
                .parameters = "1.1B",
                .size_mb = 2200,
                .download_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/pytorch_model.bin",
                .filename = "tinyllama-1.1b.onnx",
                .description = "Compact LLaMA model for edge deployment",
            };
        } else if (std.mem.eql(u8, model_name, "phi2")) {
            return OfficialModelInfo{
                .name = "Phi-2",
                .parameters = "2.7B",
                .size_mb = 5400,
                .download_url = "https://huggingface.co/microsoft/phi-2/resolve/main/pytorch_model.bin",
                .filename = "phi2.onnx",
                .description = "Microsoft's efficient reasoning model",
            };
        }

        return null;
    }

    pub fn listOfficialModels(self: *Self) void {
        std.log.info("🤖 Official Models Available for Download:", .{});
        std.log.info("==========================================", .{});

        const models = [_][]const u8{ "distilgpt2", "gpt2-small", "tinyllama", "phi2" };

        for (models) |model_name| {
            if (self.getOfficialModelInfo(model_name)) |info| {
                std.log.info("📦 {s}", .{info.name});
                std.log.info("   Parameters: {s}", .{info.parameters});
                std.log.info("   Size: {d}MB", .{info.size_mb});
                std.log.info("   Description: {s}", .{info.description});
                std.log.info("   Command: zig build cli -- download-official --model {s}", .{model_name});
                std.log.info("", .{});
            }
        }
    }

    pub fn getModelMemoryRequirement(self: *Self, model_name: []const u8) u32 {
        if (self.getOfficialModelInfo(model_name)) |info| {
            // Estimate memory requirement (typically 1.5x model size)
            return @as(u32, @intFromFloat(@as(f32, @floatFromInt(info.size_mb)) * 1.5));
        }
        return 1000; // Default 1GB
    }
};

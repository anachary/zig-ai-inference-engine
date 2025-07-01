const std = @import("std");
const print = std.debug.print;

pub const ModelDownloader = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
        };
    }

    /// Download model and tokenizer from Hugging Face
    pub fn downloadFromHuggingFace(self: *Self, repo_id: []const u8, model_dir: []const u8) !void {
        print("🔽 Downloading model from Hugging Face: {s}\n", .{repo_id});

        // Create models directory if it doesn't exist
        std.fs.cwd().makeDir(model_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        // Download ONNX model file
        try self.downloadFile(repo_id, "model.onnx", model_dir, "model_fp16.onnx");
        
        // Download tokenizer files
        try self.downloadFile(repo_id, "tokenizer.json", model_dir, "tokenizer.json");
        try self.downloadFile(repo_id, "vocab.json", model_dir, "vocab.json");
        
        // Try to download additional tokenizer files (optional)
        self.downloadFile(repo_id, "merges.txt", model_dir, "merges.txt") catch |err| {
            print("ℹ️  merges.txt not found (optional): {}\n", .{err});
        };
        
        self.downloadFile(repo_id, "config.json", model_dir, "config.json") catch |err| {
            print("ℹ️  config.json not found (optional): {}\n", .{err});
        };

        print("✅ Model download completed!\n", .{});
    }

    /// Download a specific file from Hugging Face repository
    fn downloadFile(self: *Self, repo_id: []const u8, filename: []const u8, local_dir: []const u8, local_filename: []const u8) !void {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "https://huggingface.co/{s}/resolve/main/{s}",
            .{ repo_id, filename }
        );
        defer self.allocator.free(url);

        const local_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/{s}",
            .{ local_dir, local_filename }
        );
        defer self.allocator.free(local_path);

        print("📥 Downloading {s} -> {s}\n", .{ filename, local_path });

        // Use PowerShell on Windows to download the file
        const command = try std.fmt.allocPrint(
            self.allocator,
            "powershell -Command \"Invoke-WebRequest -Uri '{s}' -OutFile '{s}'\"",
            .{ url, local_path }
        );
        defer self.allocator.free(command);

        // Execute the download command
        var child = std.ChildProcess.init(&[_][]const u8{ "cmd", "/c", command }, self.allocator);
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

        try child.spawn();
        const result = try child.wait();

        switch (result) {
            .Exited => |code| {
                if (code == 0) {
                    print("✅ Downloaded {s}\n", .{filename});
                } else {
                    print("❌ Failed to download {s} (exit code: {d})\n", .{ filename, code });
                    return error.DownloadFailed;
                }
            },
            else => {
                print("❌ Download process failed for {s}\n", .{filename});
                return error.DownloadFailed;
            },
        }
    }

    /// Download popular pre-trained models
    pub fn downloadPopularModel(self: *Self, model_name: []const u8, model_dir: []const u8) !void {
        const repo_id = switch (std.hash_map.hashString(model_name)) {
            std.hash_map.hashString("phi2") => "microsoft/phi-2",
            std.hash_map.hashString("tinyllama") => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            std.hash_map.hashString("gpt2") => "gpt2",
            std.hash_map.hashString("distilbert") => "distilbert-base-uncased",
            std.hash_map.hashString("bert") => "bert-base-uncased",
            else => {
                print("❌ Unknown model: {s}\n", .{model_name});
                print("Available models: phi2, tinyllama, gpt2, distilbert, bert\n", .{});
                return error.UnknownModel;
            },
        };

        try self.downloadFromHuggingFace(repo_id, model_dir);
    }

    /// List available models for download
    pub fn listAvailableModels(self: *Self) void {
        _ = self;
        print("📋 Available models for download:\n", .{});
        print("  • phi2        - Microsoft Phi-2 (2.7B parameters)\n", .{});
        print("  • tinyllama   - TinyLlama 1.1B Chat\n", .{});
        print("  • gpt2        - OpenAI GPT-2\n", .{});
        print("  • distilbert  - DistilBERT Base Uncased\n", .{});
        print("  • bert        - BERT Base Uncased\n", .{});
        print("\nUsage:\n", .{});
        print("  zig build cli -- download --model <model_name>\n", .{});
        print("  zig build cli -- download --repo <huggingface_repo_id>\n", .{});
    }
};

/// CLI interface for model downloading
pub fn downloadCommand(allocator: std.mem.Allocator, args: [][]const u8) !void {
    var downloader = ModelDownloader.init(allocator);

    if (args.len < 2) {
        downloader.listAvailableModels();
        return;
    }

    const command = args[0];
    const value = args[1];

    if (std.mem.eql(u8, command, "--model")) {
        try downloader.downloadPopularModel(value, "models");
    } else if (std.mem.eql(u8, command, "--repo")) {
        try downloader.downloadFromHuggingFace(value, "models");
    } else if (std.mem.eql(u8, command, "--list")) {
        downloader.listAvailableModels();
    } else {
        print("❌ Unknown download command: {s}\n", .{command});
        print("Usage:\n", .{});
        print("  --model <model_name>     Download a popular model\n", .{});
        print("  --repo <repo_id>         Download from specific Hugging Face repo\n", .{});
        print("  --list                   List available models\n", .{});
    }
}

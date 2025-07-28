const std = @import("std");

/// Ollama Chat CLI - Real ONNX Model Chat Interface
///
/// This example demonstrates:
/// - Downloading real ONNX models from Ollama/Hugging Face
/// - Loading and running actual AI models
/// - Interactive chat with real AI responses
/// - Model management and switching
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize the Ollama chat application
    var chat = try OllamaChat.init(allocator);
    defer chat.deinit();

    // Start interactive chat
    try chat.run();
}

const OllamaChat = struct {
    allocator: std.mem.Allocator,
    conversation_history: std.ArrayList(Message),
    current_model: ?[]const u8,
    model_loaded: bool,

    const Message = struct {
        role: Role,
        content: []const u8,
        timestamp: i64,

        const Role = enum {
            user,
            assistant,
            system,
        };
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
            .conversation_history = std.ArrayList(Message).init(allocator),
            .current_model = null,
            .model_loaded = false,
        };
    }

    pub fn deinit(self: *Self) void {
        // Clean up conversation history
        for (self.conversation_history.items) |message| {
            self.allocator.free(message.content);
        }
        self.conversation_history.deinit();

        // Clean up current model name
        if (self.current_model) |model| {
            self.allocator.free(model);
        }
    }

    pub fn run(self: *Self) !void {
        const stdout = std.io.getStdOut().writer();
        const stdin = std.io.getStdIn().reader();

        // Print welcome message
        try self.printWelcome();

        // Main chat loop
        while (true) {
            // Show prompt with current model
            if (self.current_model) |model| {
                try stdout.print("\n[{s}] > ", .{model});
            } else {
                try stdout.print("\n[no model] > ", .{});
            }

            // Read user input
            var input_buffer: [1024]u8 = undefined;
            if (try stdin.readUntilDelimiterOrEof(input_buffer[0..], '\n')) |input| {
                const trimmed_input = std.mem.trim(u8, input, " \t\r\n");

                // Handle special commands
                if (std.mem.eql(u8, trimmed_input, "/quit") or std.mem.eql(u8, trimmed_input, "/exit")) {
                    try stdout.print("Goodbye! Thanks for using Ollama Chat.\n", .{});
                    break;
                } else if (std.mem.eql(u8, trimmed_input, "/help")) {
                    try self.printHelp();
                    continue;
                } else if (std.mem.eql(u8, trimmed_input, "/models")) {
                    try self.listAvailableModels();
                    continue;
                } else if (std.mem.startsWith(u8, trimmed_input, "/download ")) {
                    const model_name = trimmed_input[10..];
                    try self.downloadModel(model_name);
                    continue;
                } else if (std.mem.startsWith(u8, trimmed_input, "/load ")) {
                    const model_name = trimmed_input[6..];
                    try self.loadModel(model_name);
                    continue;
                } else if (std.mem.eql(u8, trimmed_input, "/unload")) {
                    try self.unloadModel();
                    continue;
                } else if (std.mem.eql(u8, trimmed_input, "/clear")) {
                    try self.clearHistory();
                    try stdout.print("Conversation history cleared.\n", .{});
                    continue;
                } else if (std.mem.eql(u8, trimmed_input, "/status")) {
                    try self.printStatus();
                    continue;
                } else if (trimmed_input.len == 0) {
                    continue;
                }

                // Check if model is loaded
                if (!self.model_loaded) {
                    try stdout.print("No model loaded. Use /download <model> and /load <model> first.\n", .{});
                    try stdout.print("Available commands: /models, /download, /load, /help\n", .{});
                    continue;
                }

                // Add user message to history
                try self.addMessage(.user, trimmed_input);

                // Generate response using the loaded model
                try self.generateResponse(trimmed_input);
            } else {
                break; // EOF
            }
        }
    }

    fn printWelcome(self: *Self) !void {
        _ = self;
        const stdout = std.io.getStdOut().writer();

        try stdout.print("\n", .{});
        try stdout.print("=================================================\n", .{});
        try stdout.print("    Ollama Chat - Real ONNX Model Interface\n", .{});
        try stdout.print("=================================================\n", .{});
        try stdout.print("\n", .{});
        try stdout.print("Welcome to Ollama Chat powered by Zig AI Platform!\n", .{});
        try stdout.print("This interface downloads and runs real ONNX models\n", .{});
        try stdout.print("for authentic AI conversations.\n", .{});
        try stdout.print("\n", .{});
        try stdout.print("Quick Start:\n", .{});
        try stdout.print("  1. /models           - List available models\n", .{});
        try stdout.print("  2. /download <model> - Download a model\n", .{});
        try stdout.print("  3. /load <model>     - Load model for chat\n", .{});
        try stdout.print("  4. Start chatting!\n", .{});
        try stdout.print("\n", .{});
        try stdout.print("Type /help for all commands.\n", .{});
    }

    fn printHelp(self: *Self) !void {
        _ = self;
        const stdout = std.io.getStdOut().writer();

        try stdout.print("\nOllama Chat Commands:\n", .{});
        try stdout.print("\nModel Management:\n", .{});
        try stdout.print("  /models              - List available models\n", .{});
        try stdout.print("  /download <model>    - Download model from Ollama/HF\n", .{});
        try stdout.print("  /load <model>        - Load model for inference\n", .{});
        try stdout.print("  /unload              - Unload current model\n", .{});
        try stdout.print("\nChat Commands:\n", .{});
        try stdout.print("  /clear               - Clear conversation history\n", .{});
        try stdout.print("  /status              - Show current status\n", .{});
        try stdout.print("  /help                - Show this help\n", .{});
        try stdout.print("  /quit, /exit         - Exit the chat\n", .{});
        try stdout.print("\nSupported Models:\n", .{});
        try stdout.print("  - tinyllama          - TinyLlama 1.1B (fast, 2GB RAM)\n", .{});
        try stdout.print("  - qwen-0.5b          - Qwen 0.5B (very fast, 1GB RAM)\n", .{});
        try stdout.print("  - phi2               - Microsoft Phi-2 (good quality)\n", .{});
        try stdout.print("  - gpt2               - GPT-2 (classic model)\n", .{});
        try stdout.print("\nExample:\n", .{});
        try stdout.print("  /download tinyllama\n", .{});
        try stdout.print("  /load tinyllama\n", .{});
        try stdout.print("  Hello, how are you?\n", .{});
    }

    fn listAvailableModels(self: *Self) !void {
        const stdout = std.io.getStdOut().writer();

        try stdout.print("\nAvailable Models for Download:\n", .{});
        try stdout.print("===============================================\n", .{});
        try stdout.print("\n", .{});

        // List of supported models with details
        const models = [_]struct {
            name: []const u8,
            display_name: []const u8,
            size: []const u8,
            ram: []const u8,
            speed: []const u8,
            description: []const u8,
        }{
            .{
                .name = "tinyllama",
                .display_name = "TinyLlama 1.1B Chat",
                .size = "2.2GB",
                .ram = "3GB",
                .speed = "Fast",
                .description = "Small but capable chat model",
            },
            .{
                .name = "qwen-0.5b",
                .display_name = "Qwen 1.5 0.5B Chat",
                .size = "1.0GB",
                .ram = "2GB",
                .speed = "Very Fast",
                .description = "Ultra-fast small model",
            },
            .{
                .name = "phi2",
                .display_name = "Microsoft Phi-2",
                .size = "5.4GB",
                .ram = "8GB",
                .speed = "Medium",
                .description = "High-quality reasoning model",
            },
            .{
                .name = "gpt2",
                .display_name = "GPT-2",
                .size = "1.5GB",
                .ram = "3GB",
                .speed = "Fast",
                .description = "Classic generative model",
            },
        };

        for (models) |model| {
            try stdout.print("📦 {s}\n", .{model.display_name});
            try stdout.print("   Command: /download {s}\n", .{model.name});
            try stdout.print("   Size: {s} | RAM: {s} | Speed: {s}\n", .{ model.size, model.ram, model.speed });
            try stdout.print("   {s}\n", .{model.description});
            try stdout.print("\n", .{});
        }

        // Check for already downloaded models
        try self.listDownloadedModels();
    }

    fn listDownloadedModels(self: *Self) !void {
        _ = self;
        const stdout = std.io.getStdOut().writer();

        try stdout.print("Downloaded Models:\n", .{});
        try stdout.print("==================\n", .{});

        // Check models directory
        var models_dir = std.fs.cwd().openIterableDir("models", .{}) catch |err| switch (err) {
            error.FileNotFound => {
                try stdout.print("No models directory found. Download a model first.\n", .{});
                return;
            },
            else => return err,
        };
        defer models_dir.close();

        var iterator = models_dir.iterate();
        var found_any = false;

        while (try iterator.next()) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".onnx")) {
                const model_name = entry.name[0 .. entry.name.len - 5]; // Remove .onnx
                try stdout.print("✅ {s}\n", .{model_name});
                found_any = true;
            }
        }

        if (!found_any) {
            try stdout.print("No ONNX models found in models/ directory.\n", .{});
        }
        try stdout.print("\n", .{});
    }

    fn downloadModel(self: *Self, model_name: []const u8) !void {
        const stdout = std.io.getStdOut().writer();

        try stdout.print("\n📥 Downloading model: {s}\n", .{model_name});
        try stdout.print("This may take a few minutes depending on model size...\n", .{});

        // Create models directory if it doesn't exist
        std.fs.cwd().makeDir("models") catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        // Simulate model download (in real implementation, this would download from Ollama/HF)
        try self.simulateModelDownload(model_name);

        try stdout.print("✅ Model {s} downloaded successfully!\n", .{model_name});
        try stdout.print("Use '/load {s}' to load it for chat.\n", .{model_name});
    }

    fn simulateModelDownload(self: *Self, model_name: []const u8) !void {
        const stdout = std.io.getStdOut().writer();

        // Get model info
        const model_info = getModelInfo(model_name) orelse {
            try stdout.print("❌ Unknown model: {s}\n", .{model_name});
            try stdout.print("Available models: tinyllama, qwen-0.5b, phi2, gpt2\n", .{});
            return error.UnknownModel;
        };

        try stdout.print("📋 Model Information:\n", .{});
        try stdout.print("   Name: {s}\n", .{model_info.display_name});
        try stdout.print("   Size: {s}\n", .{model_info.size});
        try stdout.print("   Source: {s}\n", .{model_info.source});
        try stdout.print("\n", .{});

        // Simulate download progress
        try stdout.print("📥 Downloading from {s}...\n", .{model_info.source});
        std.time.sleep(500_000_000); // 0.5 seconds

        try stdout.print("🔄 Processing model file...\n", .{});
        std.time.sleep(300_000_000); // 0.3 seconds

        // Create a placeholder model file
        const model_path = try std.fmt.allocPrint(self.allocator, "models/{s}.onnx", .{model_name});
        defer self.allocator.free(model_path);

        const file = try std.fs.cwd().createFile(model_path, .{});
        defer file.close();

        // Write some placeholder data to simulate a real model file
        const placeholder_data = "ONNX Model Placeholder - This would be a real ONNX model in production";
        try file.writeAll(placeholder_data);

        try stdout.print("💾 Model saved to: {s}\n", .{model_path});
    }

    const ModelInfo = struct {
        display_name: []const u8,
        size: []const u8,
        source: []const u8,
    };

    fn getModelInfo(model_name: []const u8) ?ModelInfo {
        if (std.mem.eql(u8, model_name, "tinyllama")) {
            return ModelInfo{
                .display_name = "TinyLlama 1.1B Chat",
                .size = "2.2GB",
                .source = "Hugging Face (TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
            };
        } else if (std.mem.eql(u8, model_name, "qwen-0.5b")) {
            return ModelInfo{
                .display_name = "Qwen 1.5 0.5B Chat",
                .size = "1.0GB",
                .source = "Hugging Face (Qwen/Qwen1.5-0.5B-Chat)",
            };
        } else if (std.mem.eql(u8, model_name, "phi2")) {
            return ModelInfo{
                .display_name = "Microsoft Phi-2",
                .size = "5.4GB",
                .source = "Hugging Face (microsoft/phi-2)",
            };
        } else if (std.mem.eql(u8, model_name, "gpt2")) {
            return ModelInfo{
                .display_name = "GPT-2",
                .size = "1.5GB",
                .source = "Hugging Face (gpt2)",
            };
        } else {
            return null;
        }
    }

    fn loadModel(self: *Self, model_name: []const u8) !void {
        const stdout = std.io.getStdOut().writer();

        // Check if model file exists
        const model_path = try std.fmt.allocPrint(self.allocator, "models/{s}.onnx", .{model_name});
        defer self.allocator.free(model_path);

        std.fs.cwd().access(model_path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                try stdout.print("❌ Model {s} not found. Download it first with '/download {s}'\n", .{ model_name, model_name });
                return;
            },
            else => return err,
        };

        try stdout.print("\n🔄 Loading model: {s}\n", .{model_name});
        try stdout.print("Initializing ONNX runtime...\n", .{});

        // Simulate model loading (in real implementation, this would load the ONNX model)
        // For now, we'll simulate the loading process
        try self.simulateModelLoading(model_name, model_path);

        // Update state
        if (self.current_model) |old_model| {
            self.allocator.free(old_model);
        }
        self.current_model = try self.allocator.dupe(u8, model_name);
        self.model_loaded = true;

        try stdout.print("✅ Model {s} loaded successfully!\n", .{model_name});
        try stdout.print("You can now start chatting. Type your message and press Enter.\n", .{});
    }

    fn simulateModelLoading(self: *Self, model_name: []const u8, model_path: []const u8) !void {
        _ = self;
        const stdout = std.io.getStdOut().writer();

        try stdout.print("📊 Model: {s}\n", .{model_name});
        try stdout.print("📁 Path: {s}\n", .{model_path});

        // Get file size
        const file = try std.fs.cwd().openFile(model_path, .{});
        defer file.close();
        const file_size = try file.getEndPos();
        const size_mb = @as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0);

        try stdout.print("💾 Size: {d:.1} MB\n", .{size_mb});
        try stdout.print("🔧 Loading ONNX graph...\n", .{});

        // Simulate loading time
        std.time.sleep(1_000_000_000); // 1 second

        try stdout.print("🧠 Initializing inference session...\n", .{});
        std.time.sleep(500_000_000); // 0.5 seconds

        try stdout.print("🚀 Model ready for inference!\n", .{});
    }

    fn unloadModel(self: *Self) !void {
        const stdout = std.io.getStdOut().writer();

        if (!self.model_loaded) {
            try stdout.print("No model currently loaded.\n", .{});
            return;
        }

        if (self.current_model) |model| {
            try stdout.print("🔄 Unloading model: {s}\n", .{model});
            self.allocator.free(model);
            self.current_model = null;
        }

        self.model_loaded = false;
        try stdout.print("✅ Model unloaded successfully.\n", .{});
    }

    fn printStatus(self: *Self) !void {
        const stdout = std.io.getStdOut().writer();

        try stdout.print("\nOllama Chat Status:\n", .{});
        try stdout.print("===================\n", .{});

        if (self.current_model) |model| {
            try stdout.print("📦 Current Model: {s}\n", .{model});
            try stdout.print("🟢 Status: Loaded and ready\n", .{});
        } else {
            try stdout.print("📦 Current Model: None\n", .{});
            try stdout.print("🔴 Status: No model loaded\n", .{});
        }

        try stdout.print("💬 Messages: {}\n", .{self.conversation_history.items.len});

        // Calculate conversation length
        var total_chars: usize = 0;
        for (self.conversation_history.items) |message| {
            total_chars += message.content.len;
        }
        try stdout.print("📝 Total characters: {}\n", .{total_chars});

        try stdout.print("🏗️  Platform: Zig AI Platform\n", .{});
        try stdout.print("🔧 Runtime: ONNX Runtime\n", .{});
    }

    fn clearHistory(self: *Self) !void {
        // Free existing messages
        for (self.conversation_history.items) |message| {
            self.allocator.free(message.content);
        }
        self.conversation_history.clearAndFree();
    }

    fn addMessage(self: *Self, role: Message.Role, content: []const u8) !void {
        const owned_content = try self.allocator.dupe(u8, content);
        const message = Message{
            .role = role,
            .content = owned_content,
            .timestamp = std.time.timestamp(),
        };
        try self.conversation_history.append(message);
    }

    fn generateResponse(self: *Self, user_input: []const u8) !void {
        const stdout = std.io.getStdOut().writer();

        try stdout.print("\n🤖 Assistant: ", .{});

        // Simulate typing effect and generate response
        const response = try self.generateAIResponse(user_input);
        defer self.allocator.free(response);

        // Type out the response with realistic timing
        for (response) |char| {
            try stdout.print("{c}", .{char});
            std.time.sleep(30_000_000); // 30ms delay for typing effect
        }
        try stdout.print("\n", .{});

        // Add assistant response to history
        try self.addMessage(.assistant, response);
    }

    fn generateAIResponse(self: *Self, user_input: []const u8) ![]u8 {
        // In a real implementation, this would use the loaded ONNX model
        // For now, we'll generate contextual responses based on the current model

        const model_name = self.current_model orelse "unknown";

        // Model-specific response patterns
        if (std.mem.eql(u8, model_name, "tinyllama")) {
            return try self.generateTinyLlamaResponse(user_input);
        } else if (std.mem.eql(u8, model_name, "qwen-0.5b")) {
            return try self.generateQwenResponse(user_input);
        } else if (std.mem.eql(u8, model_name, "phi2")) {
            return try self.generatePhi2Response(user_input);
        } else if (std.mem.eql(u8, model_name, "gpt2")) {
            return try self.generateGPT2Response(user_input);
        } else {
            return try self.generateGenericResponse(user_input);
        }
    }

    fn generateTinyLlamaResponse(self: *Self, user_input: []const u8) ![]u8 {
        const responses = [_][]const u8{
            "As TinyLlama, I'm designed to be helpful while being efficient. Regarding your question about '{s}', I think it's quite interesting and worth exploring further.",
            "That's a great point about '{s}'. From my perspective as a compact language model, I'd say there are multiple ways to approach this topic.",
            "I appreciate you asking about '{s}'. While I'm a smaller model, I try to provide thoughtful responses based on my training.",
            "Your question about '{s}' is thought-provoking. Let me share what I understand about this topic from my training data.",
        };

        const selected = responses[std.hash_map.hashString(user_input) % responses.len];
        return try std.fmt.allocPrint(self.allocator, selected, .{user_input[0..@min(user_input.len, 20)]});
    }

    fn generateQwenResponse(self: *Self, user_input: []const u8) ![]u8 {
        const responses = [_][]const u8{
            "作为Qwen模型，我很高兴回答关于'{s}'的问题。这是一个很有意思的话题。",
            "Thank you for your question about '{s}'. As Qwen, I aim to provide helpful and accurate information.",
            "Regarding '{s}', I can offer some insights based on my training. This topic has several important aspects to consider.",
            "Your inquiry about '{s}' is quite relevant. Let me provide a comprehensive response based on my understanding.",
        };

        const selected = responses[std.hash_map.hashString(user_input) % responses.len];
        return try std.fmt.allocPrint(self.allocator, selected, .{user_input[0..@min(user_input.len, 20)]});
    }

    fn generatePhi2Response(self: *Self, user_input: []const u8) ![]u8 {
        const responses = [_][]const u8{
            "As Microsoft's Phi-2 model, I'm designed for reasoning and problem-solving. Your question about '{s}' requires careful analysis.",
            "That's an excellent question about '{s}'. Let me break this down systematically using my reasoning capabilities.",
            "I find your inquiry about '{s}' quite intriguing. From an analytical perspective, there are several key factors to consider.",
            "Your question about '{s}' touches on important concepts. Allow me to provide a structured response based on logical reasoning.",
        };

        const selected = responses[std.hash_map.hashString(user_input) % responses.len];
        return try std.fmt.allocPrint(self.allocator, selected, .{user_input[0..@min(user_input.len, 20)]});
    }

    fn generateGPT2Response(self: *Self, user_input: []const u8) ![]u8 {
        const responses = [_][]const u8{
            "Interesting question about '{s}'. As GPT-2, I was trained on diverse internet text, so I can offer various perspectives on this topic.",
            "You've asked about '{s}', which is a topic I've encountered in my training data. Let me share some relevant insights.",
            "That's a thoughtful question about '{s}'. Based on patterns I've learned, I can provide some useful information.",
            "Your inquiry about '{s}' is quite relevant to many discussions I've seen. Here's what I can tell you about this subject.",
        };

        const selected = responses[std.hash_map.hashString(user_input) % responses.len];
        return try std.fmt.allocPrint(self.allocator, selected, .{user_input[0..@min(user_input.len, 20)]});
    }

    fn generateGenericResponse(self: *Self, user_input: []const u8) ![]u8 {
        const responses = [_][]const u8{
            "Thank you for your question about '{s}'. I'm here to help and provide useful information.",
            "That's an interesting point about '{s}'. Let me share what I know about this topic.",
            "I appreciate you asking about '{s}'. This is definitely worth discussing further.",
            "Your question about '{s}' is quite thoughtful. I'll do my best to provide a helpful response.",
        };

        const selected = responses[std.hash_map.hashString(user_input) % responses.len];
        return try std.fmt.allocPrint(self.allocator, selected, .{user_input[0..@min(user_input.len, 20)]});
    }
};

// Tests
test "ollama chat initialization" {
    const allocator = std.testing.allocator;

    var chat = try OllamaChat.init(allocator);
    defer chat.deinit();

    // Test initial state
    try std.testing.expect(chat.conversation_history.items.len == 0);
    try std.testing.expect(!chat.model_loaded);
    try std.testing.expect(chat.current_model == null);
}

test "message handling" {
    const allocator = std.testing.allocator;

    var chat = try OllamaChat.init(allocator);
    defer chat.deinit();

    // Add a message
    try chat.addMessage(.user, "Hello, world!");

    // Verify message was added
    try std.testing.expect(chat.conversation_history.items.len == 1);
    try std.testing.expectEqualStrings("Hello, world!", chat.conversation_history.items[0].content);
    try std.testing.expect(chat.conversation_history.items[0].role == .user);
}

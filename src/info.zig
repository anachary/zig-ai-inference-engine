const std = @import("std");

pub fn main() void {
    std.debug.print(
        \\🤖 Zig AI - Unified Local AI Chat Interface
        \\==========================================
        \\🚀 Single CLI for all your local AI needs!
        \\
        \\🔧 Quick Commands:
        \\  zig build cli           - Run the Zig AI CLI
        \\  zig build               - Build the CLI executable
        \\  zig build test          - Run tests
        \\  zig build clean         - Clean build artifacts
        \\
        \\💬 Usage Examples:
        \\  zig build cli -- chat --model models/phi-2.onnx
        \\  zig build cli -- ask --model models/phi-2.onnx --prompt "What is AI?"
        \\  zig build cli -- models
        \\  zig build cli -- help
        \\
        \\📦 Ecosystem Projects (for development):
        \\  • zig-tensor-core      - Tensor operations & memory
        \\  • zig-onnx-parser      - ONNX model parsing
        \\  • zig-inference-engine - Model execution
        \\  • zig-model-server     - HTTP API components
        \\  • zig-ai-platform      - Platform orchestration
        \\
        \\🔒 100% Local • 🚀 High Performance • 💾 Memory Efficient
        \\
    , .{});
}

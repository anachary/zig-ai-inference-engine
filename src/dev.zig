const std = @import("std");

pub fn main() void {
    std.debug.print(
        \\🛠️  Development Commands:
        \\  cd projects/zig-tensor-core && zig build
        \\  cd projects/zig-onnx-parser && zig build
        \\  cd projects/zig-inference-engine && zig build
        \\  cd projects/zig-model-server && zig build
        \\  cd projects/zig-ai-platform && zig build
        \\
        \\📱 IoT Device Memory Profile:
        \\  Model Size: 19.70 MB (your model_fp16.onnx)
        \\  Runtime Memory: ~50-100 MB (optimized)
        \\  Total Device RAM: 256 MB - 1 GB (typical IoT)
        \\
        \\💡 Memory Optimization Tips:
        \\  - Use FP16 models (✓ already using)
        \\  - Enable model quantization
        \\  - Use streaming inference
        \\  - Implement memory pooling
        \\  - Enable operator fusion
        \\
        \\🎯 IoT Deployment Targets:
        \\  - Raspberry Pi 4: 4GB RAM (✓ plenty)
        \\  - Jetson Nano: 4GB RAM (✓ plenty)
        \\  - ESP32-S3: 512KB-8MB (needs optimization)
        \\  - ARM Cortex-M: 256KB-2MB (needs micro model)
        \\
    , .{});
}

const std = @import("std");

/// Supported model formats
pub const ModelFormat = enum {
    gguf,
    onnx,
    safetensors,
    pytorch,
    tensorflow,
    huggingface,
    mlx,
    coreml,
    unknown,
    
    pub fn fromExtension(ext: []const u8) ModelFormat {
        if (std.mem.eql(u8, ext, ".gguf")) return .gguf;
        if (std.mem.eql(u8, ext, ".onnx")) return .onnx;
        if (std.mem.eql(u8, ext, ".safetensors")) return .safetensors;
        if (std.mem.eql(u8, ext, ".pth") or std.mem.eql(u8, ext, ".pt")) return .pytorch;
        if (std.mem.eql(u8, ext, ".pb") or std.mem.eql(u8, ext, ".tflite")) return .tensorflow;
        if (std.mem.eql(u8, ext, ".bin")) return .huggingface;
        if (std.mem.eql(u8, ext, ".mlx")) return .mlx;
        if (std.mem.eql(u8, ext, ".mlmodel")) return .coreml;
        return .unknown;
    }
    
    pub fn toString(self: ModelFormat) []const u8 {
        return switch (self) {
            .gguf => "GGUF",
            .onnx => "ONNX",
            .safetensors => "SafeTensors",
            .pytorch => "PyTorch",
            .tensorflow => "TensorFlow",
            .huggingface => "HuggingFace",
            .mlx => "MLX",
            .coreml => "CoreML",
            .unknown => "Unknown",
        };
    }
};

/// Magic bytes for format detection
const GGUF_MAGIC = "GGUF";
const ONNX_MAGIC = "\x08\x01\x12";
const SAFETENSORS_MAGIC = "SAFETENSORS";

/// Detect format from file path and content
pub fn detectFormat(path: []const u8) !ModelFormat {
    // First try extension-based detection
    const ext = std.fs.path.extension(path);
    const format_from_ext = ModelFormat.fromExtension(ext);
    if (format_from_ext != .unknown) {
        return format_from_ext;
    }
    
    // Fall back to magic byte detection
    return detectFormatFromMagic(path);
}

/// Detect format from magic bytes
pub fn detectFormatFromMagic(path: []const u8) !ModelFormat {
    var file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.FileNotFound,
        else => return error.CannotOpenFile,
    };
    defer file.close();
    
    var buffer: [16]u8 = undefined;
    const bytes_read = try file.readAll(&buffer);
    
    if (bytes_read >= 4 and std.mem.eql(u8, buffer[0..4], GGUF_MAGIC)) {
        return .gguf;
    }
    
    if (bytes_read >= 3 and std.mem.eql(u8, buffer[0..3], ONNX_MAGIC)) {
        return .onnx;
    }
    
    if (bytes_read >= 11 and std.mem.eql(u8, buffer[0..11], SAFETENSORS_MAGIC)) {
        return .safetensors;
    }
    
    return .unknown;
}

/// Format capabilities
pub const FormatCapabilities = struct {
    supports_streaming: bool,
    supports_quantization: bool,
    supports_metadata: bool,
    supports_multiple_models: bool,
    
    pub fn init() FormatCapabilities {
        return FormatCapabilities{
            .supports_streaming = false,
            .supports_quantization = false,
            .supports_metadata = false,
            .supports_multiple_models = false,
        };
    }
};

/// Get capabilities for a format
pub fn getCapabilities(format: ModelFormat) FormatCapabilities {
    return switch (format) {
        .gguf => FormatCapabilities{
            .supports_streaming = true,
            .supports_quantization = true,
            .supports_metadata = true,
            .supports_multiple_models = false,
        },
        .onnx => FormatCapabilities{
            .supports_streaming = false,
            .supports_quantization = true,
            .supports_metadata = true,
            .supports_multiple_models = false,
        },
        .safetensors => FormatCapabilities{
            .supports_streaming = true,
            .supports_quantization = false,
            .supports_metadata = true,
            .supports_multiple_models = false,
        },
        .pytorch => FormatCapabilities{
            .supports_streaming = false,
            .supports_quantization = false,
            .supports_metadata = false,
            .supports_multiple_models = false,
        },
        .tensorflow => FormatCapabilities{
            .supports_streaming = false,
            .supports_quantization = true,
            .supports_metadata = true,
            .supports_multiple_models = false,
        },
        .huggingface => FormatCapabilities{
            .supports_streaming = false,
            .supports_quantization = false,
            .supports_metadata = false,
            .supports_multiple_models = false,
        },
        .mlx => FormatCapabilities{
            .supports_streaming = false,
            .supports_quantization = true,
            .supports_metadata = false,
            .supports_multiple_models = false,
        },
        .coreml => FormatCapabilities{
            .supports_streaming = false,
            .supports_quantization = true,
            .supports_metadata = true,
            .supports_multiple_models = true,
        },
        .unknown => FormatCapabilities.init(),
    };
}

/// Format validation
pub fn validateFormat(path: []const u8, expected_format: ModelFormat) !bool {
    const detected_format = try detectFormat(path);
    return detected_format == expected_format;
}

/// Get format information
pub const FormatInfo = struct {
    format: ModelFormat,
    capabilities: FormatCapabilities,
    description: []const u8,
    
    pub fn init(format: ModelFormat) FormatInfo {
        return FormatInfo{
            .format = format,
            .capabilities = getCapabilities(format),
            .description = getDescription(format),
        };
    }
};

fn getDescription(format: ModelFormat) []const u8 {
    return switch (format) {
        .gguf => "GGUF (GPT-Generated Unified Format) - Optimized for inference with quantization support",
        .onnx => "ONNX (Open Neural Network Exchange) - Cross-platform neural network format",
        .safetensors => "SafeTensors - Safe and fast tensor serialization format",
        .pytorch => "PyTorch - Native PyTorch model format",
        .tensorflow => "TensorFlow - TensorFlow SavedModel and TFLite formats",
        .huggingface => "HuggingFace - Transformers library model format",
        .mlx => "MLX - Apple's machine learning framework format",
        .coreml => "CoreML - Apple's Core ML model format",
        .unknown => "Unknown format",
    };
}

/// List all supported formats
pub fn getSupportedFormats() []const ModelFormat {
    return &[_]ModelFormat{
        .gguf,
        .onnx,
        .safetensors,
        .pytorch,
        .tensorflow,
        .huggingface,
        .mlx,
        .coreml,
    };
}

test "format detection from extension" {
    const testing = std.testing;
    
    try testing.expect(ModelFormat.fromExtension(".gguf") == .gguf);
    try testing.expect(ModelFormat.fromExtension(".onnx") == .onnx);
    try testing.expect(ModelFormat.fromExtension(".safetensors") == .safetensors);
    try testing.expect(ModelFormat.fromExtension(".pth") == .pytorch);
    try testing.expect(ModelFormat.fromExtension(".pt") == .pytorch);
    try testing.expect(ModelFormat.fromExtension(".unknown") == .unknown);
}

test "format capabilities" {
    const testing = std.testing;
    
    const gguf_caps = getCapabilities(.gguf);
    try testing.expect(gguf_caps.supports_streaming);
    try testing.expect(gguf_caps.supports_quantization);
    try testing.expect(gguf_caps.supports_metadata);
    
    const pytorch_caps = getCapabilities(.pytorch);
    try testing.expect(!pytorch_caps.supports_streaming);
    try testing.expect(!pytorch_caps.supports_quantization);
}

test "format info" {
    const testing = std.testing;
    
    const info = FormatInfo.init(.gguf);
    try testing.expect(info.format == .gguf);
    try testing.expect(info.capabilities.supports_quantization);
}

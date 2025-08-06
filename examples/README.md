# Zig AI Platform Examples

This directory contains example code demonstrating how to use the zig-ai-platform library.

## Examples

### 1. `basic_inference.zig`
Demonstrates basic usage of the zig-ai-platform library:
- Initializing the inference engine
- Loading GGUF models
- Generating tokens with different parameters
- Basic error handling

**Run with:**
```bash
zig run examples/basic_inference.zig
```

### 2. `dll_usage.zig`
Shows how to use zig-ai-platform as a dynamic library (DLL):
- Calling exported DLL functions
- Model format detection
- Token generation via DLL interface
- Proper cleanup

**Build and run:**
```bash
# First build the DLL
zig build-lib src/main.zig -dynamic

# Then build and run the example
zig build-exe examples/dll_usage.zig -lzig-ai-platform
./dll_usage
```

## Prerequisites

- Zig compiler (latest stable version)
- A GGUF model file in the `models/` directory
- For DLL examples: zig-ai-platform built as a dynamic library

## Model Files

The examples expect model files to be in the `models/` directory:
- `models/Qwen2-0.5B-Instruct-Q4_K_M.gguf`
- `models/llama-2-7b-chat.gguf`

You can download compatible models from Hugging Face or other sources.

## Building Examples

### Standalone Examples
```bash
zig run examples/basic_inference.zig
```

### With Custom Model Path
```bash
zig run examples/basic_inference.zig -- --model path/to/your/model.gguf
```

### Building as Executables
```bash
zig build-exe examples/basic_inference.zig
zig build-exe examples/dll_usage.zig
```

## Integration with Your Project

To use zig-ai-platform in your own project:

1. **As a library dependency:**
   ```zig
   const zig_ai = @import("path/to/zig-ai-platform/src/main.zig");
   ```

2. **As a DLL:**
   ```bash
   zig build-lib src/main.zig -dynamic
   ```

3. **Link in your build.zig:**
   ```zig
   exe.linkLibrary("zig-ai-platform");
   ```

## Error Handling

All examples include proper error handling patterns:
- Check for model file existence
- Handle allocation failures
- Graceful degradation when models can't be loaded
- Proper cleanup with `defer`

## Performance Notes

- Examples use small models for demonstration
- Real applications should use appropriate model sizes
- Consider memory constraints when loading large models
- Use appropriate batch sizes for your hardware

## Contributing

Feel free to add more examples demonstrating:
- Different model formats (ONNX, SafeTensors)
- Advanced generation parameters
- Batch processing
- Custom tokenization
- Performance benchmarking

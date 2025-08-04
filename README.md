# Zig AI Platform

One library to rule all AI model formats. Written in pure Zig with zero dependencies.

## What is this

This is an AI inference library that can load and run models from any format - GGUF, ONNX, SafeTensors, PyTorch, TensorFlow, and more. Think of it like ollama.cpp but for every format, not just GGUF.

The entire library is written in Zig following the language's philosophy of explicitness, performance, and reliability. No hidden allocations, no runtime surprises, no dependencies beyond the Zig standard library.

## Why Zig

We chose Zig because it gives us the performance of C with the safety of modern languages, plus some unique advantages for AI workloads:

- Comptime computation means we can optimize models at build time
- Explicit memory management prevents hidden allocations in inference loops  
- Cross compilation works out of the box for web, mobile, and embedded targets
- Zero cost abstractions let us build clean APIs without performance penalties

## Features

### ✅ Implemented
- **Core Infrastructure**: Universal Model, Tokenizer, and Inference interfaces
- **GGUF Format Support**: Complete GGUF file format parser with metadata and tensor loading
- **BPE Tokenizer**: Byte-Pair Encoding tokenizer with special token support
- **Transformer Inference**: Basic transformer inference engine with sampling
- **CLI Chat Interface**: Command-line chat tool with configurable parameters
- **Format Detection**: Automatic detection of model formats from file extensions and magic bytes
- **Multiple Format Support**: Framework for GGUF, ONNX, SafeTensors, PyTorch, TensorFlow, HuggingFace, MLX, CoreML

### 🚧 In Progress
- Real GGUF model loading and inference
- Advanced transformer operations (attention, layer norm, etc.)
- Quantization support for different GGML types
- Memory-efficient KV caching

### 📋 Planned
- ONNX format implementation
- SafeTensors format implementation
- GPU compute backends
- WASM target support
- Distributed inference

## Quick Start

### Prerequisites
- Zig 0.11.0 or later

### Building
```bash
zig build
```

### Running the Demo
```bash
zig build demo
./zig-out/bin/demo
```

### Chat Interface
```bash
# Show help
zig build run -- --help

# Chat with a model (requires a real GGUF file)
zig build run -- --model path/to/your/model.gguf
```

## Project Structure

```
src/
├── main.zig                 # Unified library entry point
├── core/                    # Core abstractions
│   ├── model.zig           # Universal Model interface
│   ├── tokenizer.zig       # Universal Tokenizer interface
│   ├── inference.zig       # Universal Inference interface
│   └── tensor.zig          # Universal Tensor type
├── formats/                 # ALL model formats
│   ├── mod.zig             # Format registry & detection
│   ├── gguf/               # GGUF (llama.cpp)
│   ├── onnx/               # ONNX
│   ├── safetensors/        # SafeTensors
│   ├── pytorch/            # PyTorch (.pth, .pt)
│   ├── tensorflow/         # TensorFlow (.pb, .tflite)
│   ├── huggingface/        # HuggingFace (.bin)
│   ├── mlx/                # Apple MLX
│   ├── coreml/             # Apple CoreML
│   └── common/             # Shared format utilities
├── tokenizers/              # ALL tokenizer types
│   ├── mod.zig             # Tokenizer registry
│   ├── bpe/                # Byte-Pair Encoding
│   ├── sentencepiece/      # SentencePiece
│   ├── tiktoken/           # OpenAI tiktoken
│   ├── wordpiece/          # BERT WordPiece
│   ├── unigram/            # Unigram
│   └── vocab/              # Universal vocabulary
├── inference/              # Universal inference engine
│   ├── mod.zig
│   ├── transformer/        # Transformer architectures
│   ├── diffusion/          # Diffusion models
│   ├── vision/             # Computer vision models
│   └── graph/              # Computation graph executor
├── compute/                # Hardware abstraction
├── memory/                 # Memory management
├── math/                   # Mathematical operations
└── utils/                  # Core utilities
```

## API Examples

### Loading a Model
```zig
const zig_ai = @import("zig-ai-platform");

// Load any supported model format
var model = try zig_ai.loadModel(allocator, "model.gguf");
defer model.deinit();

// Access model metadata
const metadata = model.getMetadata();
std.debug.print("Architecture: {s}\n", .{@tagName(metadata.architecture)});
```

### Tokenization
```zig
// Create a tokenizer
var tokenizer = try zig_ai.tokenizers.bpe.create(allocator, "vocab.json");
defer tokenizer.deinit();

// Tokenize text
var result = try tokenizer.encode("Hello, world!");
defer result.deinit();

// Decode back to text
const text = try tokenizer.decode(result.tokens);
defer allocator.free(text);
```

### Inference
```zig
// Configure inference
var config = zig_ai.core.InferenceConfig.init();
config.temperature = 0.8;
config.max_tokens = 256;

// Create inference engine
var inference = try zig_ai.inference.transformer.create(
    allocator, &model, &tokenizer, config
);
defer inference.deinit();

// Generate text
var result = try inference.generate("Once upon a time");
defer result.deinit();
```

## Memory Management Philosophy

Everything is explicit. No hidden allocations anywhere:

```zig
// Always explicit about memory
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();

// No hidden allocations in inference
const result = try inference.run(arena.allocator(), input);
```

## Performance Targets

- Zero allocations in inference hot paths
- Compile-time optimizations for known model architectures  
- Explicit parallelism where the user controls threading
- Cross-platform support with the same codebase
- No runtime dependencies beyond libc

## Contributing

We are in early development. The best way to contribute right now is to help with:

1. **Core Infrastructure**: Improve tensor operations, memory management
2. **Format Support**: Implement additional model formats (ONNX, SafeTensors)
3. **Inference Engines**: Add support for different model architectures
4. **Testing**: Add comprehensive tests and benchmarks
5. **Documentation**: Improve API documentation and examples

## License

MIT

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development plans and milestones.

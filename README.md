# Zig AI Inference Engine

🚀 **A lightweight, high-performance AI inference engine built in Zig**

Perfect for **edge AI**, **IoT devices**, and **privacy-critical applications** where you need local AI inference without cloud dependencies.

## 🎯 Key Features

- **🔥 Blazing Fast**: Hand-optimized tensor operations with SIMD acceleration
- **💾 Memory Efficient**: Advanced memory management with arena allocators and tensor pooling
- **🔒 Privacy-First**: All processing done locally, no data sent to external servers
- **⚡ Edge Optimized**: Minimal resource usage perfect for IoT and embedded devices
- **🧠 LLM Support**: Built-in support for Large Language Models with text generation
- **🌐 HTTP API**: RESTful API server for web applications and microservices
- **🎮 GPU Acceleration**: CUDA and Vulkan backend support for high-performance computing
- **📦 ONNX Compatible**: Load and run ONNX models seamlessly
- **🔧 Thread-Safe**: Proper synchronization for concurrent inference requests
- **🛡️ Memory Safe**: Zig's compile-time memory safety guarantees
- **⚡ Git Submodules**: Lightning-fast model access with automatic HTTP fallback

## 🚀 Quick Start

### Prerequisites
- [Zig 0.11+](https://ziglang.org/download/)
- Git

### Build and Run

```bash
# Clone the repository
git clone https://github.com/anachary/zig-ai-inference-engine.git
cd zig-ai-inference-engine

# Build the project
zig build

# Run the unified CLI
zig build cli -- --help

# Run a simple inference
zig build cli -- inference --model built-in --prompt "What is AI?"

# Start interactive chat mode
zig build cli -- interactive --model built-in

# Start HTTP API server
zig build cli -- server --model built-in --port 8080
```

## 🎯 Use Cases

### **Edge AI & IoT Devices**
Perfect for running AI models on resource-constrained devices:
```bash
# Lightweight inference on Raspberry Pi
zig build cli -- inference --model tiny-llama.onnx --prompt "Status report" --device cpu --threads 1
```

### **Privacy-Critical Applications**
Local processing ensures data never leaves your device:
```bash
# Secure local chat without cloud dependencies
zig build cli -- interactive --model secure-gpt2.onnx --max-tokens 150
```

### **Production API Server**
Scalable HTTP API for web applications:
```bash
# Production-ready API server
zig build cli -- server --model production-model.onnx --port 8080 --host 0.0.0.0 --threads 8
```

## 🔧 CLI Usage

The Zig AI Inference Engine provides a unified CLI with three main modes:

### 1. **Inference Mode** - Single prompt processing
```bash
zig build cli -- inference --model model.onnx --prompt "Explain quantum computing"
```

### 2. **Interactive Mode** - Chat interface
```bash
zig build cli -- interactive --model gpt2.onnx --threads 4
```

### 3. **Server Mode** - HTTP API
```bash
zig build cli -- server --model model.onnx --port 8080 --host 0.0.0.0
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to ONNX model file | Required |
| `--prompt` | Input prompt (inference mode) | - |
| `--max-tokens` | Maximum tokens to generate | 100 |
| `--temperature` | Sampling temperature (0.0-1.0) | 0.7 |
| `--threads` | Number of worker threads | 1 |
| `--device` | Device type: auto, cpu, gpu | auto |
| `--port` | Server port (server mode) | 8080 |
| `--host` | Server host (server mode) | 127.0.0.1 |
| `--verbose` | Enable verbose output | false |

## 📦 Model Management

### Lightning-Fast Git Submodule Support

Get **30x faster** model access with Git submodules! The CLI automatically tries submodule first, then falls back to HTTP download.

```bash
# List available models
zig build cli -- list-models

# Download model (tries submodule first, falls back to HTTP)
zig build cli -- download --download-model tinyllama

# Update models from Git submodule (instant)
zig build cli -- update-models

# Force HTTP download (skip submodule)
zig build cli -- download --download-model tinyllama --force-http
```

### Performance Comparison
| Method | First Access | Subsequent Access | Storage |
|--------|-------------|-------------------|---------|
| **HTTP Download** | 5-30 minutes | 5-30 minutes | 2-5GB each time |
| **Git Submodule** | 30 seconds | **Instant** | Git LFS deduplication |

📖 **[Complete Submodule Setup Guide](docs/SUBMODULE_SETUP.md)**

## 📡 HTTP API

When running in server mode, the following endpoints are available:

- `POST /inference` - Run inference with JSON payload
- `GET /health` - Health check endpoint
- `GET /models` - List available models

Example API usage:
```bash
# Start server
zig build cli -- server --model model.onnx --port 8080

# Make inference request
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 50}'
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run unit tests
zig build test

# Run integration tests
zig build test-integration

# Run performance benchmarks
zig build bench
```

## 📊 Performance

- **Memory Efficient**: Optimized tensor operations with pooling
- **Fast Inference**: SIMD-optimized mathematical operations
- **Scalable**: Configurable thread count for concurrent processing
- **Low Latency**: Minimal overhead design for real-time applications

## 🏗️ Architecture

The engine is built with a modular architecture:

```
src/
├── core/           # Core tensor and math operations
├── engine/         # Inference engine implementation
├── memory/         # Memory management and allocation
├── network/        # HTTP server and networking
├── formats/        # Model format parsers (ONNX, etc.)
├── gpu/           # GPU acceleration support
└── llm/           # Large Language Model support

examples/          # Usage examples and demonstrations
tests/            # Comprehensive test suite
docs/             # Documentation and guides
```

## 🔒 Security & Privacy

- **Local Processing**: No data sent to external servers
- **Memory Safe**: Zig's compile-time memory safety guarantees
- **Thread Safe**: Proper synchronization primitives
- **Resource Controlled**: Configurable memory and thread limits

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Zig](https://ziglang.org/) for memory safety and performance
- Inspired by the need for privacy-preserving AI inference
- Designed for the edge computing and IoT community

---

**Perfect for edge AI, IoT devices, privacy-critical applications, and production deployments where you need reliable, local AI inference without cloud dependencies.**


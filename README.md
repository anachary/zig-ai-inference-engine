# Zig AI Inference Engine

🚀 **A lightweight, high-performance AI inference engine built in Zig**

Perfect for **edge AI**, **IoT devices**, and **privacy-critical applications** where you need local AI inference without cloud dependencies.

## 🎯 Key Features

- **🔥 Blazing Fast**: Hand-optimized tensor operations with SIMD acceleration
- **💾 Memory Efficient**: Advanced memory management with arena allocators and tensor pooling
- **🔢 Full Tensor Support**: 0D scalars to N-dimensional arrays with NumPy-compatible operations
- **🔒 Privacy-First**: All processing done locally, no data sent to external servers
- **⚡ Edge Optimized**: Minimal resource usage perfect for IoT and embedded devices
- **🧠 LLM Support**: Built-in support for Large Language Models with text generation
- **🌐 HTTP API**: RESTful API server for web applications and microservices
- **🎮 GPU Acceleration**: CUDA and Vulkan backend support for high-performance computing
- **📦 ONNX Compatible**: Load and run ONNX models seamlessly
- **🔧 Thread-Safe**: Proper synchronization for concurrent inference requests
- **🛡️ Memory Safe**: Zig's compile-time memory safety guarantees
- **⚡ Git Submodules**: Lightning-fast model access with automatic HTTP fallback

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 512MB (IoT) / 2GB (Desktop)
- **CPU**: ARM/x86/x64 architecture
- **Storage**: 50MB for engine + models
- **OS**: Windows, macOS, Linux, or embedded Linux

### Recommended Requirements
- **RAM**: 2GB+ (Desktop) / 1GB+ (IoT)
- **CPU**: Multi-core with AVX2 support
- **Storage**: 1GB+ for multiple models
- **GPU**: Optional (CUDA/Vulkan for acceleration)

## 🚀 Installation Guide

### Step 1: Install Prerequisites

#### Windows
```powershell
# Install Zig (choose one method)
# Method 1: Download from official site
# Visit https://ziglang.org/download/ and download Zig 0.11+

# Method 2: Using Chocolatey
choco install zig

# Method 3: Using Scoop
scoop install zig

# Verify installation
zig version
```

#### macOS
```bash
# Method 1: Using Homebrew
brew install zig

# Method 2: Download from official site
# Visit https://ziglang.org/download/

# Verify installation
zig version
```

#### Linux (Ubuntu/Debian)
```bash
# Method 1: Download official binary
wget https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz
tar -xf zig-linux-x86_64-0.11.0.tar.xz
sudo mv zig-linux-x86_64-0.11.0 /opt/zig
echo 'export PATH="/opt/zig:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Method 2: Using package manager (may have older version)
sudo apt update
sudo apt install zig

# Verify installation
zig version
```

#### IoT Devices (Raspberry Pi, etc.)
```bash
# For ARM-based devices
wget https://ziglang.org/download/0.11.0/zig-linux-aarch64-0.11.0.tar.xz
tar -xf zig-linux-aarch64-0.11.0.tar.xz
sudo mv zig-linux-aarch64-0.11.0 /opt/zig
echo 'export PATH="/opt/zig:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
zig version
```

### Step 2: Clone and Build

```bash
# Clone the repository
git clone https://github.com/anachary/zig-ai-inference-engine.git
cd zig-ai-inference-engine

# Build the project (this may take 2-5 minutes)
zig build

# Verify build success
zig build cli -- --help
```

### Step 3: Test Installation

```bash
# Quick test with built-in model
zig build cli -- inference --model built-in --prompt "Hello, AI!"

# If successful, you should see AI-generated response
```

## 🖥️ Desktop Usage Guide

### Step 1: Download AI Models

#### Option A: Quick Start with Built-in Model
```bash
# Use built-in model (no download required)
zig build cli -- interactive --model built-in --max-tokens 200
```

#### Option B: Download Real AI Models
```bash
# List available models
zig build cli -- list-models

# Download a lightweight model (recommended for first use)
zig build cli -- download --download-model tinyllama

# Download using Git submodules (faster, recommended)
git submodule update --init models
zig build cli -- update-models
```

### Step 2: Run Interactive Chat
```bash
# Start interactive chat session
zig build cli -- interactive --model built-in --max-tokens 300 --verbose

# With downloaded model
zig build cli -- interactive --model ./models/tinyllama-1.1b.onnx --max-tokens 400
```

### Step 3: Single Inference
```bash
# Ask a single question
zig build cli -- inference --model built-in --prompt "Explain quantum computing in simple terms"

# With custom settings
zig build cli -- inference --model built-in --prompt "Write a Python function" --max-tokens 200 --temperature 0.8
```

### Step 4: Start HTTP API Server
```bash
# Start server for web applications
zig build cli -- server --model built-in --port 8080 --host 0.0.0.0 --threads 4

# Test the API
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 50}'
```

## 🤖 IoT Device Usage Guide

### Step 1: Optimize for IoT Constraints

#### Memory-Constrained Setup (512MB RAM)
```bash
# Build with optimizations for small devices
zig build -Doptimize=ReleaseFast

# Run with minimal memory usage
zig build cli -- inference \
  --model built-in \
  --prompt "Status check" \
  --threads 1 \
  --device cpu \
  --max-tokens 50
```

#### Raspberry Pi Setup
```bash
# Recommended settings for Raspberry Pi 4
zig build cli -- interactive \
  --model built-in \
  --threads 2 \
  --max-tokens 100 \
  --device cpu \
  --verbose

# For Raspberry Pi Zero (very limited resources)
zig build cli -- inference \
  --model built-in \
  --prompt "Brief status" \
  --threads 1 \
  --max-tokens 25
```

### Step 2: IoT-Specific Use Cases

#### Sensor Data Analysis
```bash
# Analyze sensor readings
zig build cli -- inference \
  --model built-in \
  --prompt "Temperature: 25°C, Humidity: 60%, Pressure: 1013hPa. Analysis:" \
  --max-tokens 100
```

#### Edge AI Monitoring
```bash
# Start lightweight monitoring server
zig build cli -- server \
  --model built-in \
  --port 8080 \
  --host 0.0.0.0 \
  --threads 1

# Query from other devices
curl -X POST http://iot-device:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "System status", "max_tokens": 30}'
```

#### Automated Responses
```bash
# Create automated response system
zig build cli -- inference \
  --model built-in \
  --prompt "Alert: Motion detected at entrance. Recommended action:" \
  --max-tokens 50
```

### Step 3: Performance Optimization for IoT

#### Check System Resources
```bash
# Monitor memory usage during inference
zig build cli -- inference \
  --model built-in \
  --prompt "Test" \
  --verbose

# The verbose output shows memory usage and performance metrics
```

#### Optimize for Battery Life
```bash
# Use minimal settings for battery-powered devices
zig build cli -- inference \
  --model built-in \
  --prompt "Quick check" \
  --threads 1 \
  --max-tokens 20 \
  --device cpu
```

## 🔧 Complete CLI Reference

### Available Commands

#### Core Operations
```bash
# Single inference
zig build cli -- inference --model <MODEL> --prompt "<TEXT>"

# Interactive chat
zig build cli -- interactive --model <MODEL>

# HTTP API server
zig build cli -- server --model <MODEL> --port <PORT>
```

#### Model Management
```bash
# List available models
zig build cli -- list-models

# Download models
zig build cli -- download --download-model <NAME>

# Update models from Git submodules
zig build cli -- update-models
```

### Configuration Options

| Option | Description | Default | Desktop | IoT |
|--------|-------------|---------|---------|-----|
| `--model` | Path to model file or "built-in" | Required | Any | built-in |
| `--prompt` | Input prompt (inference mode) | - | Any | Short |
| `--max-tokens` | Maximum tokens to generate | 100 | 100-500 | 20-100 |
| `--temperature` | Sampling temperature (0.0-1.0) | 0.7 | 0.7 | 0.5 |
| `--threads` | Number of worker threads | 1 | 2-8 | 1-2 |
| `--device` | Device type: auto, cpu, gpu | auto | auto/gpu | cpu |
| `--port` | Server port (server mode) | 8080 | 8080 | 8080 |
| `--host` | Server host (server mode) | 127.0.0.1 | 0.0.0.0 | 0.0.0.0 |
| `--verbose` | Enable verbose output | false | true | false |
| `--models-dir` | Models directory | models | models | models |
| `--force-http` | Force HTTP download | false | false | false |

## 📦 Model Management Guide

### Available Models

#### Built-in Model (Recommended for Testing)
```bash
# No download required - works immediately
zig build cli -- interactive --model built-in
```

#### Downloadable Models
| Model | Size | Use Case | Command |
|-------|------|----------|---------|
| TinyLlama | ~2GB | General chat, desktop | `--download-model tinyllama` |
| GPT-2 Small | ~500MB | Text generation, IoT | `--download-model gpt2-small` |
| DistilBERT | ~250MB | Text analysis, IoT | `--download-model distilbert` |

### Download Methods

#### Method 1: Git Submodules (Fastest)
```bash
# One-time setup
git submodule update --init models

# Download/update models (instant after setup)
zig build cli -- update-models

# Use downloaded model
zig build cli -- interactive --model ./models/tinyllama-1.1b.onnx
```

#### Method 2: Direct HTTP Download
```bash
# List available models
zig build cli -- list-models

# Download specific model
zig build cli -- download --download-model tinyllama --models-dir ./models

# Force HTTP (skip submodule check)
zig build cli -- download --download-model tinyllama --force-http
```

### Performance Comparison
| Method | First Access | Subsequent Access | Storage | Best For |
|--------|-------------|-------------------|---------|----------|
| **Built-in** | Instant | Instant | ~50MB | Testing, IoT |
| **Git Submodule** | 30 seconds | Instant | Shared | Development |
| **HTTP Download** | 5-30 minutes | 5-30 minutes | Full size | Production |

### Storage Requirements
- **Built-in model**: ~50MB (included in binary)
- **Small models**: 250MB - 500MB
- **Large models**: 2GB - 5GB
- **Multiple models**: Use Git LFS for deduplication

## 📡 HTTP API Reference

### Starting the Server

#### Desktop Server
```bash
# Full-featured server for desktop applications
zig build cli -- server \
  --model built-in \
  --port 8080 \
  --host 0.0.0.0 \
  --threads 4 \
  --verbose
```

#### IoT Server
```bash
# Lightweight server for IoT devices
zig build cli -- server \
  --model built-in \
  --port 8080 \
  --host 0.0.0.0 \
  --threads 1
```

### API Endpoints

| Endpoint | Method | Description | Example |
|----------|--------|-------------|---------|
| `/inference` | POST | Run AI inference | See below |
| `/health` | GET | Health check | `curl http://localhost:8080/health` |
| `/models` | GET | List models | `curl http://localhost:8080/models` |

### API Usage Examples

#### Basic Inference
```bash
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### IoT Sensor Analysis
```bash
curl -X POST http://iot-device:8080/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Temperature: 25°C, Status:",
    "max_tokens": 30
  }'
```

#### Health Check
```bash
# Check if server is running
curl http://localhost:8080/health

# Response: {"status": "healthy", "model": "built-in"}
```

## 🧪 Testing & Verification

### Quick Verification
```bash
# Test basic functionality
zig build cli -- inference --model built-in --prompt "Test" --verbose

# Test interactive mode
zig build cli -- interactive --model built-in

# Test server mode
zig build cli -- server --model built-in --port 8080 &
curl http://localhost:8080/health
```

### Comprehensive Testing
```bash
# Run unit tests
zig build test

# Run integration tests
zig build test-integration

# Run performance benchmarks
zig build bench

# Test specific examples
zig build run-simple_inference
zig build run-gpu_demo
```

### IoT Device Testing
```bash
# Test memory constraints
zig build cli -- inference \
  --model built-in \
  --prompt "Memory test" \
  --threads 1 \
  --verbose

# Monitor resource usage
top -p $(pgrep zig-ai)
```

## 📊 Performance Benchmarks

### Desktop Performance
- **Inference Speed**: 50-200 tokens/second
- **Memory Usage**: 100-500MB (depending on model)
- **CPU Usage**: 10-80% (configurable with threads)
- **Startup Time**: < 2 seconds

### IoT Performance
- **Inference Speed**: 10-50 tokens/second
- **Memory Usage**: 50-200MB
- **CPU Usage**: 20-60% (single thread)
- **Power Consumption**: Optimized for battery life

### Scalability
- **Concurrent Requests**: 10-100+ (server mode)
- **Thread Scaling**: Linear up to CPU cores
- **Memory Scaling**: Constant with tensor pooling
- **Model Loading**: < 5 seconds for most models

## 🛠️ Troubleshooting

### Common Issues

#### Build Errors
```bash
# Clear cache and rebuild
rm -rf zig-cache zig-out
zig build

# Check Zig version
zig version  # Should be 0.11+
```

#### Model Loading Issues
```bash
# Verify model exists
ls -la models/

# Test with built-in model first
zig build cli -- inference --model built-in --prompt "Test"

# Check model format
file models/your-model.onnx
```

#### Memory Issues on IoT
```bash
# Reduce memory usage
zig build cli -- inference \
  --model built-in \
  --prompt "Short test" \
  --threads 1 \
  --max-tokens 20

# Check available memory
free -h
```

#### Network Issues
```bash
# Test local server
zig build cli -- server --model built-in --port 8080 --host 127.0.0.1

# Check if port is available
netstat -an | grep 8080
```

### Getting Help
- 📖 **Documentation**: Check `docs/` directory
- 🐛 **Issues**: Report bugs on GitHub
- 💬 **Discussions**: Join community discussions
- 📧 **Support**: Contact maintainers

## 🔒 Security & Privacy

### Privacy Features
- **🔒 Local Processing**: All AI inference runs locally
- **🚫 No Cloud**: Zero data sent to external servers
- **🛡️ Memory Safe**: Zig prevents memory vulnerabilities
- **🔐 Isolated**: Each inference runs in isolated context

### Security Best Practices
- **🔧 Resource Limits**: Configure memory and thread limits
- **🌐 Network Security**: Use HTTPS in production
- **📝 Audit Logging**: Enable verbose mode for monitoring
- **🔄 Regular Updates**: Keep engine and models updated

## 🏗️ Architecture Overview

```
Zig AI Inference Engine
├── 🧠 Core Engine
│   ├── Tensor Operations (SIMD optimized)
│   ├── Memory Management (Arena allocators)
│   └── Thread-Safe Execution
├── 🔌 Interfaces
│   ├── CLI (Unified command-line)
│   ├── HTTP API (RESTful server)
│   └── Library API (Direct integration)
├── 📦 Model Support
│   ├── ONNX Parser
│   ├── Built-in Models
│   └── Git Submodule Integration
├── 🚀 Acceleration
│   ├── CPU (SIMD, Multi-threading)
│   ├── GPU (CUDA, Vulkan)
│   └── IoT Optimizations
└── 🔒 Security
    ├── Local Processing
    ├── Memory Safety
    └── Resource Controls
```

## 📚 Additional Resources

### Documentation
- 📖 **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- 🏗️ **[Architecture Guide](docs/ARCHITECTURE.md)** - Detailed architecture
- 🔧 **[Memory Guide](docs/MEMORY_ALLOCATION_GUIDE.md)** - Memory management
- 🎮 **[GPU Guide](docs/GPU_ARCHITECTURE.md)** - GPU acceleration
- 📦 **[Submodule Setup](docs/SUBMODULE_SETUP.md)** - Model management

### Examples
- 🚀 **Simple Inference**: `examples/simple_inference.zig`
- 🌐 **HTTP Server**: `examples/model_loading.zig`
- 🎮 **GPU Demo**: `examples/gpu_demo.zig`
- 📊 **Computation Graph**: `examples/computation_graph.zig`

### Community
- 🐛 **Issues**: [GitHub Issues](https://github.com/anachary/zig-ai-inference-engine/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/anachary/zig-ai-inference-engine/discussions)
- 📧 **Contact**: [Project Maintainers](mailto:akashnacharya@gmail.com)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/yourusername/zig-ai-inference-engine.git`
3. **Create** a feature branch: `git checkout -b feature/amazing-feature`
4. **Make** your changes and test thoroughly
5. **Commit** your changes: `git commit -m 'Add amazing feature'`
6. **Push** to your branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/anachary/zig-ai-inference-engine.git
cd zig-ai-inference-engine
zig build test  # Run tests
zig build bench # Run benchmarks
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Zig Language](https://ziglang.org/)** - For memory safety and performance
- **ONNX Community** - For standardized model formats
- **Edge AI Community** - For inspiration and feedback
- **Contributors** - For making this project better

---

## 🎯 Perfect For

✅ **Edge AI Applications** - Run AI on resource-constrained devices
✅ **IoT Deployments** - Lightweight inference for embedded systems
✅ **Privacy-Critical Apps** - Local processing, no cloud dependencies
✅ **Production APIs** - Scalable HTTP servers for web applications
✅ **Research & Development** - Fast prototyping with comprehensive APIs
✅ **Educational Use** - Learn AI inference with clear, readable code

**Start building privacy-first, edge-optimized AI applications today!**


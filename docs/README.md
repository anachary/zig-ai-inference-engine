# Documentation

## 📚 Core Documentation

### [`ARCHITECTURE.md`](./ARCHITECTURE.md)
System architecture and design overview

### [`API_REFERENCE.md`](./API_REFERENCE.md)
Complete API documentation and examples

### [`GPU_ARCHITECTURE.md`](./GPU_ARCHITECTURE.md)
GPU acceleration framework details

### [`MEMORY_ALLOCATION_GUIDE.md`](./MEMORY_ALLOCATION_GUIDE.md)
Memory management strategies and patterns

### [`SUBMODULE_SETUP.md`](./SUBMODULE_SETUP.md)
Model management and Git submodule setup

## 🚀 Quick Start

1. **Start Here**: [`../README.md`](../README.md) - Project overview
2. **Architecture**: [`ARCHITECTURE.md`](./ARCHITECTURE.md) - System design
3. **API Reference**: [`API_REFERENCE.md`](./API_REFERENCE.md) - Complete API docs
4. **Examples**: [`../examples/`](../examples/) - Code examples

## 📋 Key Topics

### Core AI Infrastructure
- **Tensors & Operators**: [`API_REFERENCE.md`](./API_REFERENCE.md)
- **System Architecture**: [`ARCHITECTURE.md`](./ARCHITECTURE.md)

### GPU & Performance
- **GPU Acceleration**: [`GPU_ARCHITECTURE.md`](./GPU_ARCHITECTURE.md)
- **Memory Management**: [`MEMORY_ALLOCATION_GUIDE.md`](./MEMORY_ALLOCATION_GUIDE.md)

### Model Management
- **ONNX Support**: [`API_REFERENCE.md`](./API_REFERENCE.md)
- **Git Submodules**: [`SUBMODULE_SETUP.md`](./SUBMODULE_SETUP.md)

## 🔧 Development

### Examples
- **Basic Usage**: [`../examples/simple_inference.zig`](../examples/simple_inference.zig)
- **GPU Demo**: [`../examples/gpu_demo.zig`](../examples/gpu_demo.zig)
- **ONNX Parser**: [`../examples/advanced_onnx_parser.zig`](../examples/advanced_onnx_parser.zig)

### Testing
```bash
zig build test                    # Run all tests
zig build run-advanced_onnx_parser # Test ONNX parser
```

### Build
```bash
zig build                        # Build project
zig build cli -- --help         # CLI help
```

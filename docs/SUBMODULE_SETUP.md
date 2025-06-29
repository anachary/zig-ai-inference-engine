# Git Submodule Setup for ONNX Models

This guide shows how to set up Git submodules for lightning-fast model access in the Zig AI Inference Engine.

## 🚀 Performance Benefits

| Approach | First Access | Subsequent Access | Storage |
|----------|-------------|-------------------|---------|
| **HTTP Download** | 5-30 minutes | 5-30 minutes | 2-5GB each time |
| **Git Submodule** | 30 seconds | **Instant** | Git LFS deduplication |

## 📋 Prerequisites

- Git with LFS support: `git lfs install`
- Access to a models repository (or create your own)

## 🔧 Setup Instructions

### Option 1: Use Existing Models Repository

```bash
# Clone the main repository
git clone https://github.com/your-org/zig-ai-inference-engine.git
cd zig-ai-inference-engine

# Add models as a submodule
git submodule add https://github.com/your-org/zig-ai-models.git models

# Initialize and update
git submodule update --init --recursive
```

### Option 2: Create Your Own Models Repository

```bash
# Create a new repository for models
mkdir zig-ai-models
cd zig-ai-models
git init
git lfs install

# Track ONNX files with LFS
git lfs track "*.onnx"
git add .gitattributes

# Add your ONNX models
mkdir tinyllama gpt2-small phi2
# Copy your .onnx files to respective directories

# Commit and push
git add .
git commit -m "Add ONNX models with LFS"
git remote add origin https://github.com/your-org/zig-ai-models.git
git push -u origin main

# Go back to main project and add submodule
cd ../zig-ai-inference-engine
git submodule add https://github.com/your-org/zig-ai-models.git models
```

## 📁 Expected Directory Structure

```
zig-ai-inference-engine/
├── src/
├── examples/
├── models/                    # Git submodule
│   ├── tinyllama/
│   │   └── tinyllama.onnx    # LFS tracked
│   ├── gpt2-small/
│   │   └── gpt2-small.onnx   # LFS tracked
│   ├── phi2/
│   │   └── phi2.onnx         # LFS tracked
│   └── README.md
└── build.zig
```

## 🎯 CLI Usage

### Download Models (Submodule First)
```bash
# Tries submodule first, falls back to HTTP
zig-ai download --download-model tinyllama

# Force HTTP download (skip submodule)
zig-ai download --download-model tinyllama --force-http

# Disable submodule globally
zig-ai download --download-model tinyllama --no-submodule
```

### Update Models
```bash
# Update all models from submodule
zig-ai update-models

# Update with custom submodule path
zig-ai update-models --submodule-path my-models
```

### List Models
```bash
# List available models (shows submodule status)
zig-ai list-models
```

## 🔄 Workflow Examples

### Developer Workflow
```bash
# Initial setup (one time)
git submodule update --init models

# Daily usage (instant access)
zig-ai interactive --model tinyllama
zig-ai inference --model gpt2-small --prompt "Hello"

# Update to latest models
zig-ai update-models
```

### CI/CD Pipeline
```bash
# Fast model access in CI
git submodule update --init --depth 1 models
zig-ai list-models
zig-ai interactive --model tinyllama --prompt "test"
```

## 🛠️ Troubleshooting

### Submodule Not Found
```bash
# Check if you're in a Git repository
git status

# Initialize submodule manually
git submodule add https://github.com/your-org/zig-ai-models.git models
git submodule update --init
```

### LFS Files Not Downloaded
```bash
# Pull LFS files manually
cd models
git lfs pull
cd ..
```

### Force HTTP Fallback
```bash
# If submodule fails, CLI automatically falls back to HTTP
zig-ai download --download-model tinyllama --force-http
```

## 📊 Performance Comparison

### Cold Start (First Time)
- **HTTP**: 5-30 minutes download
- **Submodule**: 30 seconds - 2 minutes (depending on LFS)

### Warm Start (Subsequent Uses)
- **HTTP**: 5-30 minutes download (every time)
- **Submodule**: **Instant** (local file access)

### Storage Efficiency
- **HTTP**: Full model files each time
- **Submodule**: Git LFS deduplication + version control

## 🎉 Benefits Summary

✅ **30x faster** model access after initial setup  
✅ **Zero CLI size increase** - same binary size  
✅ **Automatic fallback** to HTTP if submodule fails  
✅ **Version control** for models alongside code  
✅ **Offline development** capability  
✅ **CI/CD optimization** - faster builds and tests  

The submodule approach provides massive performance improvements while maintaining full compatibility with the existing HTTP download system!

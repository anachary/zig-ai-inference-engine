# Installation Guide

Complete installation instructions for the Zig AI Platform across different operating systems and deployment scenarios.

## 🎯 Quick Installation

### Prerequisites
- **Zig**: Version 0.11.x or later
- **Git**: For cloning the repository
- **4GB RAM**: Minimum for basic operation
- **10GB Storage**: For platform and models

### One-Command Install
```bash
# Clone and build
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform
zig build

# Verify installation
zig build test
```

## 🖥️ Operating System Specific

### 🐧 **Linux (Ubuntu/Debian)**
```bash
# Install Zig
wget https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz
tar -xf zig-linux-x86_64-0.11.0.tar.xz
sudo mv zig-linux-x86_64-0.11.0 /opt/zig
echo 'export PATH="/opt/zig:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install dependencies
sudo apt update
sudo apt install git build-essential

# Clone and build platform
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform
zig build
```

### 🍎 **macOS**
```bash
# Install Zig via Homebrew
brew install zig

# Or download directly
curl -O https://ziglang.org/download/0.11.0/zig-macos-x86_64-0.11.0.tar.xz
tar -xf zig-macos-x86_64-0.11.0.tar.xz
sudo mv zig-macos-x86_64-0.11.0 /usr/local/zig
echo 'export PATH="/usr/local/zig:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Clone and build platform
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform
zig build
```

### 🪟 **Windows**
```powershell
# Download Zig
Invoke-WebRequest -Uri "https://ziglang.org/download/0.11.0/zig-windows-x86_64-0.11.0.zip" -OutFile "zig.zip"
Expand-Archive -Path "zig.zip" -DestinationPath "C:\zig"
$env:PATH += ";C:\zig\zig-windows-x86_64-0.11.0"

# Clone and build platform
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform
zig build
```

## 🐳 Container Installation

### **Docker**
```bash
# Build container
docker build -t zig-ai-platform .

# Run container
docker run -p 8080:8080 zig-ai-platform

# With volume for models
docker run -p 8080:8080 -v $(pwd)/models:/app/models zig-ai-platform
```

### **Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'
services:
  zig-ai:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
    environment:
      - ZIG_AI_PORT=8080
```

```bash
# Start with compose
docker-compose up -d
```

## ☁️ Cloud Installation

### **Azure Container Instances**
```bash
# Create resource group
az group create --name zig-ai-rg --location eastus

# Deploy container
az container create \
  --resource-group zig-ai-rg \
  --name zig-ai-platform \
  --image zig-ai-platform:latest \
  --ports 8080 \
  --cpu 2 \
  --memory 4
```

### **Kubernetes**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zig-ai-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: zig-ai-platform
  template:
    metadata:
      labels:
        app: zig-ai-platform
    spec:
      containers:
      - name: zig-ai
        image: zig-ai-platform:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

```bash
# Deploy to Kubernetes
kubectl apply -f deployment.yaml
```

## 📱 Edge/IoT Installation

### **Raspberry Pi**
```bash
# Install Zig for ARM64
wget https://ziglang.org/download/0.11.0/zig-linux-aarch64-0.11.0.tar.xz
tar -xf zig-linux-aarch64-0.11.0.tar.xz
sudo mv zig-linux-aarch64-0.11.0 /opt/zig
echo 'export PATH="/opt/zig:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Build for ARM64
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform
zig build -Dtarget=aarch64-linux
```

### **Embedded Linux**
```bash
# Cross-compile for embedded target
zig build -Dtarget=arm-linux-musleabihf -Doptimize=ReleaseFast

# Copy to embedded device
scp zig-out/bin/zig-ai-platform user@device:/usr/local/bin/
```

## 🔧 Development Installation

### **Development Dependencies**
```bash
# Install additional development tools
# For testing
sudo apt install valgrind  # Linux memory debugging

# For documentation
npm install -g markdownlint-cli

# For benchmarking
sudo apt install hyperfine
```

### **IDE Setup**

#### **VS Code**
```json
// .vscode/settings.json
{
    "zig.path": "/opt/zig/zig",
    "zig.zls.path": "/usr/local/bin/zls",
    "files.associations": {
        "*.zig": "zig"
    }
}
```

#### **Vim/Neovim**
```vim
" Add to .vimrc or init.vim
Plug 'ziglang/zig.vim'
```

## ✅ Verification

### **Test Installation**
```bash
# Check Zig version
zig version

# Build platform
cd zig-ai-platform
zig build

# Run tests
zig build test

# Start server
zig build run-server

# Test API (in another terminal)
curl http://localhost:8080/api/v1/health
```

### **Expected Output**
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "components": {
    "tensor-core": "ready",
    "onnx-parser": "ready",
    "inference-engine": "ready",
    "model-server": "ready"
  }
}
```

## 🔧 Configuration

### **Environment Variables**
```bash
# Server configuration
export ZIG_AI_PORT=8080
export ZIG_AI_HOST=0.0.0.0
export ZIG_AI_LOG_LEVEL=info

# Model configuration
export ZIG_AI_MODEL_PATH=/path/to/models
export ZIG_AI_MAX_MODEL_SIZE=10GB

# Performance tuning
export ZIG_AI_WORKER_THREADS=4
export ZIG_AI_MEMORY_LIMIT=8GB
```

### **Configuration File**
```yaml
# config.yaml
server:
  port: 8080
  host: "0.0.0.0"
  log_level: "info"

models:
  path: "./models"
  max_size: "10GB"
  cache_size: "2GB"

performance:
  worker_threads: 4
  memory_limit: "8GB"
  enable_simd: true
```

## 🆘 Troubleshooting

### **Common Issues**

#### **Zig Not Found**
```bash
# Check PATH
echo $PATH
which zig

# Add to PATH
export PATH="/opt/zig:$PATH"
```

#### **Build Failures**
```bash
# Clean build cache
rm -rf zig-cache zig-out

# Rebuild
zig build
```

#### **Permission Errors**
```bash
# Fix permissions
chmod +x zig-out/bin/zig-ai-platform

# Or run with sudo if needed
sudo zig build install
```

#### **Memory Issues**
```bash
# Increase available memory
export ZIG_AI_MEMORY_LIMIT=16GB

# Or reduce model size
export ZIG_AI_MAX_MODEL_SIZE=2GB
```

### **Getting Help**
- **Documentation**: [Troubleshooting Guide](how-to-guides/troubleshooting.md)
- **GitHub Issues**: Report installation problems
- **Discussions**: Ask questions in GitHub Discussions
- **FAQ**: Check [Frequently Asked Questions](faq.md)

## 🚀 Next Steps

After successful installation:

1. **[Getting Started](getting-started.md)** - Quick introduction
2. **[IoT Tutorial](tutorials/iot-quick-start.md)** - First deployment
3. **[LLM Tutorial](tutorials/llm-quick-start.md)** - Cloud deployment
4. **[Examples](../examples/)** - Explore use cases

---

**Installation complete!** Ready to start with [Getting Started](getting-started.md)?

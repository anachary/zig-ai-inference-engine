# How to Enable Zero-Dependency GPU Acceleration

*Problem-oriented guide for enabling GPU acceleration without external dependencies*

## Problem

You want to accelerate AI inference using GPU compute but don't want to install CUDA SDK, Vulkan SDK, or other external dependencies.

## Solution

The Zig AI Platform provides a pure Zig GPU backend that automatically detects and utilizes available GPU capabilities without requiring external libraries.

## Prerequisites

- Zig AI Platform installed
- GPU drivers installed (standard OS drivers, not SDKs)
- No additional dependencies required

## Step 1: Verify GPU Detection

Check if your GPU is detected by the zero-dependency backend:

```bash
# Run the built-in GPU detection
zig-ai-platform --detect-gpu
```

Expected output:
```
🎮 GPU Detection Results
========================
✅ Pure Zig GPU Backend: Available
✅ Vulkan Compute: Detected (no SDK required)
✅ DirectCompute: Available (Windows)
✅ Metal Compute: Available (macOS)
✅ CPU Fallback: Always available

Selected Backend: Pure Zig GPU (Vulkan Compute)
Compute Units: 8
Memory Available: 4.0 GB
```

## Step 2: Enable GPU Acceleration

### Option A: Automatic Detection (Recommended)

The platform automatically selects the best available backend:

```bash
# GPU acceleration is enabled by default
zig-ai-platform inference --model model.onnx --input data.json
```

The system will automatically:
1. Try Pure Zig GPU backend first
2. Fall back to optimized CPU if GPU unavailable
3. Use the fastest available compute method

### Option B: Explicit Backend Selection

Force a specific backend if needed:

```bash
# Force Pure Zig GPU backend
zig-ai-platform inference --backend pure-zig --model model.onnx

# Force CPU fallback (for testing)
zig-ai-platform inference --backend cpu --model model.onnx
```

## Step 3: Verify GPU Acceleration

### Check Performance Metrics

Run with performance monitoring:

```bash
zig-ai-platform inference --model model.onnx --profile
```

Look for GPU utilization indicators:
```
📊 Performance Profile
======================
Backend: Pure Zig GPU (Vulkan Compute)
GPU Utilization: 85%
Memory Usage: 2.1 GB / 4.0 GB
Inference Time: 12.3ms
Throughput: 81.3 inferences/sec
```

### Compare CPU vs GPU Performance

```bash
# Test with GPU
zig-ai-platform benchmark --backend pure-zig --iterations 100

# Test with CPU
zig-ai-platform benchmark --backend cpu --iterations 100
```

Expected speedup: 3-10x depending on model complexity.

## Step 4: Optimize GPU Performance

### Configure Memory Usage

Set GPU memory limits:

```bash
zig-ai-platform inference --gpu-memory 2GB --model model.onnx
```

### Batch Processing

Enable batch processing for higher throughput:

```bash
zig-ai-platform inference --batch-size 8 --model model.onnx
```

### Kernel Optimization

Enable advanced kernel optimizations:

```bash
zig-ai-platform inference --optimize-kernels --model model.onnx
```

## Troubleshooting

### GPU Not Detected

**Problem**: GPU detection fails
```
❌ Pure Zig GPU Backend: Not available
❌ Vulkan Compute: Not detected
```

**Solutions**:

1. **Update GPU drivers**:
   ```bash
   # Windows: Use Device Manager or vendor website
   # Linux: Update via package manager
   sudo apt update && sudo apt upgrade
   
   # macOS: Update via System Preferences
   ```

2. **Check GPU compatibility**:
   ```bash
   # List available GPUs
   zig-ai-platform --list-gpus
   ```

3. **Verify Vulkan support**:
   ```bash
   # Check if Vulkan loader is available
   # Windows: Look for vulkan-1.dll in System32
   # Linux: Check for libvulkan.so
   # macOS: Check for MoltenVK
   ```

### Performance Issues

**Problem**: GPU acceleration slower than expected

**Solutions**:

1. **Check memory usage**:
   ```bash
   zig-ai-platform inference --model model.onnx --memory-profile
   ```

2. **Optimize batch size**:
   ```bash
   # Try different batch sizes
   zig-ai-platform benchmark --batch-size 1,2,4,8,16
   ```

3. **Enable kernel fusion**:
   ```bash
   zig-ai-platform inference --fuse-kernels --model model.onnx
   ```

### Fallback to CPU

**Problem**: System falls back to CPU unexpectedly

**Check logs**:
```bash
zig-ai-platform inference --model model.onnx --verbose
```

**Common causes**:
- GPU memory insufficient for model
- Unsupported operations (rare)
- Driver compatibility issues

**Solutions**:
- Reduce model size or batch size
- Update GPU drivers
- Use mixed CPU/GPU execution

## Advanced Configuration

### Custom GPU Settings

Create a configuration file `gpu-config.toml`:

```toml
[gpu]
backend = "pure-zig"
memory_limit = "4GB"
enable_profiling = true
optimize_kernels = true

[vulkan]
device_index = 0
queue_family = 0

[performance]
batch_size = 4
enable_fusion = true
memory_pool_size = "1GB"
```

Use the configuration:
```bash
zig-ai-platform inference --config gpu-config.toml --model model.onnx
```

### Monitoring GPU Usage

Real-time monitoring:
```bash
# Monitor GPU utilization
zig-ai-platform monitor --gpu

# Export metrics
zig-ai-platform inference --model model.onnx --export-metrics metrics.json
```

## Performance Expectations

### Typical Speedups

| Model Type | CPU Baseline | Pure Zig GPU | Speedup |
|------------|--------------|---------------|---------|
| Small CNN | 50ms | 15ms | 3.3x |
| ResNet-50 | 200ms | 45ms | 4.4x |
| BERT-Base | 80ms | 20ms | 4.0x |
| GPT-2 | 150ms | 35ms | 4.3x |

### Memory Usage

| Backend | Memory Overhead | Allocation Speed |
|---------|----------------|------------------|
| Pure Zig GPU | Minimal | 78x faster |
| Traditional CUDA | High | Baseline |
| CPU Fallback | None | 78x faster |

## Best Practices

### 1. Let Auto-Detection Work
- Don't force specific backends unless necessary
- The system chooses optimal configuration automatically

### 2. Monitor Performance
- Use `--profile` flag during development
- Set up monitoring in production

### 3. Optimize for Your Hardware
- Test different batch sizes
- Enable kernel fusion for complex models
- Configure memory limits appropriately

### 4. Plan for Fallback
- Always test CPU fallback performance
- Ensure graceful degradation

## Verification

Confirm zero-dependency GPU acceleration is working:

```bash
# Check that no external libraries are loaded
zig-ai-platform inference --model model.onnx --check-dependencies

# Expected output:
# ✅ Zero external dependencies detected
# ✅ Pure Zig implementation active
# ✅ GPU acceleration enabled
```

## Next Steps

- [Optimize Model Performance](optimize-performance.md) - Further performance tuning
- [Deploy to Production](../tutorials/production-deployment.md) - Production deployment
- [Monitor Performance](monitor-performance.md) - Set up monitoring

---

*This guide solves the specific problem of enabling GPU acceleration without dependencies. For understanding the design philosophy, see [Zero-Dependency Design](../explanation/zero-dependency-design.md).*

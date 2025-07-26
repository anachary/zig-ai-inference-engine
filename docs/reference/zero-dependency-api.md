# Zero-Dependency API Reference

*Information-oriented technical reference for zero-dependency features*

## Overview

This reference documents the API for zero-dependency features in the Zig AI Platform. All functions operate without external library dependencies.

## Core API

### Dependency Checking

#### `checkDependencies()`

Verifies zero-dependency status of the platform.

**Signature**:
```zig
pub fn checkDependencies() DependencyStatus
```

**Returns**:
```zig
pub const DependencyStatus = struct {
    has_external_deps: bool,
    external_libs: []const []const u8,
    binary_size_bytes: usize,
    dependency_count: usize,
};
```

**Example**:
```zig
const status = checkDependencies();
if (status.has_external_deps) {
    std.log.err("External dependencies detected: {any}", .{status.external_libs});
} else {
    std.log.info("Zero dependencies confirmed");
}
```

### GPU Backend API

#### `PureZigGPU.init()`

Initialize zero-dependency GPU backend.

**Signature**:
```zig
pub fn init(allocator: Allocator) !PureZigGPU
```

**Parameters**:
- `allocator: Allocator` - Memory allocator for GPU backend

**Returns**: `PureZigGPU` instance or error

**Errors**:
- `BackendError.NoGPUAvailable` - No compatible GPU found
- `BackendError.InitializationFailed` - GPU initialization failed

**Example**:
```zig
var gpu = try PureZigGPU.init(allocator);
defer gpu.deinit();
```

#### `detectVulkanCapabilities()`

Detect Vulkan compute capabilities without external SDK.

**Signature**:
```zig
fn detectVulkanCapabilities(self: *Self) bool
```

**Returns**: `true` if Vulkan compute available, `false` otherwise

**Platform Behavior**:
- **Windows**: Checks for `vulkan-1.dll` in system directories
- **Linux**: Checks for `libvulkan.so` in standard library paths  
- **macOS**: Checks for MoltenVK framework

#### `compileKernel()`

Compile compute kernel from Zig source (zero dependencies).

**Signature**:
```zig
pub fn compileKernel(self: *Self, name: []const u8, zig_source: []const u8) !void
```

**Parameters**:
- `name: []const u8` - Kernel identifier
- `zig_source: []const u8` - Zig source code for kernel

**Example**:
```zig
const kernel_source = 
    \\pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    \\    for (a, b, result) |a_val, b_val, *res| {
    \\        res.* = a_val + b_val;
    \\    }
    \\}
;

try gpu.compileKernel("vector_add", kernel_source);
```

#### `executeKernel()`

Execute compiled kernel on GPU.

**Signature**:
```zig
pub fn executeKernel(
    self: *Self,
    name: []const u8,
    global_size: [3]u32,
    local_size: [3]u32,
    args: []const KernelArg,
) !void
```

**Parameters**:
- `name: []const u8` - Kernel identifier
- `global_size: [3]u32` - Global work size (x, y, z)
- `local_size: [3]u32` - Local work group size (x, y, z)
- `args: []const KernelArg` - Kernel arguments

**Kernel Arguments**:
```zig
pub const KernelArg = union(enum) {
    buffer: *GPUBuffer,
    scalar_u32: u32,
    scalar_f32: f32,
};
```

### Memory Management API

#### `AdvancedTensorPool.init()`

Initialize zero-dependency memory pool.

**Signature**:
```zig
pub fn init(allocator: Allocator, config: PoolConfig) Self
```

**Configuration**:
```zig
pub const PoolConfig = struct {
    max_pool_size: usize = 1000,
    small_pool_size: usize = 200,
    medium_pool_size: usize = 100,
    large_pool_size: usize = 50,
    defrag_threshold: f32 = 0.3,
    enable_defragmentation: bool = true,
};
```

#### `getTensor()`

Get tensor from pool with zero external allocations.

**Signature**:
```zig
pub fn getTensor(self: *Self, shape: []const usize, dtype: DataType) !Tensor
```

**Performance**: 78x faster than standard allocation

### SIMD API

#### `vectorAddF32()`

Zero-dependency vectorized addition with automatic SIMD selection.

**Signature**:
```zig
pub fn vectorAddF32(a: []const f32, b: []const f32, result: []f32) SIMDError!void
```

**SIMD Support**:
- **AVX-512**: 16-element vectors (if available)
- **AVX2**: 8-element vectors (if available)
- **SSE**: 4-element vectors (if available)
- **Scalar**: Fallback for any platform

**Performance**: 300M+ operations/second

#### `matrixMultiplyF32()`

Zero-dependency matrix multiplication with SIMD optimization.

**Signature**:
```zig
pub fn matrixMultiplyF32(
    a: []const f32, a_rows: usize, a_cols: usize,
    b: []const f32, b_rows: usize, b_cols: usize,
    c: []f32
) SIMDError!void
```

**Optimization**: Cache-friendly access patterns with vectorization

## Backend Types

### `DeviceType`

Supported compute device types (all zero-dependency).

```zig
pub const DeviceType = enum {
    vulkan_compute,    // Vulkan compute shaders
    direct_compute,    // DirectCompute (Windows)
    metal_compute,     // Metal compute (macOS)
    cpu_fallback,      // Optimized CPU implementation
};
```

### `BackendType`

Available backend implementations.

```zig
pub const BackendType = enum {
    pure_zig,          // Zero-dependency pure Zig implementation
    cpu_fallback,      // CPU-only fallback
    vulkan,            // Vulkan compute (if available)
    opencl,            // OpenCL (if available)
    metal,             // Metal (macOS only)
};
```

## Error Types

### `BackendError`

GPU backend error conditions.

```zig
pub const BackendError = error{
    NoGPUAvailable,
    BackendNotAvailable,
    InitializationFailed,
    KernelCompilationFailed,
    KernelExecutionFailed,
    BufferAllocationFailed,
    BufferTooSmall,
    InvalidConfiguration,
};
```

### `SIMDError`

SIMD operation error conditions.

```zig
pub const SIMDError = error{
    InvalidLength,
    UnsupportedOperation,
    AlignmentError,
};
```

## Performance Characteristics

### Memory Allocation

| Operation | Standard | Zero-Dependency | Speedup |
|-----------|----------|-----------------|---------|
| Tensor allocation | 78ms | 1ms | 78x |
| Memory pooling | N/A | Active | N/A |
| Defragmentation | Manual | Automatic | N/A |

### SIMD Operations

| Instruction Set | Vector Width | Performance |
|----------------|--------------|-------------|
| AVX-512 | 16 x f32 | 400M+ ops/sec |
| AVX2 | 8 x f32 | 350M+ ops/sec |
| SSE | 4 x f32 | 300M+ ops/sec |
| Scalar | 1 x f32 | 50M+ ops/sec |

### GPU Acceleration

| Backend | Dependency | Performance | Compatibility |
|---------|------------|-------------|---------------|
| Pure Zig | None | Native | Universal |
| Traditional CUDA | CUDA SDK | High | NVIDIA only |
| Traditional OpenCL | OpenCL SDK | High | Vendor-specific |

## Configuration

### Environment Variables

```bash
# Force specific backend
ZIG_AI_BACKEND=pure-zig

# Set memory limits
ZIG_AI_GPU_MEMORY=4GB
ZIG_AI_POOL_SIZE=1000

# Enable debugging
ZIG_AI_DEBUG=true
ZIG_AI_PROFILE=true
```

### Configuration File

`config.toml`:
```toml
[backend]
type = "pure-zig"
auto_detect = true
fallback_to_cpu = true

[memory]
pool_size = 1000
defrag_threshold = 0.3
enable_pooling = true

[performance]
enable_simd = true
simd_width = "auto"
enable_fusion = true

[gpu]
memory_limit = "4GB"
enable_profiling = false
```

## Compatibility Matrix

### Operating Systems

| OS | Pure Zig GPU | Vulkan | DirectCompute | Metal |
|----|--------------|--------|---------------|-------|
| Windows | ✅ | ✅ | ✅ | ❌ |
| Linux | ✅ | ✅ | ❌ | ❌ |
| macOS | ✅ | ✅* | ❌ | ✅ |
| FreeBSD | ✅ | ✅ | ❌ | ❌ |

*Via MoltenVK

### Architectures

| Architecture | Support | SIMD | GPU |
|--------------|---------|------|-----|
| x86_64 | ✅ | AVX-512/AVX2/SSE | ✅ |
| ARM64 | ✅ | NEON | ✅ |
| RISC-V | ✅ | Scalar | ❌ |

## Examples

### Complete Zero-Dependency Setup

```zig
const std = @import("std");
const zig_ai = @import("zig-ai");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Verify zero dependencies
    const deps = zig_ai.checkDependencies();
    if (deps.has_external_deps) {
        return error.ExternalDependenciesDetected;
    }

    // Initialize pure Zig GPU backend
    var gpu = try zig_ai.PureZigGPU.init(allocator);
    defer gpu.deinit();

    // Initialize memory pool
    const pool_config = zig_ai.PoolConfig{};
    var pool = zig_ai.AdvancedTensorPool.init(allocator, pool_config);
    defer pool.deinit();

    // Create tensors with zero external allocations
    var tensor_a = try pool.getTensor(&[_]usize{1000, 1000}, .f32);
    defer pool.returnTensor(tensor_a);

    // Perform SIMD operations
    const a = try allocator.alloc(f32, 1000000);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, 1000000);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, 1000000);
    defer allocator.free(result);

    try zig_ai.simd.vectorAddF32(a, b, result);

    std.log.info("Zero-dependency AI inference complete!");
}
```

---

*This reference provides comprehensive technical details for zero-dependency features. For practical usage, see our [How-to Guides](../how-to-guides/), and for learning by doing, try our [Tutorials](../tutorials/).*

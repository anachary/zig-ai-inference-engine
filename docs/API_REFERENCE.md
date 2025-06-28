# API Reference - Zig AI Interface Engine

## Overview

This document provides comprehensive API documentation for the Zig AI Interface Engine, covering all major components and their usage patterns.

**Phase 2 Status: ✅ Complete API Implementation**

## Core APIs

### Tensor API (`src/core/tensor.zig`)

#### Tensor Creation

```zig
// Create tensor with shape and data type
var tensor = try Tensor.init(allocator, &[_]usize{2, 3}, .f32);
defer tensor.deinit();

// Create tensor from data
const data = [_]f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
var tensor = try Tensor.fromSlice(allocator, &data, &[_]usize{2, 3});
```

#### Tensor Operations

```zig
// Element access
const value = tensor.get(&[_]usize{0, 1});
try tensor.set(&[_]usize{0, 1}, 42.0);

// Shape manipulation
const reshaped = try tensor.reshape(&[_]usize{3, 2});
const transposed = try tensor.transpose();

// Mathematical operations
const sum = try tensor.add(other_tensor);
const product = try tensor.mul(other_tensor);
```

### Engine API (`src/engine/engine.zig`)

#### Engine Initialization

```zig
// Initialize AI engine with profiling
var engine = try AIEngine.init(allocator, true);
defer engine.deinit();

// Get engine capabilities
const caps = engine.getCapabilities();
std.log.info("SIMD support: {}", .{caps.simd_support});
```

#### Operator Execution

```zig
// Execute operator
const inputs = [_]Tensor{input1, input2};
const outputs = try engine.executeOperator("Add", inputs[0..]);
defer {
    for (outputs) |output| {
        var mutable_output = output;
        mutable_output.deinit();
    }
    allocator.free(outputs);
}
```

### GPU API (`src/gpu/mod.zig`)

#### GPU Context Management

```zig
// Initialize GPU context with automatic device selection
var gpu_context = try GPUContext.init(allocator);
defer gpu_context.deinit();

// Get device information
const device_info = gpu_context.getDeviceInfo();
std.log.info("Device: {s} ({s})", .{ device_info.name, @tagName(device_info.device_type) });
```

#### Memory Management

```zig
// Allocate GPU buffer
const memory_type = gpu_context.getRecommendedMemoryType(true, false);
var buffer = try gpu_context.allocateBuffer(1024, memory_type);
defer gpu_context.freeBuffer(buffer) catch {};

// Map buffer for CPU access
const ptr = try buffer.map();
defer buffer.unmap();
```

#### Kernel Execution

```zig
// Execute GPU kernel
const result = try gpu_context.executeOperator("Add", &[_]GPUBuffer{ input1, input2 }, &[_]GPUBuffer{output});
```

### HTTP Server API (`src/network/server.zig`)

#### Server Setup

```zig
// Initialize HTTP server
var server = try HTTPServer.init(allocator, "127.0.0.1", 8080);
defer server.deinit();

// Start server
try server.start();
```

#### Request Handling

```zig
// Handle inference request
const InferenceRequest = struct {
    model_name: []const u8,
    input_data: []f32,
    input_shape: []usize,
};

const InferenceResponse = struct {
    output_data: []f32,
    output_shape: []usize,
    inference_time_ms: f64,
};
```

### ONNX API (`src/formats/onnx/parser.zig`)

#### Model Loading

```zig
// Load ONNX model
var model = try ONNXModel.loadFromFile(allocator, "model.onnx");
defer model.deinit();

// Get model information
const info = model.getModelInfo();
std.log.info("Model: {s}, Version: {d}", .{ info.name, info.version });
```

#### Graph Access

```zig
// Access computation graph
const graph = model.getGraph();
std.log.info("Nodes: {d}, Inputs: {d}, Outputs: {d}", .{ 
    graph.nodes.len, graph.inputs.len, graph.outputs.len 
});
```

## Specialized APIs

### IoT Optimization API

#### IoT Context Creation

```zig
// Create IoT-optimized GPU context
var iot_context = try createIoTContext(allocator);
defer iot_context.deinit();

// Check IoT suitability
if (iot_context.device.isIoTSuitable()) {
    std.log.info("Ready for IoT deployment");
}
```

#### Memory Constraints

```zig
// Set memory limits for IoT devices
const memory_limit = 512 * 1024 * 1024; // 512MB
try iot_context.setMemoryLimit(memory_limit);

// Enable power-saving mode
try iot_context.enablePowerSaving(true);
```

### Security API

#### Secure Context

```zig
// Create security-optimized context
var secure_context = try createSecurityContext(allocator);
defer secure_context.deinit();

// Enable memory isolation
try secure_context.enableMemoryIsolation(true);
```

#### Audit Logging

```zig
// Enable audit logging
try secure_context.enableAuditLogging(true);

// Get audit trail
const audit_log = secure_context.getAuditLog();
for (audit_log) |entry| {
    std.log.info("Operation: {s}, Time: {d}", .{ entry.operation, entry.timestamp });
}
```

## Error Handling

### Common Error Types

```zig
// Tensor errors
const TensorError = error{
    InvalidShape,
    ShapeMismatch,
    IndexOutOfBounds,
    UnsupportedDataType,
    OutOfMemory,
};

// GPU errors
const GPUError = error{
    NoDevicesFound,
    DeviceInitializationFailed,
    UnsupportedDevice,
    InsufficientMemory,
    DeviceNotAvailable,
};

// Engine errors
const EngineError = error{
    OperatorNotFound,
    InvalidInput,
    ExecutionFailed,
    UnsupportedOperation,
};
```

### Error Handling Patterns

```zig
// Graceful error handling with fallback
const result = gpu_context.executeOperator("Add", inputs, outputs) catch |err| switch (err) {
    GPUError.DeviceNotAvailable => {
        std.log.warn("GPU unavailable, falling back to CPU");
        return cpu_context.executeOperator("Add", inputs, outputs);
    },
    else => return err,
};
```

## Performance Monitoring

### Profiling API

```zig
// Enable profiling
var engine = try AIEngine.init(allocator, true);

// Get performance metrics
const metrics = engine.getPerformanceMetrics();
std.log.info("Total operations: {d}", .{metrics.total_operations});
std.log.info("Average execution time: {d}ms", .{metrics.avg_execution_time});
```

### Memory Monitoring

```zig
// Get memory statistics
const memory_stats = gpu_context.getMemoryStats();
std.log.info("Total allocated: {d}MB", .{memory_stats.total_allocated / (1024 * 1024)});
std.log.info("Peak usage: {d}MB", .{memory_stats.peak_usage / (1024 * 1024)});
```

## Configuration

### Engine Configuration

```zig
const EngineConfig = struct {
    enable_simd: bool = true,
    enable_profiling: bool = false,
    memory_pool_size: usize = 64 * 1024 * 1024, // 64MB
    max_threads: u32 = 0, // Auto-detect
};

var config = EngineConfig{
    .enable_profiling = true,
    .max_threads = 4,
};

var engine = try AIEngine.initWithConfig(allocator, config);
```

### GPU Configuration

```zig
const GPUConfig = struct {
    preferred_device_type: DeviceType = .cpu,
    memory_limit: ?usize = null,
    enable_power_saving: bool = false,
    enable_security_mode: bool = false,
};

var gpu_config = GPUConfig{
    .preferred_device_type = .vulkan,
    .memory_limit = 1024 * 1024 * 1024, // 1GB
    .enable_power_saving = true,
};

var gpu_context = try GPUContext.initWithConfig(allocator, gpu_config);
```

## Best Practices

### Memory Management

1. **Always use defer for cleanup**:
```zig
var tensor = try Tensor.init(allocator, shape, .f32);
defer tensor.deinit();
```

2. **Use memory pools for frequent allocations**:
```zig
var pool = try MemoryPool.init(allocator, 1024 * 1024);
defer pool.deinit();
```

3. **Monitor memory usage in production**:
```zig
const stats = context.getMemoryStats();
if (stats.peak_usage > memory_limit) {
    std.log.warn("Memory usage approaching limit");
}
```

### Error Handling

1. **Always handle GPU fallback**:
```zig
const result = gpu_operation() catch |err| switch (err) {
    GPUError.DeviceNotAvailable => cpu_fallback(),
    else => return err,
};
```

2. **Use appropriate error types**:
```zig
fn validateTensorShape(shape: []const usize) TensorError!void {
    if (shape.len == 0) return TensorError.InvalidShape;
}
```

### Performance

1. **Enable SIMD when available**:
```zig
var engine = try AIEngine.init(allocator, true);
const caps = engine.getCapabilities();
if (caps.simd_support) {
    std.log.info("SIMD acceleration enabled");
}
```

2. **Use appropriate data types**:
```zig
// Use f16 for memory-constrained environments
var tensor = try Tensor.init(allocator, shape, .f16);

// Use i8 for quantized models
var quantized = try Tensor.init(allocator, shape, .i8);
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `simple_inference.zig` - Basic tensor operations
- `model_loading.zig` - ONNX model handling
- `gpu_demo.zig` - GPU acceleration
- `phase2_complete_demo.zig` - Full system demonstration

## Support

For additional documentation and support:

- **Architecture Guide**: `docs/ARCHITECTURE.md`
- **GPU Architecture**: `docs/GPU_ARCHITECTURE.md`
- **Memory Management**: `docs/MEMORY_ALLOCATION_GUIDE.md`
- **Implementation Guide**: `docs/IMPLEMENTATION_GUIDE.md`

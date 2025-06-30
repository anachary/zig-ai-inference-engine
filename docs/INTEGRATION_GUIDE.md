# Integration Guide: Zig AI Ecosystem

## Overview

This guide explains how the modular components of the Zig AI Ecosystem work together, their dependency relationships, and best practices for using them individually or as a unified platform.

## 🔗 Component Dependencies

### Dependency Graph

```
zig-ai-platform (Orchestrator)
├── zig-model-server (HTTP API & CLI)
│   ├── zig-inference-engine (Neural Network Execution)
│   │   └── zig-tensor-core (Tensor Operations)
│   ├── zig-onnx-parser (Model Parsing)
│   │   └── zig-tensor-core (Tensor Operations)
│   └── zig-tensor-core (Tensor Operations)
└── common/interfaces (Shared Contracts)
```

### Dependency Rules

1. **No Circular Dependencies**: Clean acyclic dependency graph
2. **Interface-Based**: Components depend on interfaces, not implementations
3. **Minimal Dependencies**: Each component only depends on what it needs
4. **Version Compatibility**: Semantic versioning ensures compatibility

## 🧮 Using zig-tensor-core Independently

### When to Use Standalone
- Building custom ML frameworks
- Scientific computing applications
- High-performance numerical computing
- Educational tensor operation learning

### Basic Usage
```zig
const std = @import("std");
const tensor_core = @import("zig-tensor-core");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create tensors
    const shape = [_]usize{3, 4};
    var tensor_a = try tensor_core.Tensor.init(allocator, &shape, .f32);
    defer tensor_a.deinit();
    
    var tensor_b = try tensor_core.Tensor.init(allocator, &shape, .f32);
    defer tensor_b.deinit();

    // Fill with data
    try tensor_a.fill(1.0);
    try tensor_b.fill(2.0);

    // Perform operations
    var result = try tensor_core.ops.add(allocator, tensor_a, tensor_b);
    defer result.deinit();

    // Access results
    const value = try result.get_f32(&[_]usize{0, 0});
    std.log.info("Result[0,0] = {}", .{value}); // Should be 3.0
}
```

### Advanced Features
```zig
// SIMD-optimized operations
var simd_result = try tensor_core.simd.vectorized_add(allocator, tensor_a, tensor_b);

// Memory-efficient operations with arena allocator
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
var arena_tensor = try tensor_core.Tensor.initArena(arena.allocator(), &shape, .f32);

// Zero-copy views
var view = try tensor_a.slice(&[_]usize{1, 2}, &[_]usize{2, 3});
```

## 📦 Using zig-onnx-parser Independently

### When to Use Standalone
- Model analysis and validation
- Format conversion tools
- Model metadata extraction
- ONNX debugging utilities

### Basic Usage
```zig
const std = @import("std");
const onnx_parser = @import("zig-onnx-parser");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse ONNX model
    var parser = try onnx_parser.Parser.init(allocator);
    defer parser.deinit();
    
    const model = try parser.parseFile("model.onnx");
    defer model.deinit();

    // Extract model information
    const metadata = model.getMetadata();
    std.log.info("Model: {s} v{s}", .{metadata.name, metadata.version});
    std.log.info("Inputs: {}, Outputs: {}", .{metadata.input_count, metadata.output_count});

    // Validate model
    const validation = try parser.validate(model);
    if (validation.is_valid) {
        std.log.info("Model is valid!");
    } else {
        for (validation.errors) |err| {
            std.log.err("Validation error: {s}", .{err.message});
        }
    }
}
```

### Advanced Features
```zig
// Stream parsing for large models
var stream_parser = try onnx_parser.StreamParser.init(allocator);
try stream_parser.parseStream(file_stream, callback);

// Model optimization
var optimizer = try onnx_parser.Optimizer.init(allocator);
const optimized_model = try optimizer.optimize(model, .{
    .fold_constants = true,
    .eliminate_dead_nodes = true,
    .fuse_operations = true,
});

// Format conversion
const converted = try onnx_parser.convert(model, .tensorflow_lite);
```

## ⚙️ Using zig-inference-engine Independently

### When to Use Standalone
- Custom inference applications
- Performance-critical inference
- Embedded inference systems
- Research and experimentation

### Basic Usage
```zig
const std = @import("std");
const inference_engine = @import("zig-inference-engine");
const tensor_core = @import("zig-tensor-core");
const onnx_parser = @import("zig-onnx-parser");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse model
    var parser = try onnx_parser.Parser.init(allocator);
    defer parser.deinit();
    const model = try parser.parseFile("model.onnx");

    // Create inference engine
    var engine = try inference_engine.Engine.init(allocator, .{
        .max_memory_mb = 1024,
        .num_threads = 4,
        .enable_profiling = true,
    });
    defer engine.deinit();

    // Load model
    try engine.loadModel(model);

    // Prepare input
    const input_shape = [_]usize{1, 3, 224, 224};
    var input = try tensor_core.Tensor.init(allocator, &input_shape, .f32);
    defer input.deinit();
    
    // Fill input with your data
    // ... populate input tensor

    // Run inference
    const outputs = try engine.infer(&[_]tensor_core.Tensor{input});
    defer allocator.free(outputs);

    // Process results
    for (outputs) |output| {
        std.log.info("Output shape: {any}", .{output.shape});
        defer output.deinit();
    }
}
```

### Advanced Features
```zig
// GPU acceleration
var gpu_engine = try inference_engine.Engine.init(allocator, .{
    .device = .gpu,
    .gpu_backend = .cuda,
});

// Batch inference
const batch_outputs = try engine.inferBatch(&[_][]tensor_core.Tensor{
    &[_]tensor_core.Tensor{input1},
    &[_]tensor_core.Tensor{input2},
    &[_]tensor_core.Tensor{input3},
});

// Custom operators
try engine.registerOperator("CustomOp", CustomOperator);

// Performance profiling
const stats = engine.getProfilingStats();
std.log.info("Inference time: {}ms", .{stats.total_time_ms});
```

## 🌐 Using zig-model-server Independently

### When to Use Standalone
- HTTP API services
- CLI tools for inference
- Web application backends
- Microservice architectures

### Basic Usage
```zig
const std = @import("std");
const model_server = @import("zig-model-server");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create server
    var server = try model_server.Server.init(allocator, .{
        .host = "127.0.0.1",
        .port = 8080,
        .max_connections = 100,
        .request_timeout_ms = 30000,
    });
    defer server.deinit();

    // Load model
    try server.loadModel("model.onnx");

    // Start server
    std.log.info("Starting server on http://127.0.0.1:8080");
    try server.start();
}
```

### CLI Usage
```bash
# Start HTTP server
cd projects/zig-model-server
zig build run -- server --model model.onnx --port 8080

# Single inference
zig build run -- inference --model model.onnx --prompt "Hello, AI!"

# Interactive mode
zig build run -- interactive --model model.onnx

# Batch processing
zig build run -- batch --model model.onnx --input-file inputs.json
```

### API Integration
```bash
# Health check
curl http://localhost:8080/health

# Single inference
curl -X POST http://localhost:8080/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"input": "What is AI?", "max_tokens": 100}'

# Model information
curl http://localhost:8080/api/v1/model/info
```

## 🎯 Using zig-ai-platform (Unified Approach)

### When to Use Unified Platform
- Complete AI workflows
- Rapid prototyping
- Production deployments
- Educational purposes

### Basic Usage
```zig
const std = @import("std");
const ai_platform = @import("zig-ai-platform");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize platform
    var platform = try ai_platform.Platform.init(allocator, .{
        .auto_configure = true,
        .enable_all_backends = true,
    });
    defer platform.deinit();

    // Load model (automatically handles parsing and engine setup)
    try platform.loadModel("model.onnx");

    // Simple inference
    const result = try platform.infer("What is machine learning?");
    defer allocator.free(result);
    
    std.log.info("AI Response: {s}", .{result});
}
```

### Advanced Workflows
```zig
// Multi-model workflow
var workflow = try ai_platform.Workflow.init(allocator);
try workflow.addModel("text_encoder", "encoder.onnx");
try workflow.addModel("decoder", "decoder.onnx");
try workflow.connect("text_encoder.output", "decoder.input");

const result = try workflow.execute(.{
    .text_encoder = .{ .input = "Hello world" }
});

// Batch processing pipeline
var pipeline = try ai_platform.Pipeline.init(allocator);
try pipeline.addStage(.preprocess, preprocess_fn);
try pipeline.addStage(.inference, inference_fn);
try pipeline.addStage(.postprocess, postprocess_fn);

const results = try pipeline.processBatch(input_batch);
```

## 🔧 Integration Patterns

### Pattern 1: Layered Integration
```zig
// Use components at different abstraction levels
const platform = @import("zig-ai-platform");     // High-level workflows
const server = @import("zig-model-server");      // HTTP API
const engine = @import("zig-inference-engine");  // Core inference
const tensors = @import("zig-tensor-core");      // Low-level operations

// Choose the right level for your needs
```

### Pattern 2: Selective Integration
```zig
// Use only what you need
const tensors = @import("zig-tensor-core");
const parser = @import("zig-onnx-parser");

// Build custom solution without inference engine
var custom_engine = MyCustomEngine.init(tensors, parser);
```

### Pattern 3: Progressive Enhancement
```zig
// Start simple, add complexity as needed
var simple_inference = try tensors.SimpleInference.init(allocator);

// Later, upgrade to full engine
var full_engine = try inference_engine.Engine.init(allocator);
try full_engine.migrateFrom(simple_inference);
```

## 🚀 Best Practices

### Dependency Management
1. **Pin Versions**: Use specific versions for production
2. **Test Compatibility**: Verify component versions work together
3. **Gradual Updates**: Update one component at a time
4. **Interface Stability**: Rely on stable interfaces, not implementations

### Performance Optimization
1. **Choose Right Level**: Use the lowest-level component that meets your needs
2. **Memory Management**: Use arena allocators for batch operations
3. **Threading**: Configure thread counts based on your workload
4. **Profiling**: Enable profiling to identify bottlenecks

### Error Handling
1. **Graceful Degradation**: Handle component failures gracefully
2. **Error Context**: Add context when propagating errors
3. **Logging**: Use structured logging for debugging
4. **Monitoring**: Monitor component health in production

### Testing Strategy
1. **Unit Tests**: Test each component independently
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Validate performance requirements

## 🔍 Troubleshooting

### Common Integration Issues

**Version Mismatch:**
```bash
# Check component versions
zig build info

# Update to compatible versions
git submodule update --remote
```

**Memory Issues:**
```zig
// Use arena allocators for temporary data
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
var temp_tensor = try tensors.Tensor.initArena(arena.allocator(), &shape, .f32);
```

**Performance Issues:**
```zig
// Enable profiling to identify bottlenecks
var engine = try inference_engine.Engine.init(allocator, .{
    .enable_profiling = true,
});

const stats = engine.getProfilingStats();
std.log.info("Bottleneck: {s} took {}ms", .{stats.slowest_op, stats.max_time});
```

This integration guide provides the foundation for effectively using the Zig AI Ecosystem components both individually and together.

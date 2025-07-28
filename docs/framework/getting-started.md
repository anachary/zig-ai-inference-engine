# Getting Started with Zig AI Platform Framework

## 🎯 Overview

The Zig AI Platform Framework provides a clean, modular foundation for building AI inference systems. This guide will get you up and running quickly.

## 📦 Installation

### Prerequisites
- **Zig 0.11.0** or later
- **Git** for cloning the repository
- **8GB RAM** minimum (16GB recommended for large models)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform

# Build the framework
zig build

# Run tests to verify installation
zig build test

# Run a quick demo
zig build run-demo
```

## 🏗️ Architecture Overview

The Zig AI Platform is organized into four main components:

```
zig-ai-platform/
├── framework/          # 🔧 Core framework and interfaces
├── implementations/    # 🚀 Concrete implementations
├── docs/              # 📚 Documentation
└── examples/          # 🎯 Real-world examples
```

### Framework Layer
- **Core interfaces** - Tensor, Operator, ExecutionContext
- **Operator framework** - Base classes and registry
- **Execution engine** - Graph execution and optimization

### Implementation Layer
- **Operators** - Arithmetic, activation, matrix, transformer ops
- **Models** - Transformer, vision, audio model support
- **Backends** - CPU, GPU, SIMD optimizations

## 🚀 Your First AI Program

### 1. Basic Tensor Operations

```zig
const std = @import("std");
const ai = @import("implementations");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create AI platform
    var platform = try ai.utils.createDefaultPlatform(allocator);
    defer platform.deinit();

    // Create tensors
    const shape = [_]usize{ 2, 3 };
    var tensor_a = try ai.utils.createTensor(allocator, &shape, .f32);
    defer tensor_a.deinit();
    var tensor_b = try ai.utils.createTensor(allocator, &shape, .f32);
    defer tensor_b.deinit();
    var result = try ai.utils.createTensor(allocator, &shape, .f32);
    defer result.deinit();

    // Set data
    const data_a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const data_b = [_]f32{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
    try ai.utils.setTensorData(&tensor_a, f32, &data_a);
    try ai.utils.setTensorData(&tensor_b, f32, &data_b);

    // Perform addition
    const inputs = [_]ai.Tensor{ tensor_a, tensor_b };
    var outputs = [_]ai.Tensor{result};
    
    var attrs = ai.utils.createAttributes(allocator);
    defer attrs.deinit();
    
    var context = ai.utils.createExecutionContext(allocator);
    
    try ai.operators.arithmetic.Add.compute(&inputs, &outputs, &attrs, &context);
    
    // Print result
    const result_data = ai.utils.getTensorData(&result, f32);
    std.log.info("Result: {any}", .{result_data});
}
```

### 2. Using the Operator Registry

```zig
const std = @import("std");
const ai = @import("implementations");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create platform with all operators
    var platform = try ai.utils.createDefaultPlatform(allocator);
    defer platform.deinit();

    // List available operators
    const operators = try platform.listOperators();
    defer allocator.free(operators);
    
    std.log.info("Available operators:");
    for (operators) |op| {
        std.log.info("  - {s} v{s}: {s}", .{ op.name, op.version, op.description });
    }

    // Check if specific operators are available
    if (platform.supportsOperator("Add", null)) {
        std.log.info("✅ Add operator is available");
    }
    
    if (platform.supportsOperator("LayerNormalization", null)) {
        std.log.info("✅ LayerNormalization operator is available");
    }
}
```

### 3. Simple Neural Network Layer

```zig
const std = @import("std");
const ai = @import("implementations");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create transformer-optimized platform
    var platform = try ai.utils.createTransformerPlatform(allocator);
    defer platform.deinit();

    // Create input tensor (batch_size=1, seq_len=4, hidden_dim=8)
    const input_shape = [_]usize{ 1, 4, 8 };
    var input = try ai.utils.createTensor(allocator, &input_shape, .f32);
    defer input.deinit();
    var output = try ai.utils.createTensor(allocator, &input_shape, .f32);
    defer output.deinit();

    // Fill input with test data
    var input_data = try allocator.alloc(f32, 32); // 1*4*8 = 32
    defer allocator.free(input_data);
    for (0..32) |i| {
        input_data[i] = @as(f32, @floatFromInt(i)) / 32.0;
    }
    try ai.utils.setTensorData(&input, f32, input_data);

    // Apply Layer Normalization
    const inputs = [_]ai.Tensor{input};
    var outputs = [_]ai.Tensor{output};
    
    var attrs = ai.utils.createAttributes(allocator);
    defer attrs.deinit();
    try attrs.set("epsilon", ai.Attributes.AttributeValue{ .float = 1e-5 });
    try attrs.set("axis", ai.Attributes.AttributeValue{ .int = -1 });
    
    var context = ai.utils.createExecutionContext(allocator);
    
    try ai.models.transformers.LayerNorm.compute(&inputs, &outputs, &attrs, &context);
    
    std.log.info("✅ Layer normalization applied successfully");
    
    // Verify output
    const output_data = ai.utils.getTensorData(&output, f32);
    std.log.info("Output shape: {any}", .{output.shape});
    std.log.info("First few values: {d:.3} {d:.3} {d:.3}", .{ output_data[0], output_data[1], output_data[2] });
}
```

## 🔧 Core Concepts

### Tensors
Tensors are the fundamental data structure for AI computations:

```zig
// Create a tensor
var tensor = try ai.utils.createTensor(allocator, &[_]usize{2, 3}, .f32);
defer tensor.deinit();

// Access tensor properties
std.log.info("Shape: {any}", .{tensor.shape});
std.log.info("Data type: {}", .{tensor.dtype});
std.log.info("Element count: {}", .{tensor.getElementCount()});

// Set and get data
const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
try tensor.setData(f32, &data);
const retrieved = tensor.getData(f32);
```

### Operators
Operators perform computations on tensors:

```zig
// Use built-in operators
try ai.operators.arithmetic.Add.compute(&inputs, &outputs, &attrs, &context);
try ai.operators.activation.ReLU.compute(&inputs, &outputs, &attrs, &context);
try ai.models.transformers.LayerNorm.compute(&inputs, &outputs, &attrs, &context);

// Check operator metadata
const metadata = ai.operators.arithmetic.Add.getMetadata();
std.log.info("Operator: {s} v{s}", .{ metadata.name, metadata.version });
```

### Execution Context
The execution context manages computation resources:

```zig
var context = ai.utils.createExecutionContext(allocator);

// Configure execution
context.device = .cpu;  // or .gpu, .auto
context.profiling_enabled = true;

// Use context for operations
try operator.compute(&inputs, &outputs, &attrs, &context);
```

### Attributes
Attributes configure operator behavior:

```zig
var attrs = ai.utils.createAttributes(allocator);
defer attrs.deinit();

// Set different attribute types
try attrs.set("epsilon", ai.Attributes.AttributeValue{ .float = 1e-5 });
try attrs.set("axis", ai.Attributes.AttributeValue{ .int = -1 });
try attrs.set("mode", ai.Attributes.AttributeValue{ .string = "linear" });

// Get attributes with defaults
const epsilon = attrs.getFloat("epsilon", 1e-6);
const axis = attrs.getInt("axis", 0);
```

## 🎯 Platform Configurations

### Default Platform
```zig
// General-purpose configuration
var platform = try ai.utils.createDefaultPlatform(allocator);
```

### Transformer Platform
```zig
// Optimized for transformer models
var platform = try ai.utils.createTransformerPlatform(allocator);
```

### Edge Platform
```zig
// Optimized for edge devices with limited resources
var platform = try ai.utils.createEdgePlatform(allocator);
```

### Custom Platform
```zig
const config = ai.AIPlatform.Config{
    .framework_config = .{
        .device = .gpu,
        .optimization_level = .aggressive,
        .enable_profiling = true,
        .max_memory_mb = 8192,
    },
    .enable_all_operators = true,
    .enable_transformer_models = true,
};

var platform = try ai.AIPlatform.init(allocator, config);
```

## 🧪 Testing Your Code

### Unit Tests
```zig
test "basic tensor operations" {
    const allocator = std.testing.allocator;
    
    var tensor = try ai.utils.createTensor(allocator, &[_]usize{2, 2}, .f32);
    defer tensor.deinit();
    
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try tensor.setData(f32, &data);
    
    const retrieved = tensor.getData(f32);
    try std.testing.expectEqualSlices(f32, &data, retrieved);
}
```

### Running Tests
```bash
# Run all tests
zig build test

# Run specific test file
zig test src/my_test.zig

# Run with verbose output
zig build test -- --verbose
```

## 📊 Performance Monitoring

### Basic Profiling
```zig
// Enable profiling
var context = ai.utils.createExecutionContext(allocator);
context.profiling_enabled = true;

// Run operations
try operator.compute(&inputs, &outputs, &attrs, &context);

// Get performance stats
const stats = platform.getStats();
std.log.info("Memory used: {} bytes", .{stats.framework_stats.total_memory_used});
std.log.info("Peak memory: {} bytes", .{stats.framework_stats.peak_memory_used});
```

### Benchmarking
```zig
const start_time = std.time.nanoTimestamp();

// Run your operations
for (0..1000) |_| {
    try operator.compute(&inputs, &outputs, &attrs, &context);
}

const end_time = std.time.nanoTimestamp();
const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
std.log.info("Average time per operation: {d:.3} ms", .{duration_ms / 1000.0});
```

## 🔍 Debugging Tips

### Common Issues

1. **Memory Errors**
   ```zig
   // Always defer tensor cleanup
   var tensor = try ai.utils.createTensor(allocator, &shape, .f32);
   defer tensor.deinit(); // Don't forget this!
   ```

2. **Shape Mismatches**
   ```zig
   // Check shapes before operations
   if (!ai.utils.shapesEqual(tensor_a.shape, tensor_b.shape)) {
       return error.ShapeMismatch;
   }
   ```

3. **Data Type Mismatches**
   ```zig
   // Ensure consistent data types
   if (tensor_a.dtype != tensor_b.dtype) {
       return error.DataTypeMismatch;
   }
   ```

### Debugging Tools
```zig
// Print tensor information
std.log.info("Tensor shape: {any}", .{tensor.shape});
std.log.info("Tensor dtype: {}", .{tensor.dtype});
std.log.info("Tensor data: {any}", .{tensor.getData(f32)[0..5]}); // First 5 elements

// Check platform status
const stats = platform.getStats();
std.log.info("Platform stats: {any}", .{stats});
```

## 📚 Next Steps

1. **Explore Examples**
   - Check out `examples/iot/` for edge deployment
   - Look at `examples/aks/` for cloud deployment
   - Try `examples/basic/` for more learning

2. **Read Documentation**
   - `docs/framework/` for framework details
   - `docs/implementations/` for operator reference
   - `docs/tutorials/` for step-by-step guides

3. **Build Something**
   - Create a custom operator
   - Deploy a model to edge devices
   - Set up distributed inference

4. **Contribute**
   - Add new operators
   - Improve documentation
   - Share your examples

## 🎯 Quick Reference

### Essential Imports
```zig
const std = @import("std");
const ai = @import("implementations");
```

### Common Patterns
```zig
// Platform setup
var platform = try ai.utils.createDefaultPlatform(allocator);
defer platform.deinit();

// Tensor creation
var tensor = try ai.utils.createTensor(allocator, &shape, .f32);
defer tensor.deinit();

// Operator execution
try operator.compute(&inputs, &outputs, &attrs, &context);
```

### Build Commands
```bash
zig build                 # Build everything
zig build test           # Run tests
zig build run-demo       # Run demo
zig build examples       # Build examples
zig build docs           # Generate docs
```

You're now ready to start building AI applications with the Zig AI Platform! 🚀

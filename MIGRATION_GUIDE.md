# Zig AI Platform Migration Guide

## 🎯 Overview

This guide helps you migrate from the old mixed architecture to the new modular framework-based architecture. **All your existing work is preserved** - we're just organizing it better!

## 📁 New Architecture Overview

### Before (Mixed Architecture)
```
zig-ai-platform/
├── src/                          # Mixed framework and implementation
├── projects/
│   ├── zig-tensor-core/          # Tensor operations
│   ├── zig-onnx-parser/          # ONNX parsing
│   ├── zig-inference-engine/     # Mixed execution code
│   └── zig-model-server/         # HTTP API
└── build.zig                     # Monolithic build
```

### After (Modular Architecture)
```
zig-ai-platform/
├── framework/                    # 🆕 Core framework (interfaces, abstractions)
│   ├── core/                     # Core interfaces and execution engine
│   ├── operators/                # Operator framework
│   └── lib.zig                   # Framework entry point
├── implementations/              # 🆕 Concrete implementations
│   ├── operators/                # All operators (arithmetic, activation, etc.)
│   ├── models/                   # Model-specific components
│   └── lib.zig                   # Complete AI platform
├── projects/                     # ✅ Existing projects (unchanged)
│   ├── zig-tensor-core/          # Still works exactly the same
│   ├── zig-onnx-parser/          # Still works exactly the same
│   ├── zig-inference-engine/     # Enhanced with framework
│   └── zig-model-server/         # Enhanced with framework
├── src/                          # ✅ Main CLI (backward compatible)
├── build.zig                     # ✅ Original build (still works)
└── build_new.zig                 # 🆕 Enhanced build system
```

## 🔄 Migration Steps

### Step 1: No Changes Required (Backward Compatibility)

Your existing code continues to work exactly as before:

```bash
# These commands still work exactly the same
zig build
zig build run
zig build test
```

### Step 2: Optional - Use New Framework (Recommended)

To take advantage of the new modular architecture:

#### 2.1 Update Build System (Optional)
```bash
# Use the new build system for enhanced features
mv build.zig build_old.zig
mv build_new.zig build.zig
```

#### 2.2 Update Imports in New Code
```zig
// Old way (still works)
const inference_engine = @import("zig-inference-engine");

// New way (recommended for new code)
const framework = @import("framework");
const implementations = @import("implementations");

// Complete AI platform with everything included
var platform = try implementations.AIPlatform.init(allocator, .{});
defer platform.deinit();
```

#### 2.3 Use Enhanced Operator System
```zig
// Old way (still works)
const Add = @import("zig-inference-engine").operators.Add;

// New way (more powerful)
const Add = implementations.operators.arithmetic.Add;

// Or use the complete registry
var registry = try implementations.operators.createBuiltinRegistry(allocator);
defer registry.deinit();

// Execute operators through registry
try registry.executeOperator("Add", inputs, outputs, &attributes, &context, null);
```

## 🆕 New Features Available

### 1. Modular Operator System
```zig
// Easy to add new operators
const MyCustomOperator = framework.BaseOperator(struct {
    pub fn getMetadata() framework.OperatorInterface.Metadata {
        return framework.OperatorInterface.Metadata{
            .name = "MyCustomOp",
            .version = "1.0.0",
            .description = "My custom operator",
            // ... metadata
        };
    }
    
    pub fn compute(inputs: []const framework.Tensor, outputs: []framework.Tensor, 
                   attributes: *const framework.Attributes, context: *framework.ExecutionContext) !void {
        // Your implementation here
    }
});

// Register with platform
try platform.getFramework().registerOperator(MyCustomOperator.getDefinition());
```

### 2. Model-Specific Components
```zig
// Transformer-specific operators now available
const LayerNorm = implementations.models.transformers.LayerNorm;
const MultiHeadAttention = implementations.models.transformers.MultiHeadAttention;
const RMSNorm = implementations.models.transformers.RMSNorm;
```

### 3. Enhanced Execution Engine
```zig
// Create and execute graphs with optimization
var graph = platform.createGraph();
defer graph.deinit();

// Add nodes to graph...
try platform.executeGraph(&graph);

// Get execution statistics
const stats = platform.getStats();
std.log.info("Memory used: {} bytes", .{stats.framework_stats.total_memory_used});
```

## 🔧 Build System Changes

### New Build Commands
```bash
# Framework-specific commands
zig build test-framework          # Test framework only
zig build test-implementations    # Test implementations only
zig build test-all               # Test everything

# Examples and demos
zig build examples               # Build all examples
zig build run-demo              # Run framework demo

# Documentation
zig build docs                  # Generate documentation

# Benchmarks
zig build benchmark             # Run performance benchmarks

# Migration help
zig build migrate              # Show migration guide
```

### Cross-Compilation Support
```bash
# Cross-compile for multiple targets
zig build cross                # All targets
zig build cross-0              # Linux x86_64
zig build cross-1              # Linux ARM64
zig build cross-2              # Windows x86_64
zig build cross-3              # macOS x86_64
zig build cross-4              # macOS ARM64
```

## 📊 What's Preserved vs Enhanced

### ✅ Completely Preserved (No Changes Needed)
- All existing ONNX parsing functionality
- All existing tensor operations
- All existing model loading
- All existing HTTP API endpoints
- All existing CLI commands
- All existing tests
- All existing build commands

### 🚀 Enhanced (Backward Compatible)
- Operator system (old operators work, new system available)
- Execution engine (old execution works, new optimizations available)
- Memory management (old patterns work, new tracking available)
- Error handling (old errors work, new detailed errors available)

### 🆕 New Additions (Optional to Use)
- Framework interfaces and abstractions
- Modular operator registry
- Model-specific architecture support
- Advanced execution optimization
- Comprehensive benchmarking
- Enhanced documentation

## 🎯 Recommended Migration Path

### For Existing Code (Conservative)
1. **Keep using existing imports** - no changes needed
2. **Gradually adopt new features** - when you need them
3. **Use new build system** - for enhanced features

### For New Code (Recommended)
1. **Use new framework imports** - `@import("framework")` and `@import("implementations")`
2. **Use AIPlatform** - `implementations.AIPlatform` for complete functionality
3. **Use operator registry** - for dynamic operator management
4. **Use model-specific components** - for transformer, vision, audio models

## 🔍 Examples

### Example 1: Basic Usage (New Way)
```zig
const std = @import("std");
const implementations = @import("implementations");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create complete AI platform
    var platform = try implementations.utils.createDefaultPlatform(allocator);
    defer platform.deinit();

    // Create tensors
    const shape = [_]usize{ 2, 3 };
    var tensor_a = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer tensor_a.deinit();
    var tensor_b = try implementations.utils.createTensor(allocator, &shape, .f32);
    defer tensor_b.deinit();

    // Set data
    const data_a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const data_b = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    try implementations.utils.setTensorData(&tensor_a, f32, &data_a);
    try implementations.utils.setTensorData(&tensor_b, f32, &data_b);

    // Create graph and execute
    var graph = platform.createGraph();
    defer graph.deinit();
    
    // TODO: Add graph building utilities
    // For now, operators can be executed directly through registry
    
    std.log.info("Platform initialized with {} operators", .{platform.getStats().total_operators});
}
```

### Example 2: Custom Operator (New Feature)
```zig
const framework = @import("framework");
const implementations = @import("implementations");

const MySquare = framework.BaseOperator(struct {
    pub fn getMetadata() framework.OperatorInterface.Metadata {
        return framework.OperatorInterface.Metadata{
            .name = "Square",
            .version = "1.0.0",
            .description = "Element-wise square operation",
            .domain = "custom",
            .min_inputs = 1,
            .max_inputs = 1,
            .min_outputs = 1,
            .max_outputs = 1,
            .type_constraints = &[_]framework.OperatorInterface.TypeConstraint{
                framework.OperatorInterface.TypeConstraint{
                    .name = "T",
                    .allowed_types = &[_]framework.Tensor.DataType{.f32},
                    .description = "Float32 tensors only",
                },
            },
        };
    }

    pub fn validate(input_shapes: []const []const usize, input_types: []const framework.Tensor.DataType, 
                    attributes: *const framework.Attributes) framework.FrameworkError!void {
        _ = attributes;
        if (input_shapes.len != 1 or input_types[0] != .f32) {
            return framework.FrameworkError.InvalidInput;
        }
    }

    pub fn inferShapes(input_shapes: []const []const usize, attributes: *const framework.Attributes, 
                       allocator: std.mem.Allocator) framework.FrameworkError![][]usize {
        _ = attributes;
        const output_shapes = try allocator.alloc([]usize, 1);
        output_shapes[0] = try allocator.dupe(usize, input_shapes[0]);
        return output_shapes;
    }

    pub fn compute(inputs: []const framework.Tensor, outputs: []framework.Tensor, 
                   attributes: *const framework.Attributes, context: *framework.ExecutionContext) framework.FrameworkError!void {
        _ = attributes;
        _ = context;
        
        const input_data = inputs[0].getData(f32);
        const output_data = outputs[0].getMutableData(f32);
        
        for (0..input_data.len) |i| {
            output_data[i] = input_data[i] * input_data[i];
        }
    }
});

pub fn registerCustomOperator(platform: *implementations.AIPlatform) !void {
    try platform.getFramework().registerOperator(MySquare.getDefinition());
}
```

## 🚀 Next Steps

1. **Try the new build system**: `mv build.zig build_old.zig && mv build_new.zig build.zig`
2. **Run the examples**: `zig build examples && zig build run-demo`
3. **Explore new operators**: Check `implementations/operators/` directory
4. **Read the documentation**: `zig build docs`
5. **Run benchmarks**: `zig build benchmark`

## ❓ FAQ

**Q: Do I need to change my existing code?**
A: No! All existing code continues to work exactly as before.

**Q: What if I want to use new features?**
A: Import the new modules (`framework`, `implementations`) and use the enhanced APIs.

**Q: Can I mix old and new approaches?**
A: Yes! The new system is designed for gradual adoption.

**Q: What about performance?**
A: The new system is designed to be faster with better optimization opportunities.

**Q: How do I add new operators?**
A: Use the new `BaseOperator` framework - it's much easier than before!

**Q: What about existing tests?**
A: All existing tests continue to work. New tests can use enhanced testing utilities.

This migration preserves all your hard work while providing a much more powerful and extensible foundation for future development! 🎉

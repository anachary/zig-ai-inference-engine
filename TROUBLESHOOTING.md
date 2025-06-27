# Troubleshooting Guide

This document covers common issues and their solutions for the Zig AI Interface Engine.

## Fixed Issues (Phase 1)

### ✅ Memory Corruption in Operator Registry (FIXED)

**Problem**: Segmentation fault when executing operators through the registry.

**Symptoms**:
```
error: Segmentation fault at address 0x0
src/engine/registry.zig:59:28: in get
```

**Root Cause**: The `ExecutionContext` was storing a pointer to the `OperatorRegistry`, but when the `InferenceEngine` struct was returned from the `init` function, the memory layout could change, invalidating the pointer.

**Solution**: 
- Removed the registry pointer from `ExecutionContext`
- Modified `execute_operator` to accept the registry as a parameter
- This ensures the registry reference is always valid when needed

**Files Changed**:
- `src/engine/registry.zig`: Updated `ExecutionContext` structure
- `src/engine/inference.zig`: Updated operator execution calls

### ✅ Memory Leaks in Tensor Management (FIXED)

**Problem**: Memory leaks when running examples, showing multiple leaked allocations.

**Symptoms**:
```
error(gpa): memory address 0x... leaked:
src/core/tensor.zig:64:44: in init
```

**Root Cause**: The tensor pool was designed to reuse tensors, but the examples were trying to return tensors to the pool at the end without proper cleanup.

**Solution**:
- Added `cleanup_tensor` method for immediate tensor deallocation
- Modified examples to use `defer` statements for proper cleanup
- Updated tensor pool to handle both pooling and immediate cleanup scenarios

**Files Changed**:
- `src/memory/pool.zig`: Added `cleanup_tensor` method
- `src/engine/inference.zig`: Added `cleanup_tensor` wrapper
- `examples/simple_inference.zig`: Updated to use proper cleanup
- `src/test.zig`: Updated test cleanup patterns

### ✅ Operator Not Found Error (FIXED)

**Problem**: `OperatorNotFound` error when trying to execute built-in operators.

**Symptoms**:
```
error: OperatorNotFound
```

**Root Cause**: Memory corruption in the operator registry prevented proper operator lookup.

**Solution**: Fixed by resolving the memory corruption issue described above.

## Current Status

### ✅ Working Features
- All tests pass (`zig build test`)
- Simple inference example works (`zig build run-simple_inference`)
- Memory management is stable
- Operator registry functions correctly
- All 6 built-in operators work (Add, Sub, Mul, ReLU, MatMul, Softmax)

### 🚧 In Development
- Model loading example
- Custom operator example
- Phase 1 demo
- HTTP server functionality

## Common Build Issues

### Zig Version Compatibility
**Requirement**: Zig 0.11+

**Check your version**:
```bash
zig version
```

**If you have an older version**, download the latest from [ziglang.org](https://ziglang.org/download/).

### Build Cache Issues
**Problem**: Stale build artifacts causing compilation errors.

**Solution**:
```bash
# Clear build cache
rm -rf zig-cache/
zig build
```

### Platform-Specific Issues

#### Windows
- Ensure you're using PowerShell or Command Prompt
- Path separators should use backslashes in error messages (this is normal)

#### macOS/Linux
- No known platform-specific issues

## Performance Troubleshooting

### SIMD Detection
**Check if SIMD is working**:
```bash
zig build run-simple_inference
# Look for SIMD operations in the output
```

**Expected**: Operations should complete without errors and show performance benefits.

### Memory Usage
**Monitor memory usage**:
- The engine should report memory statistics
- Look for "Memory usage" and "Peak memory" in the output
- Memory should be properly cleaned up (0 bytes at the end)

## Debugging Tips

### Enable Verbose Logging
```bash
zig build run-simple_inference --verbose
```

### Run Tests with Details
```bash
zig build test --summary all
```

### Check Available Build Targets
```bash
zig build -h
```

## Getting Help

### Before Reporting Issues
1. Ensure you're using Zig 0.11+
2. Clear build cache: `rm -rf zig-cache/`
3. Run tests: `zig build test`
4. Try the working example: `zig build run-simple_inference`

### Reporting Bugs
Include:
- Zig version (`zig version`)
- Operating system
- Full error output
- Steps to reproduce

### Known Limitations
- Only `simple_inference` example is currently working
- Other examples are in development
- HTTP server functionality is planned for Phase 2

## Success Indicators

When everything is working correctly, you should see:

```bash
$ zig build test
# All tests pass

$ zig build run-simple_inference
info: Simple Inference Example - Phase 1 Complete!
info: Engine initialized with enhanced features
# ... successful execution output ...
info: Phase 1 inference example completed successfully!
```

This indicates that all Phase 1 features are working correctly.

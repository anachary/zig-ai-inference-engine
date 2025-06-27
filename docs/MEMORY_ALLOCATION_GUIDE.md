# Memory Allocation Guide: AI Inference Engine

This document explains how memory allocation works in practice within our Zig AI inference engine, with concrete examples and allocation strategies.

## Allocation Hierarchy

```
Application Level
├── InferenceEngine (manages all allocations)
│   ├── OperatorRegistry (operator metadata)
│   ├── TensorPool (tensor reuse and caching)
│   ├── MemoryManager (allocation tracking)
│   └── ExecutionContext (temporary allocations)
└── User Code (tensor creation and cleanup)
```

## Core Allocation Patterns

### 1. Engine Initialization Allocations

```zig
// When InferenceEngine.init() is called:
pub fn init(allocator: Allocator, config: Config) !InferenceEngine {
    var self = InferenceEngine{
        .allocator = allocator,                    // Store allocator reference
        .operator_registry = OperatorRegistry.init(allocator),  // ~1KB
        .tensor_pool = TensorPool.init(allocator, config.pool_size), // ~2KB
        .memory_tracker = MemoryTracker.init(allocator),        // ~512B
        .execution_context = undefined,
    };
    
    // Register built-in operators (allocates operator names)
    try self.operator_registry.register_builtin_operators(); // ~6KB
    
    // Initialize execution context
    self.execution_context = ExecutionContext.init(allocator); // ~1KB
    
    return self; // Total: ~10KB allocated
}
```

**Memory Breakdown:**
```
OperatorRegistry:
├── ArrayList storage: 512 bytes
├── Operator names: 6 strings × ~10 bytes = 60 bytes
├── Hash map overhead: ~1KB
└── Total: ~1.5KB

TensorPool:
├── HashMap buckets: ~1KB
├── ArrayList headers: pool_size × 24 bytes
└── Total: ~2KB (for pool_size=20)

MemoryTracker:
├── Allocation tracking: ~512 bytes
└── Statistics storage: ~256 bytes
```

### 2. Tensor Allocation Workflow

```zig
// When engine.get_tensor() is called:
pub fn get_tensor(self: *Self, shape: []const usize, dtype: DataType) !Tensor {
    // Step 1: Check tensor pool for reusable tensor
    const pool_key = self.tensor_pool.compute_key(shape, dtype);
    if (self.tensor_pool.get_cached(pool_key)) |cached_tensor| {
        return cached_tensor; // No allocation - reuse existing
    }
    
    // Step 2: Create new tensor (multiple allocations)
    return self.tensor_pool.create_new_tensor(shape, dtype);
}

// Inside create_new_tensor():
fn create_new_tensor(self: *TensorPool, shape: []const usize, dtype: DataType) !Tensor {
    // Allocation 1: Tensor data
    const element_count = calculateElements(shape);
    const data_size = element_count * dtype.size();
    const data = try self.allocator.alloc(u8, data_size);
    
    // Allocation 2: Shape array (copy for ownership)
    const shape_copy = try self.allocator.dupe(usize, shape);
    
    // Allocation 3: Strides array
    const strides = try self.allocator.alloc(usize, shape.len);
    calculateStrides(shape, strides); // Fill stride values
    
    return Tensor{
        .data = data,
        .shape = shape_copy,
        .strides = strides,
        .dtype = dtype,
        .allocator = self.allocator,
    };
}
```

**Allocation Example for 2x3 f32 tensor:**
```
Shape: [2, 3]
├── Data allocation: 2 × 3 × 4 bytes = 24 bytes
├── Shape allocation: 2 × 8 bytes = 16 bytes  
├── Strides allocation: 2 × 8 bytes = 16 bytes
└── Total: 56 bytes + allocator overhead (~16 bytes) = 72 bytes
```

### 3. Operator Execution Allocations

```zig
// During operator execution:
pub fn execute_operator(
    self: *ExecutionContext,
    registry: *OperatorRegistry,
    name: []const u8,
    inputs: []const Tensor,
    outputs: []Tensor,
) !void {
    // Step 1: Lookup operator (no allocation)
    const operator = registry.get(name) orelse return RegistryError.OperatorNotFound;
    
    // Step 2: Execute operator (may allocate temporary data)
    try operator.forward(inputs, outputs, self.allocator);
}

// Example: Matrix multiplication temporary allocations
fn matmul_forward(inputs: []const Tensor, outputs: []Tensor, allocator: Allocator) !void {
    const a = inputs[0];
    const b = inputs[1];
    var c = outputs[0];
    
    // For large matrices, might allocate temporary buffers for optimization
    if (a.shape[0] > 64 and a.shape[1] > 64) {
        // Allocate temporary buffer for cache-friendly access
        const temp_size = a.shape[0] * b.shape[1] * @sizeOf(f32);
        const temp_buffer = try allocator.alloc(u8, temp_size);
        defer allocator.free(temp_buffer);
        
        // Use temp_buffer for optimized computation...
    }
    
    // Direct computation for smaller matrices (no extra allocation)
    // ... matrix multiplication logic
}
```

### 4. Memory Pool Management

```zig
// Tensor pool allocation strategy:
pub const TensorPool = struct {
    pools: std.AutoHashMap(u64, std.ArrayList(Tensor)),
    
    pub fn return_tensor(self: *Self, tensor: Tensor) !void {
        const key = self.compute_key(tensor.shape, tensor.dtype);
        
        // Get or create pool for this tensor size/type
        const result = try self.pools.getOrPut(key);
        if (!result.found_existing) {
            result.value_ptr.* = std.ArrayList(Tensor).init(self.allocator);
        }
        
        // Add to pool if not full, otherwise free immediately
        if (result.value_ptr.items.len < self.max_pool_size) {
            try result.value_ptr.append(tensor); // Pool storage
        } else {
            // Pool full - free the tensor immediately
            var mutable_tensor = tensor;
            mutable_tensor.deinit(); // Frees data, shape, strides
        }
    }
};
```

**Pool Memory Layout:**
```
TensorPool HashMap:
├── Key: hash(shape=[2,3], dtype=f32) = 0x1A2B3C4D
│   └── ArrayList: [Tensor1, Tensor2, Tensor3] (reusable)
├── Key: hash(shape=[4,4], dtype=f32) = 0x5E6F7A8B  
│   └── ArrayList: [Tensor4, Tensor5] (reusable)
└── Key: hash(shape=[1,1000], dtype=f32) = 0x9C8D7E6F
    └── ArrayList: [Tensor6] (reusable)
```

## Allocation Strategies by Use Case

### 1. Small Tensors (< 1KB)
```zig
// Strategy: Use tensor pool for frequent reuse
const small_tensor = try engine.get_tensor(&[_]usize{8, 8}, .f32); // 256 bytes
defer try engine.return_tensor(small_tensor); // Return to pool
```

### 2. Large Tensors (> 1MB)
```zig
// Strategy: Direct allocation/deallocation to avoid pool bloat
const large_tensor = try engine.get_tensor(&[_]usize{1000, 1000}, .f32); // 4MB
defer engine.cleanup_tensor(large_tensor); // Immediate cleanup
```

### 3. Temporary Computations
```zig
// Strategy: Arena allocator for batch cleanup
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit(); // Cleans up all at once

const temp_allocator = arena.allocator();
// All temporary allocations use temp_allocator
// Automatically freed when arena.deinit() is called
```

### 4. Model Weights (Read-only)
```zig
// Strategy: Memory mapping for large models
const model_file = try std.fs.cwd().openFile("model.bin", .{});
defer model_file.close();

const model_data = try std.os.mmap(
    null, model_size, std.os.PROT.READ, std.os.MAP.PRIVATE, 
    model_file.handle, 0
);
defer std.os.munmap(model_data);

// No heap allocation - uses virtual memory mapping
```

## Memory Allocation Debugging

### 1. Allocation Tracking
```zig
const TrackingAllocator = struct {
    backing: Allocator,
    total_allocated: std.atomic.Atomic(usize) = std.atomic.Atomic(usize).init(0),
    peak_allocated: std.atomic.Atomic(usize) = std.atomic.Atomic(usize).init(0),
    
    pub fn alloc(self: *@This(), len: usize, alignment: u29) ![]u8 {
        const result = try self.backing.rawAlloc(len, alignment, @returnAddress());
        
        const new_total = self.total_allocated.fetchAdd(len, .SeqCst) + len;
        _ = self.peak_allocated.fetchMax(new_total, .SeqCst);
        
        std.log.debug("ALLOC: {} bytes, total: {} bytes", .{ len, new_total });
        return result;
    }
    
    pub fn free(self: *@This(), buf: []u8, alignment: u29) void {
        const new_total = self.total_allocated.fetchSub(buf.len, .SeqCst) - buf.len;
        std.log.debug("FREE: {} bytes, total: {} bytes", .{ buf.len, new_total });
        
        self.backing.rawFree(buf, alignment, @returnAddress());
    }
};
```

### 2. Memory Usage Monitoring
```zig
pub fn printMemoryStats(engine: *InferenceEngine) void {
    const stats = engine.get_stats();
    
    std.log.info("=== MEMORY STATISTICS ===");
    std.log.info("Current usage: {} bytes", .{stats.memory.current_usage});
    std.log.info("Peak usage: {} bytes", .{stats.memory.peak_usage});
    std.log.info("Tensors in pool: {}", .{stats.tensor_pool.total_pooled});
    std.log.info("Pool hit rate: {d:.1}%", .{stats.tensor_pool.hit_rate * 100});
    std.log.info("Active operators: {}", .{stats.operators.total_operators});
}
```

## Best Practices

### 1. Allocation Patterns
- **Prefer stack allocation** for small, short-lived data
- **Use tensor pools** for frequently reused tensor sizes
- **Batch allocations** when possible to reduce overhead
- **Align allocations** for SIMD operations (32-byte boundaries)

### 2. Memory Management
- **Always pair allocations with cleanup** using defer
- **Use arena allocators** for temporary batch operations
- **Monitor memory usage** in production deployments
- **Profile allocation patterns** to optimize pool sizes

### 3. Performance Optimization
- **Pre-allocate common tensor sizes** during initialization
- **Reuse tensors** through the pool system when possible
- **Avoid allocations** in hot paths (inner loops)
- **Use memory mapping** for large read-only data (model weights)

This allocation strategy ensures predictable memory usage, optimal performance for AI workloads, and efficient resource utilization on edge devices.

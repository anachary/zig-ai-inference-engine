# Zig Memory Model: Complete Guide to Memory Layout and Allocation

This document provides a comprehensive understanding of how Zig manages memory, including stack, heap, and data segment layout. Understanding this is crucial for building high-performance systems like our AI inference engine.

## Memory Layout Overview

```
High Memory Addresses (0xFFFFFFFF on 32-bit, 0xFFFFFFFFFFFFFFFF on 64-bit)
┌─────────────────────────────────────────────────────────────────┐
│                        KERNEL SPACE                             │
│                     (OS Reserved)                               │
├─────────────────────────────────────────────────────────────────┤
│                         STACK                                   │
│                    (grows downward)                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Function Call Frames                                    │    │
│  │ Local Variables                                         │    │
│  │ Function Parameters                                     │    │
│  │ Return Addresses                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                    MEMORY MAPPING                               │
│                 (mmap, shared libraries)                        │
├─────────────────────────────────────────────────────────────────┤
│                         HEAP                                    │
│                     (grows upward)                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Dynamic Allocations                                     │    │
│  │ malloc/free regions                                     │    │
│  │ Zig Allocators                                          │    │
│  │ Tensor Data                                             │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                         BSS                                     │
│                 (uninitialized data)                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Zero-initialized global variables                       │    │
│  │ Static variables without initial values                 │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                        DATA                                     │
│                   (initialized data)                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Global variables with initial values                    │    │
│  │ Static variables with initial values                    │    │
│  │ String literals                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                        TEXT                                     │
│                    (program code)                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Machine code instructions                               │    │
│  │ Function definitions                                    │    │
│  │ Read-only constants                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
Low Memory Addresses (0x00000000)
```

## Zig Memory Allocation Fundamentals

### 1. Allocator Interface

Zig uses an explicit allocator pattern - no hidden allocations:

```zig
const std = @import("std");
const Allocator = std.mem.Allocator;

// All allocations must specify an allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Explicit allocation
const data = try allocator.alloc(u8, 1024);
defer allocator.free(data); // Explicit deallocation
```

### 2. Stack Allocation

**Characteristics:**
- Automatic management (no allocator needed)
- Very fast allocation/deallocation
- Limited size (typically 1-8MB)
- LIFO (Last In, First Out) order

```zig
fn stackExample() void {
    // Stack allocated - automatic cleanup
    var local_array: [1000]f32 = undefined;
    var tensor_shape = [_]usize{2, 3, 4};
    
    // These live on the stack until function returns
    const small_buffer: [64]u8 = std.mem.zeroes([64]u8);
} // All stack memory automatically freed here
```

**Stack Frame Layout:**
```
Higher Addresses
┌─────────────────┐
│ Previous Frame  │
├─────────────────┤
│ Return Address  │
├─────────────────┤
│ Saved Registers │
├─────────────────┤
│ Local Variables │
│ (in declaration │
│  order)         │
├─────────────────┤
│ Function Params │
└─────────────────┘
Lower Addresses
```

### 3. Heap Allocation

**Characteristics:**
- Manual management via allocators
- Larger size limits (system memory)
- Slower than stack
- Can be fragmented

```zig
fn heapExample(allocator: Allocator) !void {
    // Heap allocated - manual management required
    const large_tensor = try allocator.alloc(f32, 1_000_000);
    defer allocator.free(large_tensor);
    
    // Dynamic size allocation
    const dynamic_size = calculateTensorSize();
    const dynamic_tensor = try allocator.alloc(f32, dynamic_size);
    defer allocator.free(dynamic_tensor);
}
```

## Zig Allocator Types

### 1. GeneralPurposeAllocator (GPA)
```zig
var gpa = std.heap.GeneralPurposeAllocator(.{
    .safety = true,        // Enable memory safety checks
    .thread_safe = true,   // Thread-safe operations
}){};
defer _ = gpa.deinit(); // Check for leaks

const allocator = gpa.allocator();
```

**Memory Layout:**
```
┌─────────────────────────────────────────┐
│              GPA Heap                   │
├─────────────────────────────────────────┤
│ Allocation 1 │ Metadata │ Allocation 2 │
├─────────────────────────────────────────┤
│     Free     │ Metadata │     Free     │
└─────────────────────────────────────────┘
```

### 2. ArenaAllocator
```zig
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit(); // Frees ALL allocations at once

const arena_allocator = arena.allocator();
// All allocations freed when arena.deinit() is called
```

**Memory Layout:**
```
┌─────────────────────────────────────────┐
│              Arena Block 1              │
├─────────────────────────────────────────┤
│ Alloc1 │ Alloc2 │ Alloc3 │    Free     │
├─────────────────────────────────────────┤
│              Arena Block 2              │
├─────────────────────────────────────────┤
│ Alloc4 │ Alloc5 │        Free          │
└─────────────────────────────────────────┘
```

### 3. FixedBufferAllocator
```zig
var buffer: [4096]u8 = undefined;
var fba = std.heap.FixedBufferAllocator.init(&buffer);
const allocator = fba.allocator();
// Limited to buffer size, no system calls
```

### 4. StackFallbackAllocator
```zig
var buffer: [1024]u8 = undefined;
var sfa = std.heap.StackFallbackAllocator(1024).init(&buffer);
const allocator = sfa.get();
// Uses stack buffer first, falls back to heap
```

## Memory Alignment

Zig provides explicit control over memory alignment:

```zig
// Default alignment (usually 8 bytes on 64-bit)
const normal_data = try allocator.alloc(f32, 100);

// SIMD-aligned allocation (32-byte for AVX2)
const aligned_data = try allocator.alignedAlloc(f32, 32, 100);

// Compile-time alignment specification
const AlignedStruct = struct {
    data: f32 align(32),  // Force 32-byte alignment
};
```

**Alignment Layout:**
```
Memory Address:  0x1000    0x1020    0x1040    0x1060
                   │         │         │         │
32-byte aligned:   ├─────────┼─────────┼─────────┼─────────
                   │ SIMD    │ SIMD    │ SIMD    │ SIMD
                   │ Vector  │ Vector  │ Vector  │ Vector
                   │ (8xf32) │ (8xf32) │ (8xf32) │ (8xf32)
```

## AI Inference Engine Memory Usage

### Tensor Memory Layout
```zig
pub const Tensor = struct {
    data: []u8,           // Heap: Actual tensor data
    shape: []const usize, // Heap: Shape array
    strides: []const usize, // Heap: Stride array
    dtype: DataType,      // Stack: Enum value
    device: Device,       // Stack: Enum value
    allocator: Allocator, // Stack: Interface
    
    // Memory layout in heap:
    // [data_bytes...][shape_bytes...][strides_bytes...]
};
```

### Memory Pool Layout
```zig
pub const TensorPool = struct {
    // Hash map storing lists of tensors by size
    pools: std.AutoHashMap(u64, std.ArrayList(Tensor)),
    
    // Memory layout:
    // HashMap -> Buckets -> ArrayList -> Tensor objects
    //    │         │           │           │
    //   Stack     Heap        Heap       Heap
};
```

**Pool Memory Organization:**
```
┌─────────────────────────────────────────┐
│            TensorPool                   │
├─────────────────────────────────────────┤
│ Key: 24 (2x3 f32) │ ArrayList of Tensors│
├─────────────────────────────────────────┤
│ Key: 96 (4x6 f32) │ ArrayList of Tensors│
├─────────────────────────────────────────┤
│ Key: 400(10x10f32)│ ArrayList of Tensors│
└─────────────────────────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Tensor 1        │
            │ Tensor 2        │
            │ Tensor 3        │
            │ (reusable)      │
            └─────────────────┘
```

## Memory Safety in Zig

### 1. Compile-time Safety
```zig
// Buffer overflow prevention
var buffer: [10]u8 = undefined;
buffer[15] = 42; // Compile error: index out of bounds

// Use-after-free prevention
var data = try allocator.alloc(u8, 100);
allocator.free(data);
data[0] = 42; // Compile error: use after free (in debug mode)
```

### 2. Runtime Safety (Debug Mode)
```zig
// Bounds checking
const slice = data[0..10];
const value = slice[15]; // Runtime panic in debug mode

// Double-free detection
allocator.free(data);
allocator.free(data); // Runtime panic in debug mode
```

### 3. Memory Leak Detection
```zig
var gpa = std.heap.GeneralPurposeAllocator(.{
    .safety = true, // Enable leak detection
}){};
defer {
    const leaked = gpa.deinit();
    if (leaked) {
        std.log.err("Memory leaked!");
    }
}
```

## Performance Considerations

### 1. Allocation Patterns
```zig
// ❌ Bad: Many small allocations
for (0..1000) |i| {
    const small = try allocator.alloc(u8, 10);
    // ... use small
    allocator.free(small);
}

// ✅ Good: Single large allocation
const large = try allocator.alloc(u8, 10000);
defer allocator.free(large);
for (0..1000) |i| {
    const slice = large[i*10..(i+1)*10];
    // ... use slice
}
```

### 2. Memory Locality
```zig
// ✅ Good: Contiguous memory for cache efficiency
const tensor_data = try allocator.alloc(f32, width * height);

// ❌ Bad: Scattered allocations
var rows = try allocator.alloc([]f32, height);
for (rows) |*row| {
    row.* = try allocator.alloc(f32, width); // Poor cache locality
}
```

### 3. SIMD Alignment
```zig
// ✅ Optimal for SIMD operations
const simd_data = try allocator.alignedAlloc(f32, 32, size);
// Can use AVX2 instructions efficiently

// ❌ Suboptimal alignment
const normal_data = try allocator.alloc(f32, size);
// May require unaligned SIMD loads (slower)
```

## Real-World Example: Tensor Allocation in Our AI Engine

### Memory Trace of Simple Inference
```zig
// 1. Engine initialization (stack + heap)
var engine = try InferenceEngine.init(allocator, .{
    .max_memory_mb = 512,
    .tensor_pool_size = 20,
});
// Stack: engine struct (few hundred bytes)
// Heap: operator registry, tensor pool hash map

// 2. Tensor creation
var input = try engine.get_tensor(&[_]usize{2, 3}, .f32);
// Heap allocations:
//   - data: 24 bytes (2*3*4 bytes for f32)
//   - shape: 16 bytes (2*8 bytes for usize)
//   - strides: 16 bytes (2*8 bytes for usize)
//   Total: ~56 bytes + allocator overhead

// 3. Operator execution
try engine.execute_operator("Add", &inputs, &outputs);
// Stack: function parameters, local variables
// Heap: temporary calculations (if any)

// 4. Cleanup
engine.cleanup_tensor(input);
// Heap: memory returned to allocator
```

### Memory Layout During Inference
```
Stack (per thread):
┌─────────────────────────────────────────┐
│ main() frame                            │
│ ├─ engine: InferenceEngine             │
│ ├─ shape: [2]usize                     │
│ ├─ input: Tensor (struct)              │
│ └─ local variables                     │
├─────────────────────────────────────────┤
│ execute_operator() frame                │
│ ├─ op_name: []const u8                 │
│ ├─ inputs: []const Tensor              │
│ ├─ outputs: []Tensor                   │
│ └─ operator: Operator                  │
├─────────────────────────────────────────┤
│ Add.forward() frame                     │
│ ├─ loop indices                        │
│ ├─ SIMD vectors                        │
│ └─ temporary calculations              │
└─────────────────────────────────────────┘

Heap:
┌─────────────────────────────────────────┐
│ OperatorRegistry                        │
│ ├─ operators: ArrayList                │
│ └─ operator names (strings)            │
├─────────────────────────────────────────┤
│ TensorPool                              │
│ ├─ HashMap buckets                     │
│ └─ Tensor ArrayLists                   │
├─────────────────────────────────────────┤
│ Tensor Data                             │
│ ├─ input.data: [24]u8                 │
│ ├─ input.shape: [2]usize               │
│ ├─ input.strides: [2]usize             │
│ ├─ output.data: [24]u8                │
│ └─ ... more tensors                    │
└─────────────────────────────────────────┘
```

## Advanced Memory Patterns

### 1. Memory-Mapped Model Loading
```zig
// For large models, use memory mapping instead of heap allocation
const model_file = try std.fs.cwd().openFile("model.bin", .{});
defer model_file.close();

const model_size = try model_file.getEndPos();
const model_data = try std.os.mmap(
    null,                    // Let OS choose address
    model_size,              // Size to map
    std.os.PROT.READ,        // Read-only
    std.os.MAP.PRIVATE,      // Private mapping
    model_file.handle,       // File handle
    0,                       // Offset
);
defer std.os.munmap(model_data);

// model_data is now mapped into virtual memory
// OS loads pages on-demand (lazy loading)
```

### 2. Custom Allocator for AI Workloads
```zig
const AIAllocator = struct {
    backing_allocator: Allocator,
    tensor_arena: std.heap.ArenaAllocator,
    temp_arena: std.heap.ArenaAllocator,

    pub fn init(backing: Allocator) AIAllocator {
        return AIAllocator{
            .backing_allocator = backing,
            .tensor_arena = std.heap.ArenaAllocator.init(backing),
            .temp_arena = std.heap.ArenaAllocator.init(backing),
        };
    }

    pub fn allocTensor(self: *AIAllocator, size: usize) ![]u8 {
        // Long-lived tensor data
        return self.tensor_arena.allocator().alloc(u8, size);
    }

    pub fn allocTemp(self: *AIAllocator, size: usize) ![]u8 {
        // Temporary computation data
        return self.temp_arena.allocator().alloc(u8, size);
    }

    pub fn clearTemp(self: *AIAllocator) void {
        // Clear all temporary allocations
        self.temp_arena.deinit();
        self.temp_arena = std.heap.ArenaAllocator.init(self.backing_allocator);
    }
};
```

### 3. Zero-Copy Tensor Views
```zig
pub const TensorView = struct {
    data: []u8,           // Points to existing data (no allocation)
    shape: []const usize, // Points to existing shape (no allocation)
    strides: []const usize,
    dtype: DataType,

    // Create view without copying data
    pub fn slice(self: TensorView, start: []const usize, end: []const usize) TensorView {
        // Calculate new data pointer and shape
        const offset = self.calculateOffset(start);
        return TensorView{
            .data = self.data[offset..],
            .shape = calculateNewShape(start, end),
            .strides = self.strides,
            .dtype = self.dtype,
        };
    }
};
```

## Memory Debugging and Profiling

### 1. Allocation Tracking
```zig
const TrackingAllocator = struct {
    backing_allocator: Allocator,
    total_allocated: usize = 0,
    peak_allocated: usize = 0,
    allocation_count: usize = 0,

    pub fn alloc(self: *TrackingAllocator, len: usize, alignment: u29) ![]u8 {
        const result = try self.backing_allocator.rawAlloc(len, alignment, @returnAddress());
        self.total_allocated += len;
        self.allocation_count += 1;
        self.peak_allocated = @max(self.peak_allocated, self.total_allocated);

        std.log.debug("Allocated {} bytes, total: {}, peak: {}", .{
            len, self.total_allocated, self.peak_allocated
        });

        return result;
    }

    pub fn free(self: *TrackingAllocator, buf: []u8, alignment: u29) void {
        self.total_allocated -= buf.len;
        self.backing_allocator.rawFree(buf, alignment, @returnAddress());

        std.log.debug("Freed {} bytes, total: {}", .{ buf.len, self.total_allocated });
    }
};
```

### 2. Memory Layout Visualization
```zig
pub fn printMemoryLayout(allocator: Allocator) void {
    const info = @import("builtin").os.tag;

    // Get current memory usage
    if (info == .linux) {
        const proc_status = std.fs.openFileAbsolute("/proc/self/status", .{}) catch return;
        defer proc_status.close();

        var buf: [4096]u8 = undefined;
        const bytes_read = proc_status.readAll(&buf) catch return;

        // Parse VmRSS (resident memory) and VmSize (virtual memory)
        std.log.info("Memory layout:");
        std.log.info("Virtual memory size: {} KB", .{parseMemoryValue(buf[0..bytes_read], "VmSize:")});
        std.log.info("Resident memory: {} KB", .{parseMemoryValue(buf[0..bytes_read], "VmRSS:")});
        std.log.info("Stack size: {} KB", .{parseMemoryValue(buf[0..bytes_read], "VmStk:")});
        std.log.info("Heap size: {} KB", .{parseMemoryValue(buf[0..bytes_read], "VmData:")});
    }
}
```

This comprehensive memory model understanding enables you to:
1. **Optimize performance** by choosing appropriate allocation strategies
2. **Debug memory issues** by understanding layout and ownership
3. **Design efficient data structures** for AI workloads
4. **Ensure memory safety** through Zig's compile-time guarantees
5. **Profile and tune** memory usage for specific hardware constraints

## Step-by-Step Memory Allocation Workflow

### Complete Example: Matrix Multiplication Memory Flow

Let's trace exactly how memory is allocated when running a simple matrix multiplication:

```zig
const std = @import("std");
const lib = @import("zig-ai-engine");

pub fn main() !void {
    // STEP 1: Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== MEMORY ALLOCATION WORKFLOW ===");

    // STEP 2: Engine initialization
    std.log.info("Step 1: Initializing engine...");
    var engine = try lib.Engine.init(allocator, .{
        .max_memory_mb = 256,
        .num_threads = 2,
        .tensor_pool_size = 10,
    });
    defer engine.deinit();

    // STEP 3: Create input tensors
    std.log.info("Step 2: Creating tensors...");
    const shape_a = [_]usize{ 2, 3 };  // 2x3 matrix
    const shape_b = [_]usize{ 3, 2 };  // 3x2 matrix
    const shape_c = [_]usize{ 2, 2 };  // 2x2 result

    var matrix_a = try engine.get_tensor(&shape_a, .f32);
    var matrix_b = try engine.get_tensor(&shape_b, .f32);
    var result = try engine.get_tensor(&shape_c, .f32);

    // STEP 4: Fill with data
    std.log.info("Step 3: Filling tensors with data...");
    try fillMatrix(matrix_a, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    try fillMatrix(matrix_b, &[_]f32{ 1, 2, 3, 4, 5, 6 });

    // STEP 5: Execute operation
    std.log.info("Step 4: Executing matrix multiplication...");
    const inputs = [_]lib.Tensor{ matrix_a, matrix_b };
    var outputs = [_]lib.Tensor{result};
    try engine.execute_operator("MatMul", &inputs, &outputs);

    // STEP 6: Print results
    std.log.info("Step 5: Results computed!");
    try printMatrix(result);

    // STEP 7: Cleanup
    std.log.info("Step 6: Cleaning up memory...");
    engine.cleanup_tensor(matrix_a);
    engine.cleanup_tensor(matrix_b);
    engine.cleanup_tensor(result);

    std.log.info("=== WORKFLOW COMPLETE ===");
}

fn fillMatrix(tensor: lib.Tensor, data: []const f32) !void {
    for (data, 0..) |value, i| {
        const row = i / tensor.shape[1];
        const col = i % tensor.shape[1];
        try tensor.set_f32(&[_]usize{ row, col }, value);
    }
}

fn printMatrix(tensor: lib.Tensor) !void {
    std.log.info("Matrix {}x{}:", .{ tensor.shape[0], tensor.shape[1] });
    for (0..tensor.shape[0]) |row| {
        var row_str = std.ArrayList(u8).init(std.heap.page_allocator);
        defer row_str.deinit();

        for (0..tensor.shape[1]) |col| {
            const value = try tensor.get_f32(&[_]usize{ row, col });
            try row_str.writer().print("{d:.1} ", .{value});
        }
        std.log.info("  [{s}]", .{row_str.items});
    }
}
```

### Memory Allocation Timeline

```
Time →  Memory Event                    Stack Usage    Heap Usage    Total
────────────────────────────────────────────────────────────────────────────
T0      Program start                   4KB            0KB           4KB
T1      GPA initialization              4KB            64KB          68KB
T2      Engine.init() called            8KB            64KB          72KB
T3      ├─ OperatorRegistry.init()      8KB            65KB          73KB
T4      ├─ TensorPool.init()            8KB            66KB          74KB
T5      ├─ register_builtin_operators() 8KB            68KB          76KB
T6      └─ ExecutionContext.init()      8KB            68KB          76KB
T7      get_tensor(2x3) called          12KB           68KB          80KB
T8      ├─ Tensor.init()                12KB           68KB          80KB
T9      ├─ alloc data (24 bytes)        12KB           68KB          80KB
T10     ├─ alloc shape (16 bytes)       12KB           68KB          80KB
T11     └─ alloc strides (16 bytes)     12KB           68KB          80KB
T12     get_tensor(3x2) called          16KB           69KB          85KB
T13     ├─ Similar allocations...       16KB           69KB          85KB
T14     get_tensor(2x2) called          20KB           70KB          90KB
T15     ├─ Similar allocations...       20KB           70KB          90KB
T16     fillMatrix() operations         24KB           70KB          94KB
T17     execute_operator("MatMul")      28KB           70KB          98KB
T18     ├─ Registry lookup              28KB           70KB          98KB
T19     ├─ MatMul.forward()             32KB           70KB          102KB
T20     ├─ SIMD operations              32KB           70KB          102KB
T21     └─ Write results                32KB           70KB          102KB
T22     cleanup_tensor() calls          20KB           70KB          90KB
T23     ├─ Free tensor data             20KB           69KB          89KB
T24     ├─ Free shape array             20KB           69KB          89KB
T25     └─ Free strides array           20KB           69KB          89KB
T26     engine.deinit()                 8KB            64KB          72KB
T27     gpa.deinit()                    4KB            0KB           4KB
T28     Program end                     0KB            0KB           0KB
```

### Detailed Memory Layout at Peak Usage (T21)

```
STACK (32KB total):
┌─────────────────────────────────────────┐ ← Stack Pointer (SP)
│ MatMul.forward() Frame (4KB)            │
│ ├─ Local variables                      │
│ ├─ Loop counters                        │
│ ├─ SIMD vectors (__m256)                │
│ └─ Temporary calculations               │
├─────────────────────────────────────────┤
│ execute_operator() Frame (4KB)          │
│ ├─ op_name: "MatMul"                    │
│ ├─ inputs: [2]Tensor                    │
│ ├─ outputs: [1]Tensor                   │
│ └─ operator: Operator                   │
├─────────────────────────────────────────┤
│ main() Frame (24KB)                     │
│ ├─ gpa: GeneralPurposeAllocator         │
│ ├─ allocator: Allocator                 │
│ ├─ engine: InferenceEngine              │
│ ├─ shape_a, shape_b, shape_c: [N]usize  │
│ ├─ matrix_a, matrix_b, result: Tensor   │
│ └─ inputs, outputs arrays               │
└─────────────────────────────────────────┘ ← Stack Base

HEAP (70KB total):
┌─────────────────────────────────────────┐
│ GPA Metadata (64KB)                    │
│ ├─ Free list management                 │
│ ├─ Allocation headers                   │
│ └─ Debug information                    │
├─────────────────────────────────────────┤
│ OperatorRegistry (1KB)                  │
│ ├─ ArrayList storage                    │
│ ├─ Operator names: "Add", "MatMul"...   │
│ └─ Operator function pointers           │
├─────────────────────────────────────────┤
│ TensorPool HashMap (1KB)                │
│ ├─ Hash buckets                         │
│ └─ ArrayList headers                    │
├─────────────────────────────────────────┤
│ Tensor A Data (4KB)                     │
│ ├─ data: [24]u8 (2x3x4 bytes)          │
│ ├─ shape: [16]u8 (2x8 bytes)           │
│ └─ strides: [16]u8 (2x8 bytes)         │
├─────────────────────────────────────────┤
│ Tensor B Data (4KB)                     │
│ ├─ data: [24]u8 (3x2x4 bytes)          │
│ ├─ shape: [16]u8 (2x8 bytes)           │
│ └─ strides: [16]u8 (2x8 bytes)         │
├─────────────────────────────────────────┤
│ Result Tensor Data (4KB)                │
│ ├─ data: [16]u8 (2x2x4 bytes)          │
│ ├─ shape: [16]u8 (2x8 bytes)           │
│ └─ strides: [16]u8 (2x8 bytes)         │
└─────────────────────────────────────────┘
```

### Memory Access Patterns During MatMul

```zig
// Inside MatMul.forward() - memory access pattern:
fn forward(inputs: []const Tensor, outputs: []Tensor, allocator: Allocator) !void {
    const a = inputs[0];  // Stack: pointer to heap data
    const b = inputs[1];  // Stack: pointer to heap data
    var c = outputs[0];   // Stack: pointer to heap data

    // Memory access pattern for C[i][j] = Σ A[i][k] * B[k][j]
    for (0..a.shape[0]) |i| {        // i = 0, 1
        for (0..b.shape[1]) |j| {    // j = 0, 1
            var sum: f32 = 0.0;      // Stack: local variable

            for (0..a.shape[1]) |k| { // k = 0, 1, 2
                // Memory reads from heap:
                const a_val = try a.get_f32(&[_]usize{i, k});
                const b_val = try b.get_f32(&[_]usize{k, j});

                // Computation on stack:
                sum += a_val * b_val;
            }

            // Memory write to heap:
            try c.set_f32(&[_]usize{i, j}, sum);
        }
    }
}
```

### Memory Access Visualization

```
Matrix A (2x3):     Matrix B (3x2):     Result C (2x2):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 1.0 │ 2.0 │ 3.0 │ │ 1.0 │ 2.0 │     │ 22.0│ 28.0│
├─────┼─────┼─────┤ ├─────┼─────┤     ├─────┼─────┤
│ 4.0 │ 5.0 │ 6.0 │ │ 3.0 │ 4.0 │     │ 49.0│ 64.0│
└─────────────────┘ │ 5.0 │ 6.0 │     └─────────────┘
                    └─────────────┘

Memory Layout in Heap:
A.data: [1.0][2.0][3.0][4.0][5.0][6.0] ← 24 bytes, row-major
B.data: [1.0][2.0][3.0][4.0][5.0][6.0] ← 24 bytes, row-major
C.data: [22.0][28.0][49.0][64.0]       ← 16 bytes, row-major

Access Pattern for C[0][0] = 22.0:
1. Read A[0][0] = 1.0  (offset 0)
2. Read B[0][0] = 1.0  (offset 0)
3. Read A[0][1] = 2.0  (offset 4)
4. Read B[1][0] = 3.0  (offset 8)
5. Read A[0][2] = 3.0  (offset 8)
6. Read B[2][0] = 5.0  (offset 16)
7. Compute: 1*1 + 2*3 + 3*5 = 22.0
8. Write C[0][0] = 22.0 (offset 0)
```

This workflow demonstrates how Zig's explicit memory management provides:
1. **Predictable allocation patterns** - no hidden allocations
2. **Efficient memory layout** - contiguous data for cache efficiency
3. **Clear ownership semantics** - explicit cleanup prevents leaks
4. **Performance optimization** - SIMD-aligned allocations when needed
5. **Memory safety** - compile-time bounds checking and leak detection

The key insight is that Zig gives you explicit control over every allocation, enabling predictable performance crucial for real-time AI inference on resource-constrained devices.

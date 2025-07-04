const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Mutex = std.Thread.Mutex;
const Atomic = std.atomic.Atomic;

const DistributedTensor = @import("inference_coordinator.zig").DistributedTensor;

/// Memory pool for distributed tensors
pub const DistributedMemoryPool = struct {
    allocator: Allocator,
    pools: HashMap(usize, TensorPool),
    total_allocated: Atomic(u64),
    total_freed: Atomic(u64),
    max_memory_bytes: u64,
    pool_mutex: Mutex,
    
    const Self = @This();
    
    const TensorPool = struct {
        size_bytes: usize,
        available: ArrayList(*DistributedTensor),
        in_use: ArrayList(*DistributedTensor),
        mutex: Mutex,
        
        fn init(allocator: Allocator, size_bytes: usize) TensorPool {
            return TensorPool{
                .size_bytes = size_bytes,
                .available = ArrayList(*DistributedTensor).init(allocator),
                .in_use = ArrayList(*DistributedTensor).init(allocator),
                .mutex = Mutex{},
            };
        }
        
        fn deinit(self: *TensorPool, allocator: Allocator) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            // Free all tensors
            for (self.available.items) |tensor| {
                tensor.deinit(allocator);
                allocator.destroy(tensor);
            }
            for (self.in_use.items) |tensor| {
                tensor.deinit(allocator);
                allocator.destroy(tensor);
            }
            
            self.available.deinit();
            self.in_use.deinit();
        }
    };
    
    pub fn init(allocator: Allocator, max_memory_bytes: u64) Self {
        return Self{
            .allocator = allocator,
            .pools = HashMap(usize, TensorPool).init(allocator),
            .total_allocated = Atomic(u64).init(0),
            .total_freed = Atomic(u64).init(0),
            .max_memory_bytes = max_memory_bytes,
            .pool_mutex = Mutex{},
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.pool_mutex.lock();
        defer self.pool_mutex.unlock();
        
        var iterator = self.pools.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.pools.deinit();
    }
    
    /// Allocate tensor from pool
    pub fn allocateTensor(self: *Self, shape: []const u32, dtype: DistributedTensor.DataType) !*DistributedTensor {
        const size_bytes = self.calculateTensorSize(shape, dtype);
        
        // Check memory limit
        const current_allocated = self.total_allocated.load(.Monotonic);
        if (current_allocated + size_bytes > self.max_memory_bytes) {
            return error.OutOfMemory;
        }
        
        // Get or create pool for this size
        var pool = try self.getOrCreatePool(size_bytes);
        
        pool.mutex.lock();
        defer pool.mutex.unlock();
        
        // Try to reuse existing tensor
        if (pool.available.items.len > 0) {
            const tensor = pool.available.pop();
            try pool.in_use.append(tensor);
            
            // Update tensor metadata
            tensor.shape = try self.allocator.dupe(u32, shape);
            tensor.dtype = dtype;
            
            return tensor;
        }
        
        // Create new tensor
        const tensor = try self.allocator.create(DistributedTensor);
        tensor.* = try DistributedTensor.init(self.allocator, shape, dtype);
        
        try pool.in_use.append(tensor);
        _ = self.total_allocated.fetchAdd(size_bytes, .Monotonic);
        
        return tensor;
    }
    
    /// Return tensor to pool
    pub fn deallocateTensor(self: *Self, tensor: *DistributedTensor) !void {
        const size_bytes = self.calculateTensorSizeFromTensor(tensor);
        
        var pool = self.pools.getPtr(size_bytes) orelse return error.InvalidTensor;
        
        pool.mutex.lock();
        defer pool.mutex.unlock();
        
        // Find and remove from in_use
        for (pool.in_use.items, 0..) |in_use_tensor, i| {
            if (in_use_tensor == tensor) {
                _ = pool.in_use.swapRemove(i);
                try pool.available.append(tensor);
                _ = self.total_freed.fetchAdd(size_bytes, .Monotonic);
                return;
            }
        }
        
        return error.TensorNotFound;
    }
    
    /// Get or create pool for specific size
    fn getOrCreatePool(self: *Self, size_bytes: usize) !*TensorPool {
        self.pool_mutex.lock();
        defer self.pool_mutex.unlock();
        
        if (self.pools.getPtr(size_bytes)) |pool| {
            return pool;
        }
        
        const pool = TensorPool.init(self.allocator, size_bytes);
        try self.pools.put(size_bytes, pool);
        
        return self.pools.getPtr(size_bytes).?;
    }
    
    /// Calculate tensor size in bytes
    fn calculateTensorSize(self: *Self, shape: []const u32, dtype: DistributedTensor.DataType) usize {
        _ = self;
        
        var total_elements: usize = 1;
        for (shape) |dim| {
            total_elements *= dim;
        }
        
        const element_size: usize = switch (dtype) {
            .f32 => 4,
            .f16 => 2,
            .i32 => 4,
            .i8 => 1,
        };
        
        return total_elements * element_size;
    }
    
    /// Calculate tensor size from existing tensor
    fn calculateTensorSizeFromTensor(self: *Self, tensor: *const DistributedTensor) usize {
        return self.calculateTensorSize(tensor.shape, tensor.dtype);
    }
    
    /// Get memory statistics
    pub fn getStats(self: *Self) MemoryStats {
        return MemoryStats{
            .total_allocated_bytes = self.total_allocated.load(.Monotonic),
            .total_freed_bytes = self.total_freed.load(.Monotonic),
            .current_usage_bytes = self.total_allocated.load(.Monotonic) - self.total_freed.load(.Monotonic),
            .max_memory_bytes = self.max_memory_bytes,
            .pool_count = self.pools.count(),
        };
    }
    
    pub const MemoryStats = struct {
        total_allocated_bytes: u64,
        total_freed_bytes: u64,
        current_usage_bytes: u64,
        max_memory_bytes: u64,
        pool_count: u32,
    };
};

/// Tensor serialization for network transfer
pub const TensorSerializer = struct {
    allocator: Allocator,
    compression_enabled: bool,
    
    const Self = @This();
    
    pub const SerializationFormat = enum {
        binary,
        json,
        compressed_binary,
    };
    
    pub fn init(allocator: Allocator, compression_enabled: bool) Self {
        return Self{
            .allocator = allocator,
            .compression_enabled = compression_enabled,
        };
    }
    
    /// Serialize tensor to bytes
    pub fn serialize(self: *Self, tensor: *const DistributedTensor, format: SerializationFormat) ![]u8 {
        switch (format) {
            .binary => return try self.serializeBinary(tensor),
            .json => return try self.serializeJson(tensor),
            .compressed_binary => return try self.serializeCompressedBinary(tensor),
        }
    }
    
    /// Deserialize tensor from bytes
    pub fn deserialize(self: *Self, data: []const u8, format: SerializationFormat) !DistributedTensor {
        switch (format) {
            .binary => return try self.deserializeBinary(data),
            .json => return try self.deserializeJson(data),
            .compressed_binary => return try self.deserializeCompressedBinary(data),
        }
    }
    
    /// Binary serialization (most efficient)
    fn serializeBinary(self: *Self, tensor: *const DistributedTensor) ![]u8 {
        // Calculate total size needed
        const header_size = @sizeOf(BinaryHeader);
        const shape_size = tensor.shape.len * @sizeOf(u32);
        const data_size = tensor.data.len * @sizeOf(f32);
        const total_size = header_size + shape_size + data_size;
        
        var buffer = try self.allocator.alloc(u8, total_size);
        var offset: usize = 0;
        
        // Write header
        const header = BinaryHeader{
            .magic = 0x54454E53, // "TENS"
            .version = 1,
            .dtype = @intFromEnum(tensor.dtype),
            .shape_len = @intCast(tensor.shape.len),
            .data_len = @intCast(tensor.data.len),
            .shard_id = tensor.shard_id,
            .layer_id = tensor.layer_id,
        };
        
        @memcpy(buffer[offset..offset + header_size], std.mem.asBytes(&header));
        offset += header_size;
        
        // Write shape
        const shape_bytes = std.mem.sliceAsBytes(tensor.shape);
        @memcpy(buffer[offset..offset + shape_size], shape_bytes);
        offset += shape_size;
        
        // Write data
        const data_bytes = std.mem.sliceAsBytes(tensor.data);
        @memcpy(buffer[offset..offset + data_size], data_bytes);
        
        return buffer;
    }
    
    /// Binary deserialization
    fn deserializeBinary(self: *Self, data: []const u8) !DistributedTensor {
        if (data.len < @sizeOf(BinaryHeader)) {
            return error.InvalidData;
        }
        
        var offset: usize = 0;
        
        // Read header
        const header = @as(*const BinaryHeader, @ptrCast(@alignCast(data[offset..].ptr))).*;
        offset += @sizeOf(BinaryHeader);
        
        if (header.magic != 0x54454E53) {
            return error.InvalidMagic;
        }
        
        // Read shape
        const shape_size = header.shape_len * @sizeOf(u32);
        if (offset + shape_size > data.len) {
            return error.InvalidData;
        }
        
        const shape_data = data[offset..offset + shape_size];
        const shape = try self.allocator.alloc(u32, header.shape_len);
        @memcpy(std.mem.sliceAsBytes(shape), shape_data);
        offset += shape_size;
        
        // Read tensor data
        const data_size = header.data_len * @sizeOf(f32);
        if (offset + data_size > data.len) {
            return error.InvalidData;
        }
        
        const tensor_data = try self.allocator.alloc(f32, header.data_len);
        const tensor_data_bytes = std.mem.sliceAsBytes(tensor_data);
        @memcpy(tensor_data_bytes, data[offset..offset + data_size]);
        
        return DistributedTensor{
            .data = tensor_data,
            .shape = shape,
            .dtype = @enumFromInt(header.dtype),
            .shard_id = header.shard_id,
            .layer_id = header.layer_id,
        };
    }
    
    /// JSON serialization (human readable, less efficient)
    fn serializeJson(self: *Self, tensor: *const DistributedTensor) ![]u8 {
        const TensorJson = struct {
            shape: []u32,
            dtype: DistributedTensor.DataType,
            shard_id: u32,
            layer_id: u32,
            data: []f32,
        };
        
        const tensor_json = TensorJson{
            .shape = tensor.shape,
            .dtype = tensor.dtype,
            .shard_id = tensor.shard_id,
            .layer_id = tensor.layer_id,
            .data = tensor.data,
        };
        
        return try std.json.stringifyAlloc(self.allocator, tensor_json, .{});
    }
    
    /// JSON deserialization
    fn deserializeJson(self: *Self, data: []const u8) !DistributedTensor {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, data, .{});
        defer parsed.deinit();
        
        // Extract fields from JSON
        const obj = parsed.value.object;
        
        // Parse shape
        const shape_array = obj.get("shape").?.array;
        const shape = try self.allocator.alloc(u32, shape_array.items.len);
        for (shape_array.items, 0..) |item, i| {
            shape[i] = @intCast(item.integer);
        }
        
        // Parse data
        const data_array = obj.get("data").?.array;
        const tensor_data = try self.allocator.alloc(f32, data_array.items.len);
        for (data_array.items, 0..) |item, i| {
            tensor_data[i] = @floatCast(item.float);
        }
        
        return DistributedTensor{
            .data = tensor_data,
            .shape = shape,
            .dtype = .f32, // Simplified for now
            .shard_id = @intCast(obj.get("shard_id").?.integer),
            .layer_id = @intCast(obj.get("layer_id").?.integer),
        };
    }
    
    /// Compressed binary serialization
    fn serializeCompressedBinary(self: *Self, tensor: *const DistributedTensor) ![]u8 {
        // First serialize to binary
        const binary_data = try self.serializeBinary(tensor);
        defer self.allocator.free(binary_data);
        
        // Then compress (simplified - would use actual compression library)
        // For now, just return binary data
        return try self.allocator.dupe(u8, binary_data);
    }
    
    /// Compressed binary deserialization
    fn deserializeCompressedBinary(self: *Self, data: []const u8) !DistributedTensor {
        // Decompress first (simplified)
        // For now, just deserialize as binary
        return try self.deserializeBinary(data);
    }
    
    const BinaryHeader = packed struct {
        magic: u32,
        version: u16,
        dtype: u8,
        shape_len: u32,
        data_len: u32,
        shard_id: u32,
        layer_id: u32,
    };
};

/// Memory transfer optimization
pub const TransferOptimizer = struct {
    allocator: Allocator,
    chunk_size: usize,
    parallel_transfers: u8,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .chunk_size = 1024 * 1024, // 1MB chunks
            .parallel_transfers = 4,
        };
    }
    
    /// Optimize tensor transfer between VMs
    pub fn optimizeTransfer(self: *Self, tensor: *const DistributedTensor, target_vm: []const u8) !TransferPlan {
        _ = target_vm;
        
        const tensor_size = tensor.data.len * @sizeOf(f32);
        const chunk_count = (tensor_size + self.chunk_size - 1) / self.chunk_size;
        
        var chunks = try self.allocator.alloc(TransferChunk, chunk_count);
        
        for (chunks, 0..) |*chunk, i| {
            const start_offset = i * self.chunk_size;
            const end_offset = @min(start_offset + self.chunk_size, tensor_size);
            
            chunk.* = TransferChunk{
                .chunk_id = @intCast(i),
                .start_offset = start_offset,
                .size = end_offset - start_offset,
                .priority = if (i < self.parallel_transfers) .high else .normal,
            };
        }
        
        return TransferPlan{
            .chunks = chunks,
            .total_size = tensor_size,
            .estimated_time_ms = self.estimateTransferTime(tensor_size),
        };
    }
    
    /// Estimate transfer time
    fn estimateTransferTime(self: *Self, size_bytes: usize) u64 {
        _ = self;
        
        // Assume 100 MB/s network speed
        const network_speed_bps = 100 * 1024 * 1024;
        const transfer_time_ms = (size_bytes * 1000) / network_speed_bps;
        
        return @intCast(transfer_time_ms);
    }
    
    pub const TransferChunk = struct {
        chunk_id: u32,
        start_offset: usize,
        size: usize,
        priority: Priority,
        
        pub const Priority = enum {
            low,
            normal,
            high,
        };
    };
    
    pub const TransferPlan = struct {
        chunks: []TransferChunk,
        total_size: usize,
        estimated_time_ms: u64,
        
        pub fn deinit(self: *TransferPlan, allocator: Allocator) void {
            allocator.free(self.chunks);
        }
    };
};

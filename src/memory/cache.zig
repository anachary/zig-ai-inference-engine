const std = @import("std");

/// Tensor cache for reusing intermediate computation results
pub const TensorCache = struct {
    entries: std.HashMap(CacheKey, CacheEntry, CacheKeyContext, std.hash_map.default_max_load_percentage),
    total_size: usize,
    max_size: usize,
    allocator: std.mem.Allocator,
    
    const CacheKey = struct {
        hash: u64,
        shape: []const usize,
        
        pub fn eql(self: CacheKey, other: CacheKey) bool {
            if (self.hash != other.hash) return false;
            if (self.shape.len != other.shape.len) return false;
            for (self.shape, other.shape) |a, b| {
                if (a != b) return false;
            }
            return true;
        }
    };
    
    const CacheEntry = struct {
        data: []f32,
        shape: []usize,
        last_used: u64,
        size: usize,
        
        pub fn deinit(self: *CacheEntry, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            allocator.free(self.shape);
        }
    };
    
    const CacheKeyContext = struct {
        pub fn hash(self: @This(), key: CacheKey) u64 {
            _ = self;
            return key.hash;
        }
        
        pub fn eql(self: @This(), a: CacheKey, b: CacheKey) bool {
            _ = self;
            return a.eql(b);
        }
    };
    
    pub fn init(allocator: std.mem.Allocator, max_size: usize) !TensorCache {
        return TensorCache{
            .entries = std.HashMap(CacheKey, CacheEntry, CacheKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .total_size = 0,
            .max_size = max_size,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TensorCache) void {
        var iterator = self.entries.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.shape);
        }
        self.entries.deinit();
    }
    
    pub fn getUsed(self: *TensorCache) usize {
        return self.total_size;
    }
    
    pub fn get(self: *TensorCache, shape: []const usize) ?[]f32 {
        const key = self.createKey(shape);
        defer self.allocator.free(key.shape);
        
        if (self.entries.getPtr(key)) |entry| {
            entry.last_used = std.time.timestamp();
            return entry.data;
        }
        
        return null;
    }
    
    pub fn put(self: *TensorCache, shape: []const usize, data: []const f32) !void {
        const size = data.len * @sizeOf(f32);
        
        // Check if we need to evict entries
        while (self.total_size + size > self.max_size and self.entries.count() > 0) {
            try self.evictLRU();
        }
        
        // Don't cache if too large
        if (size > self.max_size) return;
        
        const key = self.createKey(shape);
        const data_copy = try self.allocator.dupe(f32, data);
        
        const entry = CacheEntry{
            .data = data_copy,
            .shape = try self.allocator.dupe(usize, shape),
            .last_used = std.time.timestamp(),
            .size = size,
        };
        
        try self.entries.put(key, entry);
        self.total_size += size;
    }
    
    pub fn remove(self: *TensorCache, shape: []const usize) void {
        const key = self.createKey(shape);
        defer self.allocator.free(key.shape);
        
        if (self.entries.fetchRemove(key)) |kv| {
            self.total_size -= kv.value.size;
            kv.value.deinit(self.allocator);
            self.allocator.free(kv.key.shape);
        }
    }
    
    pub fn clear(self: *TensorCache) void {
        var iterator = self.entries.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.shape);
        }
        self.entries.clearAndFree();
        self.total_size = 0;
    }
    
    fn createKey(self: *TensorCache, shape: []const usize) CacheKey {
        // Compute hash of shape
        var hasher = std.hash.Wyhash.init(0);
        for (shape) |dim| {
            hasher.update(std.mem.asBytes(&dim));
        }
        
        return CacheKey{
            .hash = hasher.final(),
            .shape = self.allocator.dupe(usize, shape) catch unreachable,
        };
    }
    
    fn evictLRU(self: *TensorCache) !void {
        var oldest_key: ?CacheKey = null;
        var oldest_time: u64 = std.math.maxInt(u64);
        
        var iterator = self.entries.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.last_used < oldest_time) {
                oldest_time = entry.value_ptr.last_used;
                oldest_key = entry.key_ptr.*;
            }
        }
        
        if (oldest_key) |key| {
            if (self.entries.fetchRemove(key)) |kv| {
                self.total_size -= kv.value.size;
                kv.value.deinit(self.allocator);
                self.allocator.free(kv.key.shape);
            }
        }
    }
    
    pub fn getStats(self: *TensorCache) CacheStats {
        return CacheStats{
            .entries = self.entries.count(),
            .total_size = self.total_size,
            .max_size = self.max_size,
            .hit_rate = 0.0, // Would need to track hits/misses
        };
    }
};

/// Cache statistics
pub const CacheStats = struct {
    entries: u32,
    total_size: usize,
    max_size: usize,
    hit_rate: f32,
    
    pub fn print(self: CacheStats) void {
        std.log.info("Cache Stats:", .{});
        std.log.info("  Entries: {d}", .{self.entries});
        std.log.info("  Size: {d:.1}MB / {d:.1}MB ({d:.1}%)", .{
            @as(f64, @floatFromInt(self.total_size)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.max_size)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.total_size)) / @as(f64, @floatFromInt(self.max_size)) * 100.0,
        });
        std.log.info("  Hit Rate: {d:.1}%", .{self.hit_rate * 100.0});
    }
};

/// Specialized KV-cache for transformer attention
pub const KVCache = struct {
    keys: [][]f32,
    values: [][]f32,
    current_length: usize,
    max_length: usize,
    num_layers: usize,
    d_model: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(
        allocator: std.mem.Allocator,
        num_layers: usize,
        max_length: usize,
        d_model: usize,
    ) !KVCache {
        var keys = try allocator.alloc([]f32, num_layers);
        var values = try allocator.alloc([]f32, num_layers);
        
        for (0..num_layers) |i| {
            keys[i] = try allocator.alloc(f32, max_length * d_model);
            values[i] = try allocator.alloc(f32, max_length * d_model);
            @memset(keys[i], 0.0);
            @memset(values[i], 0.0);
        }
        
        return KVCache{
            .keys = keys,
            .values = values,
            .current_length = 0,
            .max_length = max_length,
            .num_layers = num_layers,
            .d_model = d_model,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *KVCache) void {
        for (self.keys) |key_layer| {
            self.allocator.free(key_layer);
        }
        for (self.values) |value_layer| {
            self.allocator.free(value_layer);
        }
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
    }
    
    pub fn reset(self: *KVCache) void {
        self.current_length = 0;
        for (self.keys) |key_layer| {
            @memset(key_layer, 0.0);
        }
        for (self.values) |value_layer| {
            @memset(value_layer, 0.0);
        }
    }
    
    pub fn append(
        self: *KVCache,
        layer: usize,
        new_keys: []const f32,
        new_values: []const f32,
    ) !void {
        if (layer >= self.num_layers) return error.LayerIndexOutOfBounds;
        if (new_keys.len != new_values.len) return error.KeyValueLengthMismatch;
        
        const seq_len = new_keys.len / self.d_model;
        if (self.current_length + seq_len > self.max_length) return error.CacheOverflow;
        
        const start_offset = self.current_length * self.d_model;
        const end_offset = start_offset + new_keys.len;
        
        @memcpy(self.keys[layer][start_offset..end_offset], new_keys);
        @memcpy(self.values[layer][start_offset..end_offset], new_values);
        
        if (layer == self.num_layers - 1) {
            // Update length only after processing the last layer
            self.current_length += seq_len;
        }
    }
    
    pub fn getKeys(self: *KVCache, layer: usize) []const f32 {
        if (layer >= self.num_layers) return &[_]f32{};
        const end_offset = self.current_length * self.d_model;
        return self.keys[layer][0..end_offset];
    }
    
    pub fn getValues(self: *KVCache, layer: usize) []const f32 {
        if (layer >= self.num_layers) return &[_]f32{};
        const end_offset = self.current_length * self.d_model;
        return self.values[layer][0..end_offset];
    }
    
    pub fn getCurrentLength(self: *KVCache) usize {
        return self.current_length;
    }
    
    pub fn getUsedMemory(self: *KVCache) usize {
        return self.current_length * self.d_model * self.num_layers * 2 * @sizeOf(f32);
    }
};

test "tensor cache basic" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var cache = try TensorCache.init(allocator, 1024);
    defer cache.deinit();
    
    const shape = [_]usize{ 2, 3 };
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    
    try cache.put(&shape, &data);
    
    if (cache.get(&shape)) |cached_data| {
        try testing.expect(cached_data.len == data.len);
        for (cached_data, data) |cached, original| {
            try testing.expect(cached == original);
        }
    } else {
        try testing.expect(false);
    }
}

test "kv cache basic" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var kv_cache = try KVCache.init(allocator, 2, 10, 4);
    defer kv_cache.deinit();
    
    const keys = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const values = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    
    try kv_cache.append(0, &keys, &values);
    try kv_cache.append(1, &keys, &values);
    
    try testing.expect(kv_cache.getCurrentLength() == 1);
    
    const cached_keys = kv_cache.getKeys(0);
    try testing.expect(cached_keys.len == 4);
    try testing.expect(cached_keys[0] == 1.0);
}

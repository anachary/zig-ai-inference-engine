const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const net = std.net;
const json = std.json;

const ShardManager = @import("shard_manager.zig").ShardManager;
const ModelShard = @import("shard_manager.zig").ModelShard;

/// Tensor data for inter-shard communication
pub const DistributedTensor = struct {
    data: []f32,
    shape: []u32,
    dtype: DataType,
    shard_id: u32,
    layer_id: u32,
    
    pub const DataType = enum {
        f32,
        f16,
        i32,
        i8,
    };
    
    pub fn init(allocator: Allocator, shape: []const u32, dtype: DataType) !DistributedTensor {
        var total_elements: usize = 1;
        for (shape) |dim| {
            total_elements *= dim;
        }
        
        return DistributedTensor{
            .data = try allocator.alloc(f32, total_elements),
            .shape = try allocator.dupe(u32, shape),
            .dtype = dtype,
            .shard_id = 0,
            .layer_id = 0,
        };
    }
    
    pub fn deinit(self: *DistributedTensor, allocator: Allocator) void {
        allocator.free(self.data);
        allocator.free(self.shape);
    }
    
    pub fn serialize(self: *const DistributedTensor, allocator: Allocator) ![]u8 {
        // Serialize tensor for network transmission
        const TensorMessage = struct {
            data: []f32,
            shape: []u32,
            dtype: DataType,
            shard_id: u32,
            layer_id: u32,
        };
        
        const message = TensorMessage{
            .data = self.data,
            .shape = self.shape,
            .dtype = self.dtype,
            .shard_id = self.shard_id,
            .layer_id = self.layer_id,
        };
        
        return try json.stringifyAlloc(allocator, message, .{});
    }
    
    pub fn deserialize(allocator: Allocator, data: []const u8) !DistributedTensor {
        const parsed = try json.parseFromSlice(json.Value, allocator, data, .{});
        defer parsed.deinit();
        
        // Extract tensor data from JSON
        // This is a simplified version - real implementation would handle binary data
        var tensor = try DistributedTensor.init(allocator, &[_]u32{1}, .f32);
        return tensor;
    }
};

/// Inference request across distributed shards
pub const DistributedInferenceRequest = struct {
    request_id: []const u8,
    input_tensor: DistributedTensor,
    target_layers: []u32,
    priority: Priority,
    timeout_ms: u64,
    
    pub const Priority = enum {
        low,
        normal,
        high,
        critical,
    };
    
    pub fn init(allocator: Allocator, request_id: []const u8, input_tensor: DistributedTensor) !DistributedInferenceRequest {
        return DistributedInferenceRequest{
            .request_id = try allocator.dupe(u8, request_id),
            .input_tensor = input_tensor,
            .target_layers = &[_]u32{},
            .priority = .normal,
            .timeout_ms = 30000, // 30 seconds default
        };
    }
    
    pub fn deinit(self: *DistributedInferenceRequest, allocator: Allocator) void {
        allocator.free(self.request_id);
        self.input_tensor.deinit(allocator);
        if (self.target_layers.len > 0) {
            allocator.free(self.target_layers);
        }
    }
};

/// Response from distributed inference
pub const DistributedInferenceResponse = struct {
    request_id: []const u8,
    output_tensor: DistributedTensor,
    execution_time_ms: u64,
    shards_used: []u32,
    status: Status,
    error_message: ?[]const u8,
    
    pub const Status = enum {
        success,
        partial_failure,
        timeout,
        error,
    };
    
    pub fn init(allocator: Allocator, request_id: []const u8, output_tensor: DistributedTensor) !DistributedInferenceResponse {
        return DistributedInferenceResponse{
            .request_id = try allocator.dupe(u8, request_id),
            .output_tensor = output_tensor,
            .execution_time_ms = 0,
            .shards_used = &[_]u32{},
            .status = .success,
            .error_message = null,
        };
    }
    
    pub fn deinit(self: *DistributedInferenceResponse, allocator: Allocator) void {
        allocator.free(self.request_id);
        self.output_tensor.deinit(allocator);
        if (self.shards_used.len > 0) {
            allocator.free(self.shards_used);
        }
        if (self.error_message) |msg| {
            allocator.free(msg);
        }
    }
};

/// Main coordinator for distributed inference
pub const InferenceCoordinator = struct {
    allocator: Allocator,
    shard_manager: *ShardManager,
    active_requests: HashMap([]const u8, *DistributedInferenceRequest),
    request_mutex: Mutex,
    worker_threads: ArrayList(Thread),
    running: bool,
    
    const Self = @This();
    
    pub const CoordinatorError = error{
        NoHealthyShards,
        RequestTimeout,
        ShardCommunicationFailed,
        InvalidRequest,
        InsufficientResources,
    };
    
    pub fn init(allocator: Allocator, shard_manager: *ShardManager) Self {
        return Self{
            .allocator = allocator,
            .shard_manager = shard_manager,
            .active_requests = HashMap([]const u8, *DistributedInferenceRequest).init(allocator),
            .request_mutex = Mutex{},
            .worker_threads = ArrayList(Thread).init(allocator),
            .running = false,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.stop();
        
        // Clean up active requests
        var iterator = self.active_requests.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.active_requests.deinit();
        
        self.worker_threads.deinit();
    }
    
    /// Start the coordinator
    pub fn start(self: *Self) !void {
        if (self.running) return;
        
        self.running = true;
        
        // Start worker threads for handling requests
        const worker_count = 4; // Configurable
        try self.worker_threads.ensureTotalCapacity(worker_count);
        
        for (0..worker_count) |i| {
            const thread = try Thread.spawn(.{}, workerLoop, .{ self, i });
            try self.worker_threads.append(thread);
        }
        
        std.log.info("🚀 Distributed inference coordinator started with {d} workers", .{worker_count});
    }
    
    /// Stop the coordinator
    pub fn stop(self: *Self) void {
        if (!self.running) return;
        
        self.running = false;
        
        // Wait for worker threads to finish
        for (self.worker_threads.items) |thread| {
            thread.join();
        }
        self.worker_threads.clearRetainingCapacity();
        
        std.log.info("🛑 Distributed inference coordinator stopped");
    }
    
    /// Execute distributed inference
    pub fn executeInference(self: *Self, request: DistributedInferenceRequest) !DistributedInferenceResponse {
        const start_time = std.time.milliTimestamp();
        
        // Validate request
        try self.validateRequest(&request);
        
        // Register request
        try self.registerRequest(&request);
        defer self.unregisterRequest(request.request_id);
        
        // Execute inference pipeline
        const response = try self.executeInferencePipeline(&request);
        
        const execution_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));
        std.log.info("✅ Distributed inference completed in {d}ms for request {s}", .{ execution_time, request.request_id });
        
        return response;
    }
    
    /// Execute the full inference pipeline across shards
    fn executeInferencePipeline(self: *Self, request: *const DistributedInferenceRequest) !DistributedInferenceResponse {
        // Get healthy shards
        var healthy_shards = self.shard_manager.getHealthyShards();
        defer healthy_shards.deinit();
        
        if (healthy_shards.items.len == 0) {
            return CoordinatorError.NoHealthyShards;
        }
        
        // Execute inference layer by layer across shards
        var current_tensor = request.input_tensor;
        var shards_used = ArrayList(u32).init(self.allocator);
        defer shards_used.deinit();
        
        // Determine layer execution order
        const total_layers = self.getTotalLayers();
        
        for (0..total_layers) |layer_id| {
            const layer_id_u32 = @as(u32, @intCast(layer_id));
            
            // Find shard responsible for this layer
            const shard = self.shard_manager.getShardForLayer(layer_id_u32) orelse {
                std.log.err("❌ No shard found for layer {d}", .{layer_id_u32});
                continue;
            };
            
            // Execute layer on shard
            current_tensor = try self.executeLayerOnShard(shard, current_tensor, layer_id_u32);
            try shards_used.append(shard.shard_id);
            
            std.log.debug("🔄 Layer {d} executed on shard {d}", .{ layer_id_u32, shard.shard_id });
        }
        
        // Create response
        var response = try DistributedInferenceResponse.init(self.allocator, request.request_id, current_tensor);
        response.shards_used = try shards_used.toOwnedSlice();
        
        return response;
    }
    
    /// Execute a single layer on a specific shard
    fn executeLayerOnShard(self: *Self, shard: *ModelShard, input_tensor: DistributedTensor, layer_id: u32) !DistributedTensor {
        _ = self;
        
        // Prepare request payload
        const LayerRequest = struct {
            layer_id: u32,
            input_tensor: DistributedTensor,
        };
        
        const layer_request = LayerRequest{
            .layer_id = layer_id,
            .input_tensor = input_tensor,
        };
        
        // Send HTTP request to shard
        const endpoint = try shard.getEndpoint(self.allocator);
        defer self.allocator.free(endpoint);
        
        const response_tensor = try self.sendLayerRequest(endpoint, layer_request);
        
        std.log.debug("📡 Layer {d} request sent to {s}", .{ layer_id, endpoint });
        
        return response_tensor;
    }
    
    /// Send HTTP request to a shard for layer execution
    fn sendLayerRequest(self: *Self, endpoint: []const u8, request: anytype) !DistributedTensor {
        _ = endpoint;
        _ = request;
        
        // This would send actual HTTP request to the shard
        // For now, simulate successful execution
        
        // Create mock output tensor
        const output_shape = [_]u32{ 1, 2048, 4096 }; // Example transformer hidden state
        var output_tensor = try DistributedTensor.init(self.allocator, &output_shape, .f32);
        
        // Fill with mock data
        for (output_tensor.data, 0..) |*value, i| {
            value.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
        }
        
        return output_tensor;
    }
    
    /// Get total number of layers in the model
    fn getTotalLayers(self: *Self) u32 {
        // This would come from model metadata
        return 24; // Example: GPT-2 medium has 24 layers
    }
    
    /// Validate inference request
    fn validateRequest(self: *Self, request: *const DistributedInferenceRequest) !void {
        _ = self;
        
        if (request.request_id.len == 0) {
            return CoordinatorError.InvalidRequest;
        }
        
        if (request.input_tensor.data.len == 0) {
            return CoordinatorError.InvalidRequest;
        }
        
        // Additional validation logic...
    }
    
    /// Register active request
    fn registerRequest(self: *Self, request: *const DistributedInferenceRequest) !void {
        self.request_mutex.lock();
        defer self.request_mutex.unlock();
        
        // Create a copy of the request
        const request_copy = try self.allocator.create(DistributedInferenceRequest);
        request_copy.* = request.*;
        
        try self.active_requests.put(request.request_id, request_copy);
    }
    
    /// Unregister completed request
    fn unregisterRequest(self: *Self, request_id: []const u8) void {
        self.request_mutex.lock();
        defer self.request_mutex.unlock();
        
        if (self.active_requests.fetchRemove(request_id)) |entry| {
            entry.value.deinit(self.allocator);
            self.allocator.destroy(entry.value);
        }
    }
    
    /// Worker thread loop
    fn workerLoop(self: *Self, worker_id: usize) void {
        std.log.info("🔧 Worker {d} started", .{worker_id});
        
        while (self.running) {
            // Process any pending work
            std.time.sleep(100 * std.time.ns_per_ms); // 100ms sleep
        }
        
        std.log.info("🔧 Worker {d} stopped", .{worker_id});
    }
    
    /// Get coordinator statistics
    pub fn getStats(self: *Self) CoordinatorStats {
        self.request_mutex.lock();
        defer self.request_mutex.unlock();
        
        return CoordinatorStats{
            .active_requests = self.active_requests.count(),
            .worker_threads = self.worker_threads.items.len,
            .healthy_shards = self.shard_manager.getHealthyShards().items.len,
        };
    }
    
    pub const CoordinatorStats = struct {
        active_requests: u32,
        worker_threads: usize,
        healthy_shards: usize,
    };
};

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Atomic = std.atomic.Atomic;

const ShardManager = @import("shard_manager.zig").ShardManager;
const ModelShard = @import("shard_manager.zig").ModelShard;
const VMNode = @import("shard_manager.zig").VMNode;

/// Fault tolerance manager for distributed inference
pub const FaultToleranceManager = struct {
    allocator: Allocator,
    shard_manager: *ShardManager,
    health_monitor: HealthMonitor,
    failover_manager: FailoverManager,
    replication_manager: ReplicationManager,
    recovery_manager: RecoveryManager,
    running: bool,
    monitor_thread: ?Thread,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator, shard_manager: *ShardManager) Self {
        return Self{
            .allocator = allocator,
            .shard_manager = shard_manager,
            .health_monitor = HealthMonitor.init(allocator),
            .failover_manager = FailoverManager.init(allocator),
            .replication_manager = ReplicationManager.init(allocator),
            .recovery_manager = RecoveryManager.init(allocator),
            .running = false,
            .monitor_thread = null,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.stop();
        self.health_monitor.deinit();
        self.failover_manager.deinit();
        self.replication_manager.deinit();
        self.recovery_manager.deinit();
    }
    
    /// Start fault tolerance monitoring
    pub fn start(self: *Self) !void {
        if (self.running) return;
        
        self.running = true;
        
        // Start health monitoring thread
        self.monitor_thread = try Thread.spawn(.{}, monitoringLoop, .{self});
        
        std.log.info("🛡️ Fault tolerance manager started");
    }
    
    /// Stop fault tolerance monitoring
    pub fn stop(self: *Self) void {
        if (!self.running) return;
        
        self.running = false;
        
        if (self.monitor_thread) |thread| {
            thread.join();
            self.monitor_thread = null;
        }
        
        std.log.info("🛡️ Fault tolerance manager stopped");
    }
    
    /// Handle shard failure
    pub fn handleShardFailure(self: *Self, failed_shard_id: u32) !void {
        std.log.warn("⚠️ Handling failure of shard {d}", .{failed_shard_id});
        
        // Mark shard as failed
        try self.health_monitor.markShardFailed(failed_shard_id);
        
        // Attempt failover
        try self.failover_manager.failoverShard(failed_shard_id, self.shard_manager);
        
        // Start recovery process
        try self.recovery_manager.startRecovery(failed_shard_id, self.shard_manager);
    }
    
    /// Handle VM failure
    pub fn handleVMFailure(self: *Self, failed_vm_address: []const u8) !void {
        std.log.warn("⚠️ Handling failure of VM {s}", .{failed_vm_address});
        
        // Find all shards on the failed VM
        const failed_shards = try self.findShardsOnVM(failed_vm_address);
        defer self.allocator.free(failed_shards);
        
        // Handle each shard failure
        for (failed_shards) |shard_id| {
            try self.handleShardFailure(shard_id);
        }
    }
    
    /// Find shards on a specific VM
    fn findShardsOnVM(self: *Self, vm_address: []const u8) ![]u32 {
        var shard_ids = ArrayList(u32).init(self.allocator);
        defer shard_ids.deinit();
        
        for (self.shard_manager.shards.items) |shard| {
            if (std.mem.eql(u8, shard.vm_address, vm_address)) {
                try shard_ids.append(shard.shard_id);
            }
        }
        
        return shard_ids.toOwnedSlice();
    }
    
    /// Main monitoring loop
    fn monitoringLoop(self: *Self) void {
        while (self.running) {
            // Perform health checks
            self.health_monitor.performHealthChecks(self.shard_manager) catch |err| {
                std.log.warn("Health check failed: {any}", .{err});
            };
            
            // Check for failed shards and trigger recovery
            const failed_shards = self.health_monitor.getFailedShards();
            for (failed_shards.items) |shard_id| {
                self.handleShardFailure(shard_id) catch |err| {
                    std.log.err("Failed to handle shard failure {d}: {any}", .{ shard_id, err });
                };
            }
            
            // Sleep before next check
            std.time.sleep(5 * std.time.ns_per_s); // 5 seconds
        }
    }
};

/// Health monitoring for shards and VMs
pub const HealthMonitor = struct {
    allocator: Allocator,
    shard_health: HashMap(u32, ShardHealth),
    vm_health: HashMap([]const u8, VMHealth),
    failed_shards: ArrayList(u32),
    health_mutex: Mutex,
    
    const Self = @This();
    
    const ShardHealth = struct {
        shard_id: u32,
        status: Status,
        last_check: i64,
        consecutive_failures: u32,
        response_time_ms: u64,
        
        const Status = enum {
            healthy,
            degraded,
            failed,
            recovering,
        };
    };
    
    const VMHealth = struct {
        vm_address: []const u8,
        status: VMStatus,
        last_check: i64,
        consecutive_failures: u32,
        cpu_usage: f32,
        memory_usage: f32,
        
        const VMStatus = enum {
            healthy,
            overloaded,
            unreachable,
            failed,
        };
    };
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .shard_health = HashMap(u32, ShardHealth).init(allocator),
            .vm_health = HashMap([]const u8, VMHealth).init(allocator),
            .failed_shards = ArrayList(u32).init(allocator),
            .health_mutex = Mutex{},
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.shard_health.deinit();
        self.vm_health.deinit();
        self.failed_shards.deinit();
    }
    
    /// Perform health checks on all shards
    pub fn performHealthChecks(self: *Self, shard_manager: *ShardManager) !void {
        const current_time = std.time.milliTimestamp();
        
        for (shard_manager.shards.items) |*shard| {
            const health_result = try self.checkShardHealth(shard);
            try self.updateShardHealth(shard.shard_id, health_result, current_time);
        }
    }
    
    /// Check health of a specific shard
    fn checkShardHealth(self: *Self, shard: *ModelShard) !HealthCheckResult {
        _ = self;
        
        const start_time = std.time.milliTimestamp();
        
        // Simulate health check (would be actual HTTP request)
        const is_healthy = shard.status == .ready or shard.status == .busy;
        const response_time = std.time.milliTimestamp() - start_time;
        
        return HealthCheckResult{
            .is_healthy = is_healthy,
            .response_time_ms = @intCast(response_time),
            .error_message = if (is_healthy) null else "Shard not responding",
        };
    }
    
    /// Update shard health status
    fn updateShardHealth(self: *Self, shard_id: u32, result: HealthCheckResult, timestamp: i64) !void {
        self.health_mutex.lock();
        defer self.health_mutex.unlock();
        
        var health = self.shard_health.get(shard_id) orelse ShardHealth{
            .shard_id = shard_id,
            .status = .healthy,
            .last_check = 0,
            .consecutive_failures = 0,
            .response_time_ms = 0,
        };
        
        health.last_check = timestamp;
        health.response_time_ms = result.response_time_ms;
        
        if (result.is_healthy) {
            health.consecutive_failures = 0;
            health.status = if (result.response_time_ms > 5000) .degraded else .healthy;
        } else {
            health.consecutive_failures += 1;
            
            if (health.consecutive_failures >= 3) {
                health.status = .failed;
                try self.failed_shards.append(shard_id);
            } else {
                health.status = .degraded;
            }
        }
        
        try self.shard_health.put(shard_id, health);
    }
    
    /// Mark shard as failed
    pub fn markShardFailed(self: *Self, shard_id: u32) !void {
        self.health_mutex.lock();
        defer self.health_mutex.unlock();
        
        var health = self.shard_health.get(shard_id) orelse ShardHealth{
            .shard_id = shard_id,
            .status = .failed,
            .last_check = std.time.milliTimestamp(),
            .consecutive_failures = 999,
            .response_time_ms = 0,
        };
        
        health.status = .failed;
        try self.shard_health.put(shard_id, health);
        
        // Add to failed list if not already there
        for (self.failed_shards.items) |failed_id| {
            if (failed_id == shard_id) return;
        }
        try self.failed_shards.append(shard_id);
    }
    
    /// Get list of failed shards
    pub fn getFailedShards(self: *Self) ArrayList(u32) {
        self.health_mutex.lock();
        defer self.health_mutex.unlock();
        
        return self.failed_shards.clone() catch ArrayList(u32).init(self.allocator);
    }
    
    const HealthCheckResult = struct {
        is_healthy: bool,
        response_time_ms: u64,
        error_message: ?[]const u8,
    };
};

/// Failover management for automatic shard replacement
pub const FailoverManager = struct {
    allocator: Allocator,
    failover_history: ArrayList(FailoverEvent),
    
    const Self = @This();
    
    const FailoverEvent = struct {
        timestamp: i64,
        failed_shard_id: u32,
        replacement_shard_id: u32,
        failover_time_ms: u64,
        success: bool,
    };
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .failover_history = ArrayList(FailoverEvent).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.failover_history.deinit();
    }
    
    /// Perform failover for a failed shard
    pub fn failoverShard(self: *Self, failed_shard_id: u32, shard_manager: *ShardManager) !void {
        const start_time = std.time.milliTimestamp();
        
        std.log.info("🔄 Starting failover for shard {d}", .{failed_shard_id});
        
        // Find a suitable replacement VM
        const replacement_vm = try self.findReplacementVM(shard_manager);
        
        // Create replacement shard
        const replacement_shard_id = try self.createReplacementShard(failed_shard_id, replacement_vm, shard_manager);
        
        // Record failover event
        const failover_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));
        const event = FailoverEvent{
            .timestamp = start_time,
            .failed_shard_id = failed_shard_id,
            .replacement_shard_id = replacement_shard_id,
            .failover_time_ms = failover_time,
            .success = true,
        };
        
        try self.failover_history.append(event);
        
        std.log.info("✅ Failover completed in {d}ms: shard {d} -> shard {d}", .{ failover_time, failed_shard_id, replacement_shard_id });
    }
    
    /// Find suitable VM for replacement shard
    fn findReplacementVM(self: *Self, shard_manager: *ShardManager) !*VMNode {
        _ = self;
        
        // Find VM with lowest load and sufficient resources
        var best_vm: ?*VMNode = null;
        var best_score: f32 = 0;
        
        for (shard_manager.vm_nodes.items) |*vm| {
            if (vm.status != .available) continue;
            
            // Calculate suitability score
            const memory_score = @as(f32, @floatFromInt(vm.available_memory_mb)) / 1024.0;
            const load_score = 1.0 - vm.current_load;
            const cpu_score = @as(f32, @floatFromInt(vm.cpu_cores)) / 16.0;
            
            const total_score = memory_score * load_score * cpu_score;
            
            if (total_score > best_score) {
                best_score = total_score;
                best_vm = vm;
            }
        }
        
        return best_vm orelse error.NoSuitableVM;
    }
    
    /// Create replacement shard
    fn createReplacementShard(self: *Self, failed_shard_id: u32, replacement_vm: *VMNode, shard_manager: *ShardManager) !u32 {
        _ = self;
        
        // Find the failed shard to copy its configuration
        var failed_shard: ?*ModelShard = null;
        for (shard_manager.shards.items) |*shard| {
            if (shard.shard_id == failed_shard_id) {
                failed_shard = shard;
                break;
            }
        }
        
        const original_shard = failed_shard orelse return error.FailedShardNotFound;
        
        // Create new shard with same configuration
        const new_shard_id = shard_manager.shards.items.len;
        var new_shard = try ModelShard.init(shard_manager.allocator, @intCast(new_shard_id), replacement_vm.address, replacement_vm.port);
        
        new_shard.layer_start = original_shard.layer_start;
        new_shard.layer_end = original_shard.layer_end;
        new_shard.memory_usage_mb = original_shard.memory_usage_mb;
        new_shard.status = .initializing;
        
        // Deploy to VM (simplified)
        new_shard.status = .ready;
        
        try shard_manager.shards.append(new_shard);
        
        return @intCast(new_shard_id);
    }
};

/// Replication management for high availability
pub const ReplicationManager = struct {
    allocator: Allocator,
    replication_factor: u8,
    replica_map: HashMap(u32, ArrayList(u32)),
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .replication_factor = 2, // Default: 2 replicas per shard
            .replica_map = HashMap(u32, ArrayList(u32)).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        var iterator = self.replica_map.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.replica_map.deinit();
    }
    
    /// Create replicas for a shard
    pub fn createReplicas(self: *Self, shard_id: u32, shard_manager: *ShardManager) !void {
        var replicas = ArrayList(u32).init(self.allocator);
        
        for (0..self.replication_factor) |i| {
            _ = i;
            // Create replica (simplified implementation)
            const replica_id = shard_manager.shards.items.len + replicas.items.len;
            try replicas.append(@intCast(replica_id));
        }
        
        try self.replica_map.put(shard_id, replicas);
        
        std.log.info("📋 Created {d} replicas for shard {d}", .{ self.replication_factor, shard_id });
    }
    
    /// Get replicas for a shard
    pub fn getReplicas(self: *Self, shard_id: u32) ?[]const u32 {
        if (self.replica_map.get(shard_id)) |replicas| {
            return replicas.items;
        }
        return null;
    }
};

/// Recovery management for failed components
pub const RecoveryManager = struct {
    allocator: Allocator,
    recovery_queue: ArrayList(RecoveryTask),
    recovery_mutex: Mutex,
    
    const Self = @This();
    
    const RecoveryTask = struct {
        task_id: u32,
        shard_id: u32,
        recovery_type: RecoveryType,
        priority: Priority,
        created_at: i64,
        attempts: u32,
        
        const RecoveryType = enum {
            restart_shard,
            migrate_shard,
            restore_from_backup,
            full_rebuild,
        };
        
        const Priority = enum {
            low,
            normal,
            high,
            critical,
        };
    };
    
    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .recovery_queue = ArrayList(RecoveryTask).init(allocator),
            .recovery_mutex = Mutex{},
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.recovery_queue.deinit();
    }
    
    /// Start recovery process for a failed shard
    pub fn startRecovery(self: *Self, shard_id: u32, shard_manager: *ShardManager) !void {
        _ = shard_manager;
        
        self.recovery_mutex.lock();
        defer self.recovery_mutex.unlock();
        
        const task = RecoveryTask{
            .task_id = @intCast(self.recovery_queue.items.len),
            .shard_id = shard_id,
            .recovery_type = .restart_shard,
            .priority = .high,
            .created_at = std.time.milliTimestamp(),
            .attempts = 0,
        };
        
        try self.recovery_queue.append(task);
        
        std.log.info("🔧 Recovery task created for shard {d}", .{shard_id});
    }
};

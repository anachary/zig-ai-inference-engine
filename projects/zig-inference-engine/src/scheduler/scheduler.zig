const std = @import("std");
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;

// Import common interfaces
const TensorInterface = @import("../../../common/interfaces/tensor.zig").TensorInterface;

/// Task types for the scheduler
pub const TaskType = enum {
    operator_execution,
    memory_transfer,
    synchronization,
    cleanup,
};

/// Task priority levels
pub const Priority = enum {
    low,
    normal,
    high,
    critical,
    
    pub fn toValue(self: Priority) u8 {
        return switch (self) {
            .low => 0,
            .normal => 1,
            .high => 2,
            .critical => 3,
        };
    }
};

/// Task status
pub const TaskStatus = enum {
    pending,
    running,
    completed,
    failed,
    cancelled,
};

/// Task execution context
pub const TaskContext = struct {
    task_id: u64,
    task_type: TaskType,
    priority: Priority,
    status: TaskStatus,
    dependencies: std.ArrayList(u64),
    inputs: []const TensorInterface,
    outputs: []TensorInterface,
    operator_name: []const u8,
    attributes: std.StringHashMap([]const u8),
    start_time: i64,
    end_time: i64,
    error_info: ?[]const u8,
    
    pub fn init(allocator: Allocator, task_id: u64, task_type: TaskType) TaskContext {
        return TaskContext{
            .task_id = task_id,
            .task_type = task_type,
            .priority = .normal,
            .status = .pending,
            .dependencies = std.ArrayList(u64).init(allocator),
            .inputs = &[_]TensorInterface{},
            .outputs = &[_]TensorInterface{},
            .operator_name = "",
            .attributes = std.StringHashMap([]const u8).init(allocator),
            .start_time = 0,
            .end_time = 0,
            .error_info = null,
        };
    }
    
    pub fn deinit(self: *TaskContext) void {
        self.dependencies.deinit();
        self.attributes.deinit();
    }
};

/// Task execution function signature
pub const TaskExecuteFn = *const fn (context: *TaskContext, allocator: Allocator) anyerror!void;

/// Task definition
pub const Task = struct {
    context: TaskContext,
    execute_fn: TaskExecuteFn,
    
    pub fn init(context: TaskContext, execute_fn: TaskExecuteFn) Task {
        return Task{
            .context = context,
            .execute_fn = execute_fn,
        };
    }
};

/// Worker thread state
pub const WorkerState = enum {
    idle,
    running,
    stopping,
    stopped,
};

/// Worker thread
pub const Worker = struct {
    id: u32,
    thread: ?Thread,
    state: WorkerState,
    scheduler: *TaskScheduler,
    
    pub fn init(id: u32, scheduler: *TaskScheduler) Worker {
        return Worker{
            .id = id,
            .thread = null,
            .state = .idle,
            .scheduler = scheduler,
        };
    }
    
    pub fn start(self: *Worker) !void {
        self.state = .running;
        self.thread = try Thread.spawn(.{}, workerLoop, .{self});
    }
    
    pub fn stop(self: *Worker) void {
        self.state = .stopping;
        if (self.thread) |thread| {
            thread.join();
            self.thread = null;
        }
        self.state = .stopped;
    }
    
    fn workerLoop(self: *Worker) void {
        while (self.state == .running) {
            if (self.scheduler.getNextTask()) |task| {
                self.executeTask(task);
            } else {
                // No tasks available, wait a bit
                std.time.sleep(1_000_000); // 1ms
            }
        }
    }
    
    fn executeTask(self: *Worker, task: *Task) void {
        task.context.status = .running;
        task.context.start_time = std.time.nanoTimestamp();
        
        task.execute_fn(&task.context, self.scheduler.allocator) catch |err| {
            task.context.status = .failed;
            task.context.error_info = @errorName(err);
            std.log.err("Task {} failed: {}", .{ task.context.task_id, err });
            return;
        };
        
        task.context.end_time = std.time.nanoTimestamp();
        task.context.status = .completed;
        
        // Notify scheduler that task is complete
        self.scheduler.onTaskComplete(task);
    }
};

/// Scheduler statistics
pub const SchedulerStats = struct {
    total_tasks: u64,
    completed_tasks: u64,
    failed_tasks: u64,
    pending_tasks: u64,
    running_tasks: u64,
    average_execution_time_ms: f32,
    worker_utilization: f32,
    queue_depth: usize,
};

/// Multi-threaded task scheduler
pub const TaskScheduler = struct {
    allocator: Allocator,
    workers: std.ArrayList(Worker),
    task_queue: std.PriorityQueue(Task, void, taskCompareFn),
    completed_tasks: std.ArrayList(Task),
    task_counter: u64,
    mutex: Mutex,
    condition: Condition,
    stats: SchedulerStats,
    max_workers: u32,
    
    const Self = @This();

    /// Initialize the task scheduler
    pub fn init(allocator: Allocator, num_threads: ?u32) !Self {
        const worker_count = num_threads orelse @max(1, std.Thread.getCpuCount() catch 4);
        
        var self = Self{
            .allocator = allocator,
            .workers = std.ArrayList(Worker).init(allocator),
            .task_queue = std.PriorityQueue(Task, void, taskCompareFn).init(allocator, {}),
            .completed_tasks = std.ArrayList(Task).init(allocator),
            .task_counter = 0,
            .mutex = Mutex{},
            .condition = Condition{},
            .stats = std.mem.zeroes(SchedulerStats),
            .max_workers = worker_count,
        };
        
        // Create worker threads
        try self.workers.ensureTotalCapacity(worker_count);
        for (0..worker_count) |i| {
            var worker = Worker.init(@intCast(i), &self);
            try self.workers.append(worker);
        }
        
        return self;
    }

    /// Deinitialize the scheduler
    pub fn deinit(self: *Self) void {
        // Stop all workers
        for (self.workers.items) |*worker| {
            worker.stop();
        }
        
        // Clean up tasks
        while (self.task_queue.removeOrNull()) |task| {
            var mutable_task = task;
            mutable_task.context.deinit();
        }
        
        for (self.completed_tasks.items) |*task| {
            task.context.deinit();
        }
        
        self.workers.deinit();
        self.task_queue.deinit();
        self.completed_tasks.deinit();
    }

    /// Start the scheduler
    pub fn start(self: *Self) !void {
        for (self.workers.items) |*worker| {
            try worker.start();
        }
        std.log.info("Task scheduler started with {} workers", .{self.workers.items.len});
    }

    /// Stop the scheduler
    pub fn stop(self: *Self) void {
        for (self.workers.items) |*worker| {
            worker.stop();
        }
        std.log.info("Task scheduler stopped", .{});
    }

    /// Submit a task for execution
    pub fn submitTask(self: *Self, task: Task) !u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var mutable_task = task;
        mutable_task.context.task_id = self.task_counter;
        self.task_counter += 1;
        
        try self.task_queue.add(mutable_task);
        self.stats.total_tasks += 1;
        self.stats.pending_tasks += 1;
        
        // Notify workers
        self.condition.signal();
        
        return mutable_task.context.task_id;
    }

    /// Get the next task from the queue
    pub fn getNextTask(self: *Self) ?*Task {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.task_queue.removeOrNull()) |task| {
            self.stats.pending_tasks -= 1;
            self.stats.running_tasks += 1;
            
            // Return a pointer to the task (this is a simplified approach)
            // In a real implementation, you'd want better memory management
            var heap_task = self.allocator.create(Task) catch return null;
            heap_task.* = task;
            return heap_task;
        }
        
        return null;
    }

    /// Called when a task completes
    pub fn onTaskComplete(self: *Self, task: *Task) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.stats.running_tasks -= 1;
        
        if (task.context.status == .completed) {
            self.stats.completed_tasks += 1;
        } else if (task.context.status == .failed) {
            self.stats.failed_tasks += 1;
        }
        
        // Update average execution time
        const execution_time_ms = @as(f32, @floatFromInt(task.context.end_time - task.context.start_time)) / 1_000_000.0;
        if (self.stats.completed_tasks > 0) {
            self.stats.average_execution_time_ms = 
                (self.stats.average_execution_time_ms * @as(f32, @floatFromInt(self.stats.completed_tasks - 1)) + execution_time_ms) / 
                @as(f32, @floatFromInt(self.stats.completed_tasks));
        } else {
            self.stats.average_execution_time_ms = execution_time_ms;
        }
        
        // Store completed task
        self.completed_tasks.append(task.*) catch {};
        
        // Free the heap-allocated task
        self.allocator.destroy(task);
    }

    /// Get scheduler statistics
    pub fn getStats(self: *const Self) SchedulerStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var stats = self.stats;
        stats.queue_depth = self.task_queue.len;
        
        // Calculate worker utilization
        const active_workers = @as(f32, @floatFromInt(stats.running_tasks));
        const total_workers = @as(f32, @floatFromInt(self.workers.items.len));
        stats.worker_utilization = if (total_workers > 0) active_workers / total_workers else 0.0;
        
        return stats;
    }

    /// Wait for all tasks to complete
    pub fn waitForCompletion(self: *Self) void {
        while (true) {
            self.mutex.lock();
            const has_pending = self.stats.pending_tasks > 0 or self.stats.running_tasks > 0;
            self.mutex.unlock();
            
            if (!has_pending) break;
            
            std.time.sleep(1_000_000); // 1ms
        }
    }
};

/// Task comparison function for priority queue
fn taskCompareFn(context: void, a: Task, b: Task) std.math.Order {
    _ = context;
    
    // Higher priority tasks come first
    const priority_cmp = std.math.order(b.context.priority.toValue(), a.context.priority.toValue());
    if (priority_cmp != .eq) return priority_cmp;
    
    // If same priority, earlier tasks come first
    return std.math.order(a.context.task_id, b.context.task_id);
}

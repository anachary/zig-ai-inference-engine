const std = @import("std");

/// GPU support module for Zig AI Interface Engine
///
/// This module provides a lightweight GPU abstraction layer designed for:
/// - IoT devices with limited GPU resources
/// - Data security applications requiring efficient inference
/// - Lightweight LLM model execution
///
/// Supported backends:
/// - CPU fallback (always available)
/// - CUDA (when available)
/// - Vulkan Compute (cross-platform)
/// - OpenCL (future support)

// Re-export all GPU functionality
pub const device = @import("device.zig");
pub const memory = @import("memory.zig");
pub const kernels = @import("kernels.zig");

// Re-export key types for convenience
pub const GPUDevice = device.GPUDevice;
pub const DeviceType = device.DeviceType;
pub const DeviceCapabilities = device.DeviceCapabilities;
pub const ModelSizeCategory = device.ModelSizeCategory;
pub const GPUBuffer = memory.GPUBuffer;
pub const GPUMemoryPool = memory.GPUMemoryPool;
pub const MemoryType = memory.MemoryType;
pub const KernelExecutor = kernels.KernelExecutor;
pub const KernelType = kernels.KernelType;
pub const KernelParams = kernels.KernelParams;

// Re-export errors
pub const DeviceError = device.DeviceError;
pub const MemoryError = memory.MemoryError;
pub const KernelError = kernels.KernelError;

const Allocator = std.mem.Allocator;

/// Integrated GPU compute context for lightweight inference
pub const GPUContext = struct {
    allocator: Allocator,
    device: GPUDevice,
    memory_pool: memory.GPUMemoryPool,
    kernel_executor: kernels.KernelExecutor,
    is_initialized: bool,

    const Self = @This();

    /// Initialize GPU context with automatic device selection
    pub fn init(allocator: Allocator) !Self {
        std.log.info("🚀 Initializing GPU Context for Lightweight Inference", .{});

        // Initialize GPU device
        var gpu_device = try GPUDevice.init(allocator);
        errdefer gpu_device.deinit();

        // Check if device is suitable for lightweight inference
        if (!gpu_device.supportsLightweightInference()) {
            std.log.warn("Selected GPU device may not be optimal for lightweight inference", .{});
        }

        // Check IoT suitability
        if (gpu_device.isIoTSuitable()) {
            std.log.info("✅ Device is suitable for IoT deployment", .{});
        } else {
            std.log.info("⚠️  Device may not be optimal for IoT constraints", .{});
        }

        var context = Self{
            .allocator = allocator,
            .device = gpu_device,
            .memory_pool = undefined,
            .kernel_executor = undefined,
            .is_initialized = true,
        };

        // Initialize memory pool with reference to device in context
        context.memory_pool = memory.GPUMemoryPool.init(allocator, &context.device);

        // Initialize kernel executor with reference to device in context
        context.kernel_executor = kernels.KernelExecutor.init(allocator, &context.device, &context.memory_pool);

        // Pre-compile essential kernels for lightweight inference
        try context.precompileEssentialKernels();

        std.log.info("✅ GPU Context initialized successfully", .{});
        return context;
    }

    /// Initialize GPU context with specific device selection
    pub fn initWithDevice(allocator: Allocator, device_caps: DeviceCapabilities) !Self {
        std.log.info("🎯 Initializing GPU Context with specific device: {s}", .{device_caps.name});

        // Create device with specific capabilities
        var gpu_device = GPUDevice{
            .allocator = allocator,
            .capabilities = device_caps,
            .is_initialized = false,
            .context = null,
        };

        try gpu_device.initializeDevice();
        errdefer gpu_device.deinit();

        var context = Self{
            .allocator = allocator,
            .device = gpu_device,
            .memory_pool = undefined,
            .kernel_executor = undefined,
            .is_initialized = true,
        };

        // Initialize memory pool and kernel executor with references to device in context
        context.memory_pool = memory.GPUMemoryPool.init(allocator, &context.device);
        context.kernel_executor = kernels.KernelExecutor.init(allocator, &context.device, &context.memory_pool);

        try context.precompileEssentialKernels();

        return context;
    }

    /// Deinitialize GPU context and cleanup all resources
    pub fn deinit(self: *Self) void {
        if (self.is_initialized) {
            std.log.info("🧹 Cleaning up GPU Context", .{});

            self.kernel_executor.deinit();
            self.memory_pool.deinit();
            self.device.deinit();

            self.is_initialized = false;
            std.log.info("✅ GPU Context cleanup complete", .{});
        }
    }

    /// Allocate GPU buffer for tensor data
    pub fn allocateBuffer(self: *Self, size: usize, memory_type: MemoryType) !GPUBuffer {
        return self.memory_pool.allocate(size, memory_type);
    }

    /// Free GPU buffer
    pub fn freeBuffer(self: *Self, buffer: GPUBuffer) !void {
        return self.memory_pool.free(buffer);
    }

    /// Execute operator on GPU
    pub fn executeOperator(
        self: *Self,
        op_name: []const u8,
        inputs: []GPUBuffer,
        outputs: []GPUBuffer,
    ) !void {
        return self.kernel_executor.executeOperator(op_name, inputs, outputs);
    }

    /// Execute specific kernel with custom parameters
    pub fn executeKernel(
        self: *Self,
        kernel_type: KernelType,
        inputs: []GPUBuffer,
        outputs: []GPUBuffer,
        params: KernelParams,
    ) !void {
        return self.kernel_executor.executeKernel(kernel_type, inputs, outputs, params);
    }

    /// Get device information
    pub fn getDeviceInfo(self: *const Self) DeviceCapabilities {
        return self.device.capabilities;
    }

    /// Get memory usage statistics
    pub fn getMemoryStats(self: *const Self) struct {
        device_memory: struct { total: usize, available: usize, used: usize },
        pool_stats: struct { total_allocated: usize, peak_usage: usize, free_blocks: usize, allocated_blocks: usize },
    } {
        const device_info = self.device.getMemoryInfo();
        const pool_info = self.memory_pool.getStats();

        return .{
            .device_memory = .{
                .total = device_info.total,
                .available = device_info.available,
                .used = device_info.used,
            },
            .pool_stats = .{
                .total_allocated = pool_info.total_allocated,
                .peak_usage = pool_info.peak_usage,
                .free_blocks = pool_info.free_blocks,
                .allocated_blocks = pool_info.allocated_blocks,
            },
        };
    }

    /// Check if context is ready for inference
    pub fn isReadyForInference(self: *const Self) bool {
        return self.is_initialized and
            self.device.supportsLightweightInference() and
            self.memory_pool.getStats().total_allocated < self.device.capabilities.memory_total / 2;
    }

    /// Get recommended memory type for tensor usage
    pub fn getRecommendedMemoryType(self: *const Self, is_input: bool, is_output: bool) MemoryType {
        return memory.selectOptimalMemoryType(self.device.capabilities, is_input, is_output);
    }

    /// Private: Pre-compile essential kernels for lightweight inference
    fn precompileEssentialKernels(self: *Self) !void {
        std.log.info("🔧 Pre-compiling essential kernels for lightweight inference...", .{});

        const essential_kernels = [_]KernelType{
            .vector_add,
            .vector_mul,
            .matrix_multiply,
            .relu_activation,
            .softmax,
            .quantize_int8,
            .dequantize_int8,
        };

        for (essential_kernels) |kernel_type| {
            self.kernel_executor.compileKernel(kernel_type) catch |err| {
                std.log.warn("Failed to compile kernel {s}: {}", .{ @tagName(kernel_type), err });
                // Continue with other kernels
            };
        }

        std.log.info("✅ Essential kernels compiled", .{});
    }
};

/// Utility functions for GPU context management
/// Create GPU context with automatic best device selection
pub fn createOptimalContext(allocator: Allocator) !GPUContext {
    return GPUContext.init(allocator);
}

/// Create GPU context for IoT deployment (prioritizes low memory usage)
pub fn createIoTContext(allocator: Allocator) !GPUContext {
    const devices = try device.enumerateDevices(allocator);
    defer allocator.free(devices);

    // Find the most suitable device for IoT
    var best_device: ?DeviceCapabilities = null;
    var best_score: f32 = 0.0;

    for (devices) |dev| {
        // Create a temporary device to check IoT suitability
        const temp_device = GPUDevice{
            .allocator = allocator,
            .capabilities = dev,
            .is_initialized = false,
            .context = null,
        };

        if (!temp_device.isIoTSuitable()) continue;

        var score: f32 = 0.0;

        // Prefer lower memory usage for IoT
        const memory_gb = @as(f32, @floatFromInt(dev.memory_total)) / (1024.0 * 1024.0 * 1024.0);
        score += (4.0 - memory_gb) * 10.0; // Prefer smaller memory footprint

        // Prefer unified memory for simplicity
        if (dev.supports_unified_memory) score += 20.0;

        // Prefer quantization support for efficiency
        if (dev.supports_int8) score += 15.0;
        if (dev.supports_fp16) score += 10.0;

        // Device type preference for IoT
        switch (dev.device_type) {
            .cpu => score += 15.0, // CPU is always available and power-efficient
            .vulkan => score += 25.0, // Vulkan is cross-platform and efficient
            .cuda => score += 10.0, // CUDA is powerful but may not be IoT-friendly
            .opencl => score += 20.0,
        }

        if (score > best_score) {
            best_score = score;
            best_device = dev;
        }
    }

    if (best_device) |dev| {
        std.log.info("🎯 Selected IoT-optimized device: {s}", .{dev.name});
        return GPUContext.initWithDevice(allocator, dev);
    } else {
        std.log.warn("⚠️  No IoT-suitable device found, using default selection", .{});
        return GPUContext.init(allocator);
    }
}

/// Create GPU context for data security applications (prioritizes reliability)
pub fn createSecurityContext(allocator: Allocator) !GPUContext {
    const devices = try device.enumerateDevices(allocator);
    defer allocator.free(devices);

    // For security applications, prefer stable and well-tested backends
    for (devices) |dev| {
        switch (dev.device_type) {
            .cpu => {
                // CPU is most reliable for security applications
                std.log.info("🔒 Selected security-optimized device: CPU (most reliable)", .{});
                return GPUContext.initWithDevice(allocator, dev);
            },
            else => continue,
        }
    }

    // Fallback to default selection
    std.log.warn("⚠️  Using default device selection for security context", .{});
    return GPUContext.init(allocator);
}

/// Get system GPU capabilities summary
pub fn getSystemCapabilities(allocator: Allocator) !struct {
    total_devices: usize,
    iot_suitable_devices: usize,
    inference_capable_devices: usize,
    total_memory_gb: f32,
    supports_quantization: bool,
} {
    const devices = try device.enumerateDevices(allocator);
    defer allocator.free(devices);

    var iot_suitable: usize = 0;
    var inference_capable: usize = 0;
    var total_memory: usize = 0;
    var supports_quant = false;

    for (devices) |dev| {
        // Create a temporary device to check capabilities
        const temp_device = GPUDevice{
            .allocator = allocator,
            .capabilities = dev,
            .is_initialized = false,
            .context = null,
        };

        if (temp_device.isIoTSuitable()) iot_suitable += 1;
        if (temp_device.supportsLightweightInference()) inference_capable += 1;
        total_memory += dev.memory_total;
        if (dev.supports_int8 or dev.supports_fp16) supports_quant = true;
    }

    return .{
        .total_devices = devices.len,
        .iot_suitable_devices = iot_suitable,
        .inference_capable_devices = inference_capable,
        .total_memory_gb = @as(f32, @floatFromInt(total_memory)) / (1024.0 * 1024.0 * 1024.0),
        .supports_quantization = supports_quant,
    };
}

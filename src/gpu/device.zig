const std = @import("std");
const Allocator = std.mem.Allocator;

/// GPU device types supported by the Zig AI Interface Engine
///
/// The engine supports multiple compute backends with automatic fallback:
/// - CPU: Always available, optimized for IoT devices and security applications
/// - CUDA: NVIDIA GPU acceleration (future implementation)
/// - Vulkan: Cross-platform GPU compute (future implementation)
/// - OpenCL: Legacy GPU support (future consideration)
pub const DeviceType = enum {
    /// CPU fallback device - always available, optimized for:
    /// - IoT devices with limited GPU resources
    /// - Security-critical applications requiring deterministic behavior
    /// - Development and testing environments
    cpu,

    /// NVIDIA CUDA GPU acceleration (future implementation)
    /// - High-performance neural network inference
    /// - Optimized for data center and workstation deployments
    cuda,

    /// Cross-platform Vulkan compute (future implementation)
    /// - Universal GPU acceleration across vendors
    /// - Optimized for mobile and embedded GPU devices
    vulkan,

    /// OpenCL legacy support (future consideration)
    /// - Broad compatibility with older GPU hardware
    opencl,
};

/// Comprehensive GPU device capabilities and hardware properties
///
/// This structure provides detailed information about compute device capabilities,
/// enabling the engine to make optimal decisions for IoT deployment and security applications.
pub const DeviceCapabilities = struct {
    /// Primary device type (CPU, CUDA, Vulkan, OpenCL)
    device_type: DeviceType,

    /// Unique device identifier within the system
    device_id: u32,

    /// Human-readable device name for logging and diagnostics
    name: []const u8,

    /// Total device memory in bytes (RAM for CPU, VRAM for GPU)
    memory_total: usize,

    /// Currently available memory in bytes
    memory_free: usize,

    /// Number of compute units (CPU cores, GPU SMs, etc.)
    compute_units: u32,

    /// Maximum work group size for parallel execution
    max_work_group_size: u32,

    /// Hardware support for 16-bit floating point operations
    supports_fp16: bool,

    /// Hardware support for 8-bit integer quantization
    supports_int8: bool,

    /// Support for unified memory architecture (important for IoT devices)
    supports_unified_memory: bool,
};

/// GPU device management and initialization errors
pub const DeviceError = error{
    /// No compatible compute devices found in the system
    NoDevicesFound,

    /// Device initialization failed (driver issues, hardware problems)
    DeviceInitializationFailed,

    /// Device type not supported by current build configuration
    UnsupportedDevice,

    /// Insufficient memory available for operation
    InsufficientMemory,

    /// Device is busy or not available for use
    DeviceNotAvailable,
};

/// GPU device interface for IoT and data security applications
///
/// The GPUDevice provides a unified interface for compute device management,
/// with special optimizations for:
/// - IoT devices with limited memory and compute resources
/// - Security applications requiring deterministic and auditable behavior
/// - Cross-platform deployment with automatic fallback capabilities
///
/// Key Features:
/// - Automatic device detection and selection
/// - IoT suitability assessment
/// - Memory-efficient operation
/// - Security-focused design patterns
pub const GPUDevice = struct {
    /// Memory allocator for device management
    allocator: Allocator,

    /// Device capabilities and hardware properties
    capabilities: DeviceCapabilities,

    /// Device initialization status
    is_initialized: bool,

    /// Backend-specific context (CUDA context, Vulkan device, etc.)
    context: ?*anyopaque,

    const Self = @This();

    /// Initialize GPU device with automatic detection and selection
    ///
    /// This function performs comprehensive device detection and selects the most
    /// suitable compute device for the current environment. The selection prioritizes:
    /// 1. Vulkan compute devices (cross-platform, IoT-friendly)
    /// 2. CUDA devices (high performance, if available)
    /// 3. CPU fallback (always available, security-optimized)
    ///
    /// The selected device is automatically initialized and ready for use.
    ///
    /// Returns: Initialized GPUDevice ready for compute operations
    /// Errors: DeviceError if no suitable device can be initialized
    pub fn init(allocator: Allocator) !Self {
        var device = Self{
            .allocator = allocator,
            .capabilities = undefined,
            .is_initialized = false,
            .context = null,
        };

        // Detect available GPU devices
        const detected_device = try detectBestDevice(allocator);
        device.capabilities = detected_device;

        // Initialize the selected device
        try device.initializeDevice();

        std.log.info("GPU Device initialized: {s} (Type: {s}, Memory: {d}MB)", .{
            device.capabilities.name,
            @tagName(device.capabilities.device_type),
            device.capabilities.memory_total / (1024 * 1024),
        });

        return device;
    }

    /// Deinitialize GPU device and cleanup resources
    pub fn deinit(self: *Self) void {
        if (self.is_initialized) {
            self.cleanupDevice();
            self.is_initialized = false;
        }

        if (self.context) |ctx| {
            // Backend-specific cleanup
            _ = ctx;
            self.context = null;
        }
    }

    /// Check if device supports the required operations for lightweight LLM inference
    pub fn supportsLightweightInference(self: *const Self) bool {
        return self.capabilities.memory_total >= 512 * 1024 * 1024 and // At least 512MB
            self.capabilities.compute_units >= 4 and // Minimum compute capability
            (self.capabilities.supports_fp16 or self.capabilities.supports_int8); // Quantization support
    }

    /// Get current memory usage statistics
    pub fn getMemoryInfo(self: *const Self) struct { total: usize, free: usize, used: usize } {
        const used = self.capabilities.memory_total - self.capabilities.memory_free;
        return .{
            .total = self.capabilities.memory_total,
            .free = self.capabilities.memory_free,
            .used = used,
        };
    }

    /// Check if device is suitable for IoT deployment
    pub fn isIoTSuitable(self: *const Self) bool {
        const memory_mb = self.capabilities.memory_total / (1024 * 1024);
        return memory_mb <= 4096 and // Max 4GB for IoT constraints
            self.capabilities.device_type != .cuda or // Prefer non-CUDA for IoT
            memory_mb <= 2048; // Smaller CUDA devices acceptable
    }

    /// Private: Detect the best available GPU device
    fn detectBestDevice(allocator: Allocator) !DeviceCapabilities {
        // Try to detect devices in order of preference for lightweight inference

        // 1. Try Vulkan compute (cross-platform, good for IoT)
        if (detectVulkanDevice(allocator)) |vulkan_device| {
            return vulkan_device;
        } else |_| {}

        // 2. Try CUDA (if available, good performance)
        if (detectCudaDevice(allocator)) |cuda_device| {
            return cuda_device;
        } else |_| {}

        // 3. Fallback to CPU (always available)
        return detectCpuDevice(allocator);
    }

    /// Private: Detect Vulkan compute device
    fn detectVulkanDevice(allocator: Allocator) !DeviceCapabilities {
        _ = allocator;

        // TODO: Implement Vulkan device detection
        // For now, return error to fallback to other backends
        return DeviceError.NoDevicesFound;
    }

    /// Private: Detect CUDA device
    fn detectCudaDevice(allocator: Allocator) !DeviceCapabilities {
        _ = allocator;

        // TODO: Implement CUDA device detection
        // For now, return error to fallback to CPU
        return DeviceError.NoDevicesFound;
    }

    /// Private: Detect CPU as compute device
    fn detectCpuDevice(allocator: Allocator) DeviceCapabilities {
        _ = allocator;

        const cpu_count = std.Thread.getCpuCount() catch 4;
        const total_memory = 1024 * 1024 * 1024; // 1GB default for CPU

        return DeviceCapabilities{
            .device_type = .cpu,
            .device_id = 0,
            .name = "CPU Fallback Device",
            .memory_total = total_memory,
            .memory_free = total_memory * 3 / 4, // Assume 75% available
            .compute_units = @intCast(cpu_count),
            .max_work_group_size = 1, // Sequential execution
            .supports_fp16 = false, // CPU typically uses fp32
            .supports_int8 = true, // CPU can handle int8 quantization
            .supports_unified_memory = true, // CPU has unified memory
        };
    }

    /// Initialize the selected device
    pub fn initializeDevice(self: *Self) !void {
        switch (self.capabilities.device_type) {
            .cpu => {
                // CPU device is always ready
                self.is_initialized = true;
            },
            .cuda => {
                // TODO: Initialize CUDA context
                return DeviceError.DeviceInitializationFailed;
            },
            .vulkan => {
                // TODO: Initialize Vulkan compute context
                return DeviceError.DeviceInitializationFailed;
            },
            .opencl => {
                // TODO: Future OpenCL support
                return DeviceError.UnsupportedDevice;
            },
        }
    }

    /// Private: Cleanup device-specific resources
    fn cleanupDevice(self: *Self) void {
        switch (self.capabilities.device_type) {
            .cpu => {
                // No cleanup needed for CPU
            },
            .cuda => {
                // TODO: Cleanup CUDA context
            },
            .vulkan => {
                // TODO: Cleanup Vulkan context
            },
            .opencl => {
                // TODO: Future OpenCL cleanup
            },
        }
    }
};

/// Utility function to get all available devices
pub fn enumerateDevices(allocator: Allocator) ![]DeviceCapabilities {
    var devices = std.ArrayList(DeviceCapabilities).init(allocator);
    defer devices.deinit();

    // Always add CPU as fallback
    try devices.append(GPUDevice.detectCpuDevice(allocator));

    // Try to add GPU devices
    if (GPUDevice.detectVulkanDevice(allocator)) |vulkan_device| {
        try devices.append(vulkan_device);
    } else |_| {}

    if (GPUDevice.detectCudaDevice(allocator)) |cuda_device| {
        try devices.append(cuda_device);
    } else |_| {}

    return devices.toOwnedSlice();
}

/// Utility function to select best device for lightweight LLM inference
pub fn selectBestDeviceForInference(allocator: Allocator) !DeviceCapabilities {
    const devices = try enumerateDevices(allocator);
    defer allocator.free(devices);

    // Score devices based on suitability for lightweight inference
    var best_device: ?DeviceCapabilities = null;
    var best_score: f32 = 0.0;

    for (devices) |device| {
        var score: f32 = 0.0;

        // Memory score (more is better, but diminishing returns)
        const memory_gb = @as(f32, @floatFromInt(device.memory_total)) / (1024.0 * 1024.0 * 1024.0);
        score += @min(memory_gb * 10.0, 40.0); // Cap at 4GB worth of points

        // Compute units score
        score += @as(f32, @floatFromInt(device.compute_units)) * 2.0;

        // Device type preference (Vulkan > CUDA > CPU for IoT)
        switch (device.device_type) {
            .vulkan => score += 20.0,
            .cuda => score += 15.0,
            .cpu => score += 5.0,
            .opencl => score += 10.0,
        }

        // Quantization support bonus
        if (device.supports_fp16) score += 10.0;
        if (device.supports_int8) score += 15.0;

        // IoT suitability bonus
        if (memory_gb <= 4.0) score += 5.0;

        if (score > best_score) {
            best_score = score;
            best_device = device;
        }
    }

    return best_device orelse return DeviceError.NoDevicesFound;
}

const std = @import("std");
const Allocator = std.mem.Allocator;
const TensorInterface = @import("tensor.zig").TensorInterface;

/// Device abstraction interface for compute backends
/// Supports CPU, GPU, NPU, and other accelerators
pub const DeviceInterface = struct {
    /// Device types
    pub const DeviceType = enum {
        cpu,
        gpu,
        npu,
        tpu,
        fpga,
        custom,
    };

    /// Device capabilities
    pub const Capabilities = struct {
        supports_fp32: bool,
        supports_fp16: bool,
        supports_int8: bool,
        supports_simd: bool,
        supports_parallel: bool,
        max_memory_mb: usize,
        compute_units: usize,
        memory_bandwidth_gbps: f32,
        peak_flops: f64,
    };

    /// Device memory information
    pub const MemoryInfo = struct {
        total_bytes: usize,
        available_bytes: usize,
        used_bytes: usize,
        fragmentation_ratio: f32,
    };

    /// Device statistics
    pub const DeviceStats = struct {
        utilization_percent: f32,
        temperature_celsius: f32,
        power_watts: f32,
        memory_info: MemoryInfo,
        operations_per_second: f64,
    };

    /// Device errors
    pub const DeviceError = error{
        DeviceNotFound,
        DeviceNotSupported,
        DeviceBusy,
        OutOfDeviceMemory,
        KernelLaunchFailed,
        SynchronizationFailed,
        InvalidOperation,
        DriverError,
    };

    /// Device operations
    pub const Operations = struct {
        /// Get device type
        getTypeFn: *const fn (ctx: *anyopaque) DeviceType,
        
        /// Get device capabilities
        getCapabilitiesFn: *const fn (ctx: *anyopaque) Capabilities,
        
        /// Get device statistics
        getStatsFn: *const fn (ctx: *anyopaque) DeviceStats,
        
        /// Initialize device
        initFn: *const fn (ctx: *anyopaque) DeviceError!void,
        
        /// Deinitialize device
        deinitFn: *const fn (ctx: *anyopaque) void,
        
        /// Allocate device memory
        allocMemoryFn: *const fn (ctx: *anyopaque, size: usize, alignment: usize) DeviceError!*anyopaque,
        
        /// Free device memory
        freeMemoryFn: *const fn (ctx: *anyopaque, ptr: *anyopaque) void,
        
        /// Copy memory to device
        copyToDeviceFn: *const fn (ctx: *anyopaque, dst: *anyopaque, src: []const u8) DeviceError!void,
        
        /// Copy memory from device
        copyFromDeviceFn: *const fn (ctx: *anyopaque, dst: []u8, src: *const anyopaque) DeviceError!void,
        
        /// Synchronize device operations
        synchronizeFn: *const fn (ctx: *anyopaque) DeviceError!void,
        
        /// Execute kernel/operation
        executeKernelFn: *const fn (ctx: *anyopaque, kernel: *const anyopaque, inputs: []const TensorInterface, outputs: []TensorInterface) DeviceError!void,
    };

    impl: Operations,
    ctx: *anyopaque,

    pub fn init(ctx: *anyopaque, impl: Operations) DeviceInterface {
        return DeviceInterface{
            .impl = impl,
            .ctx = ctx,
        };
    }

    pub fn getType(self: *const DeviceInterface) DeviceType {
        return self.impl.getTypeFn(self.ctx);
    }

    pub fn getCapabilities(self: *const DeviceInterface) Capabilities {
        return self.impl.getCapabilitiesFn(self.ctx);
    }

    pub fn getStats(self: *const DeviceInterface) DeviceStats {
        return self.impl.getStatsFn(self.ctx);
    }

    pub fn initialize(self: *DeviceInterface) DeviceError!void {
        return self.impl.initFn(self.ctx);
    }

    pub fn deinitialize(self: *DeviceInterface) void {
        self.impl.deinitFn(self.ctx);
    }

    pub fn allocMemory(self: *DeviceInterface, size: usize, alignment: usize) DeviceError!*anyopaque {
        return self.impl.allocMemoryFn(self.ctx, size, alignment);
    }

    pub fn freeMemory(self: *DeviceInterface, ptr: *anyopaque) void {
        self.impl.freeMemoryFn(self.ctx, ptr);
    }

    pub fn copyToDevice(self: *DeviceInterface, dst: *anyopaque, src: []const u8) DeviceError!void {
        return self.impl.copyToDeviceFn(self.ctx, dst, src);
    }

    pub fn copyFromDevice(self: *DeviceInterface, dst: []u8, src: *const anyopaque) DeviceError!void {
        return self.impl.copyFromDeviceFn(self.ctx, dst, src);
    }

    pub fn synchronize(self: *DeviceInterface) DeviceError!void {
        return self.impl.synchronizeFn(self.ctx);
    }

    pub fn executeKernel(self: *DeviceInterface, kernel: *const anyopaque, inputs: []const TensorInterface, outputs: []TensorInterface) DeviceError!void {
        return self.impl.executeKernelFn(self.ctx, kernel, inputs, outputs);
    }
};

/// Kernel interface for device operations
pub const KernelInterface = struct {
    /// Kernel types
    pub const KernelType = enum {
        elementwise,
        reduction,
        convolution,
        matrix_multiply,
        activation,
        normalization,
        custom,
    };

    /// Kernel configuration
    pub const Config = struct {
        kernel_type: KernelType,
        grid_size: [3]usize,
        block_size: [3]usize,
        shared_memory_bytes: usize,
        registers_per_thread: usize,
    };

    /// Kernel operations
    pub const Operations = struct {
        /// Compile kernel
        compileFn: *const fn (ctx: *anyopaque, source: []const u8, config: Config) DeviceInterface.DeviceError!*anyopaque,
        
        /// Launch kernel
        launchFn: *const fn (ctx: *anyopaque, kernel: *const anyopaque, inputs: []const TensorInterface, outputs: []TensorInterface) DeviceInterface.DeviceError!void,
        
        /// Get kernel info
        getInfoFn: *const fn (ctx: *anyopaque, kernel: *const anyopaque) Config,
        
        /// Destroy kernel
        destroyFn: *const fn (ctx: *anyopaque, kernel: *anyopaque) void,
    };

    impl: Operations,
    ctx: *anyopaque,

    pub fn init(ctx: *anyopaque, impl: Operations) KernelInterface {
        return KernelInterface{
            .impl = impl,
            .ctx = ctx,
        };
    }

    pub fn compile(self: *KernelInterface, source: []const u8, config: Config) DeviceInterface.DeviceError!*anyopaque {
        return self.impl.compileFn(self.ctx, source, config);
    }

    pub fn launch(self: *KernelInterface, kernel: *const anyopaque, inputs: []const TensorInterface, outputs: []TensorInterface) DeviceInterface.DeviceError!void {
        return self.impl.launchFn(self.ctx, kernel, inputs, outputs);
    }

    pub fn getInfo(self: *KernelInterface, kernel: *const anyopaque) Config {
        return self.impl.getInfoFn(self.ctx, kernel);
    }

    pub fn destroy(self: *KernelInterface, kernel: *anyopaque) void {
        self.impl.destroyFn(self.ctx, kernel);
    }
};

/// Device manager for handling multiple devices
pub const DeviceManager = struct {
    devices: std.ArrayList(DeviceInterface),
    allocator: Allocator,
    default_device: ?usize,

    pub fn init(allocator: Allocator) DeviceManager {
        return DeviceManager{
            .devices = std.ArrayList(DeviceInterface).init(allocator),
            .allocator = allocator,
            .default_device = null,
        };
    }

    pub fn deinit(self: *DeviceManager) void {
        for (self.devices.items) |*device| {
            device.deinitialize();
        }
        self.devices.deinit();
    }

    pub fn addDevice(self: *DeviceManager, device: DeviceInterface) !usize {
        try self.devices.append(device);
        if (self.default_device == null) {
            self.default_device = self.devices.items.len - 1;
        }
        return self.devices.items.len - 1;
    }

    pub fn getDevice(self: *DeviceManager, index: usize) ?*DeviceInterface {
        if (index >= self.devices.items.len) return null;
        return &self.devices.items[index];
    }

    pub fn getDefaultDevice(self: *DeviceManager) ?*DeviceInterface {
        if (self.default_device) |idx| {
            return self.getDevice(idx);
        }
        return null;
    }

    pub fn setDefaultDevice(self: *DeviceManager, index: usize) bool {
        if (index >= self.devices.items.len) return false;
        self.default_device = index;
        return true;
    }

    pub fn findDeviceByType(self: *DeviceManager, device_type: DeviceInterface.DeviceType) ?*DeviceInterface {
        for (self.devices.items) |*device| {
            if (device.getType() == device_type) {
                return device;
            }
        }
        return null;
    }

    pub fn getDeviceCount(self: *const DeviceManager) usize {
        return self.devices.items.len;
    }
};

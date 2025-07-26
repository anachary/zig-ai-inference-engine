const std = @import("std");
const Allocator = std.mem.Allocator;
const c = @cImport({
    @cInclude("cuda_runtime.h");
    @cInclude("cublas_v2.h");
    @cInclude("cudnn.h");
});

// Import common interfaces
const common_interfaces = @import("common-interfaces");
const TensorInterface = common_interfaces.TensorInterface;
const backend = @import("backend.zig");
const BackendError = backend.BackendError;
const MemoryInfo = backend.MemoryInfo;
const Kernel = backend.Kernel;

/// CUDA-specific errors
pub const CudaError = error{
    CudaNotAvailable,
    CudaInitializationFailed,
    CudaMemoryError,
    CudaKernelError,
    CudaCompilationError,
    CudaSynchronizationError,
} || BackendError;

/// CUDA device properties
pub const CudaDeviceInfo = struct {
    device_id: i32,
    name: [256]u8,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    total_global_memory: usize,
    multiprocessor_count: i32,
    max_threads_per_block: i32,
    max_threads_per_multiprocessor: i32,
    warp_size: i32,
    memory_clock_rate: i32,
    memory_bus_width: i32,
};

/// CUDA memory buffer
pub const CudaBuffer = struct {
    ptr: ?*anyopaque,
    size: usize,
    device_id: i32,

    pub fn init(size: usize, device_id: i32) !CudaBuffer {
        var ptr: ?*anyopaque = null;
        const result = c.cudaMalloc(&ptr, size);
        if (result != c.cudaSuccess) {
            return CudaError.CudaMemoryError;
        }

        return CudaBuffer{
            .ptr = ptr,
            .size = size,
            .device_id = device_id,
        };
    }

    pub fn deinit(self: *CudaBuffer) void {
        if (self.ptr) |ptr| {
            _ = c.cudaFree(ptr);
            self.ptr = null;
        }
    }

    pub fn copyFromHost(self: *CudaBuffer, host_data: []const u8) !void {
        if (host_data.len > self.size) {
            return CudaError.CudaMemoryError;
        }

        const result = c.cudaMemcpy(self.ptr, host_data.ptr, host_data.len, c.cudaMemcpyHostToDevice);
        if (result != c.cudaSuccess) {
            return CudaError.CudaMemoryError;
        }
    }

    pub fn copyToHost(self: *CudaBuffer, host_data: []u8) !void {
        if (host_data.len > self.size) {
            return CudaError.CudaMemoryError;
        }

        const result = c.cudaMemcpy(host_data.ptr, self.ptr, host_data.len, c.cudaMemcpyDeviceToHost);
        if (result != c.cudaSuccess) {
            return CudaError.CudaMemoryError;
        }
    }
};

/// CUDA kernel wrapper
pub const CudaKernel = struct {
    module: c.CUmodule,
    function: c.CUfunction,
    name: []const u8,
    compiled: bool,

    pub fn init(name: []const u8) CudaKernel {
        return CudaKernel{
            .module = undefined,
            .function = undefined,
            .name = name,
            .compiled = false,
        };
    }

    pub fn deinit(self: *CudaKernel) void {
        if (self.compiled) {
            _ = c.cuModuleUnload(self.module);
            self.compiled = false;
        }
    }
};

/// CUDA backend implementation
pub const CudaBackend = struct {
    allocator: Allocator,
    device_count: i32,
    current_device: i32,
    device_info: ?CudaDeviceInfo,
    cublas_handle: ?c.cublasHandle_t,
    cudnn_handle: ?c.cudnnHandle_t,
    kernels: std.StringHashMap(CudaKernel),
    buffers: std.ArrayList(CudaBuffer),
    initialized: bool,

    const Self = @This();

    pub fn init(allocator: Allocator) !Self {
        var self = Self{
            .allocator = allocator,
            .device_count = 0,
            .current_device = 0,
            .device_info = null,
            .cublas_handle = null,
            .cudnn_handle = null,
            .kernels = std.StringHashMap(CudaKernel).init(allocator),
            .buffers = std.ArrayList(CudaBuffer).init(allocator),
            .initialized = false,
        };

        try self.initialize();
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.initialized) {
            self.shutdown();
        }

        // Clean up kernels
        var kernel_iter = self.kernels.iterator();
        while (kernel_iter.next()) |entry| {
            var kernel = entry.value_ptr;
            kernel.deinit();
        }
        self.kernels.deinit();

        // Clean up buffers
        for (self.buffers.items) |*buffer| {
            buffer.deinit();
        }
        self.buffers.deinit();
    }

    /// Check if CUDA is available
    pub fn isAvailable() bool {
        var device_count: i32 = 0;
        const result = c.cudaGetDeviceCount(&device_count);
        return result == c.cudaSuccess and device_count > 0;
    }

    /// Initialize CUDA backend
    fn initialize(self: *Self) !void {
        // Initialize CUDA runtime
        var result = c.cudaGetDeviceCount(&self.device_count);
        if (result != c.cudaSuccess or self.device_count == 0) {
            return CudaError.CudaNotAvailable;
        }

        // Set device
        result = c.cudaSetDevice(0);
        if (result != c.cudaSuccess) {
            return CudaError.CudaInitializationFailed;
        }
        self.current_device = 0;

        // Get device properties
        var props: c.cudaDeviceProp = undefined;
        result = c.cudaGetDeviceProperties(&props, 0);
        if (result != c.cudaSuccess) {
            return CudaError.CudaInitializationFailed;
        }

        self.device_info = CudaDeviceInfo{
            .device_id = 0,
            .name = props.name,
            .compute_capability_major = props.major,
            .compute_capability_minor = props.minor,
            .total_global_memory = props.totalGlobalMem,
            .multiprocessor_count = props.multiProcessorCount,
            .max_threads_per_block = props.maxThreadsPerBlock,
            .max_threads_per_multiprocessor = props.maxThreadsPerMultiProcessor,
            .warp_size = props.warpSize,
            .memory_clock_rate = props.memoryClockRate,
            .memory_bus_width = props.memoryBusWidth,
        };

        // Initialize cuBLAS
        var cublas_handle: c.cublasHandle_t = undefined;
        var cublas_result = c.cublasCreate(&cublas_handle);
        if (cublas_result == c.CUBLAS_STATUS_SUCCESS) {
            self.cublas_handle = cublas_handle;
        }

        // Initialize cuDNN
        var cudnn_handle: c.cudnnHandle_t = undefined;
        var cudnn_result = c.cudnnCreate(&cudnn_handle);
        if (cudnn_result == c.CUDNN_STATUS_SUCCESS) {
            self.cudnn_handle = cudnn_handle;
        }

        self.initialized = true;
        std.log.info("CUDA backend initialized successfully", .{});
        if (self.device_info) |info| {
            std.log.info("Device: {s}", .{info.name});
            std.log.info("Compute Capability: {}.{}", .{ info.compute_capability_major, info.compute_capability_minor });
            std.log.info("Global Memory: {d:.2} GB", .{@as(f64, @floatFromInt(info.total_global_memory)) / (1024.0 * 1024.0 * 1024.0)});
        }
    }

    /// Shutdown CUDA backend
    fn shutdown(self: *Self) void {
        if (self.cudnn_handle) |handle| {
            _ = c.cudnnDestroy(handle);
            self.cudnn_handle = null;
        }

        if (self.cublas_handle) |handle| {
            _ = c.cublasDestroy(handle);
            self.cublas_handle = null;
        }

        _ = c.cudaDeviceReset();
        self.initialized = false;
        std.log.info("CUDA backend shutdown", .{});
    }

    /// Allocate GPU buffer
    pub fn allocateBuffer(self: *Self, size: usize) !CudaBuffer {
        var buffer = try CudaBuffer.init(size, self.current_device);
        try self.buffers.append(buffer);
        return buffer;
    }

    /// Get memory information
    pub fn getMemoryInfo(self: *Self) !MemoryInfo {
        _ = self;
        var free_bytes: usize = 0;
        var total_bytes: usize = 0;

        const result = c.cudaMemGetInfo(&free_bytes, &total_bytes);
        if (result != c.cudaSuccess) {
            return CudaError.CudaMemoryError;
        }

        return MemoryInfo{
            .total_bytes = total_bytes,
            .available_bytes = free_bytes,
            .used_bytes = total_bytes - free_bytes,
            .fragmentation_ratio = 0.0, // TODO: Calculate fragmentation
        };
    }

    /// Synchronize device
    pub fn synchronize(self: *Self) !void {
        _ = self;
        const result = c.cudaDeviceSynchronize();
        if (result != c.cudaSuccess) {
            return CudaError.CudaSynchronizationError;
        }
    }

    /// Compile CUDA kernel from PTX or CUBIN
    pub fn compileKernel(self: *Self, name: []const u8, ptx_source: []const u8) !void {
        var cuda_kernel = CudaKernel.init(name);

        // Load module from PTX
        const result = c.cuModuleLoadDataEx(&cuda_kernel.module, ptx_source.ptr, 0, null, null);
        if (result != c.CUDA_SUCCESS) {
            return CudaError.CudaCompilationError;
        }

        // Get function
        const func_result = c.cuModuleGetFunction(&cuda_kernel.function, cuda_kernel.module, name.ptr);
        if (func_result != c.CUDA_SUCCESS) {
            _ = c.cuModuleUnload(cuda_kernel.module);
            return CudaError.CudaCompilationError;
        }

        cuda_kernel.compiled = true;
        try self.kernels.put(name, cuda_kernel);

        std.log.info("CUDA kernel compiled: {s}", .{name});
    }

    /// Execute CUDA kernel
    pub fn executeKernel(
        self: *Self,
        name: []const u8,
        grid_size: [3]u32,
        block_size: [3]u32,
        args: []const ?*anyopaque,
    ) !void {
        const kernel = self.kernels.get(name) orelse return CudaError.CudaKernelError;

        if (!kernel.compiled) {
            return CudaError.CudaKernelError;
        }

        const result = c.cuLaunchKernel(
            kernel.function,
            grid_size[0],
            grid_size[1],
            grid_size[2],
            block_size[0],
            block_size[1],
            block_size[2],
            0, // shared memory
            null, // stream
            @ptrCast(args.ptr),
            null,
        );

        if (result != c.CUDA_SUCCESS) {
            return CudaError.CudaKernelError;
        }
    }
};

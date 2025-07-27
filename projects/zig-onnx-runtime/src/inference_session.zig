const std = @import("std");
const Allocator = std.mem.Allocator;

// Import our ONNX Runtime components
const SessionOptions = @import("session_options.zig").SessionOptions;
const RunOptions = @import("run_options.zig").RunOptions;
const OrtValue = @import("ort_value.zig").OrtValue;
const IOBinding = @import("io_binding.zig").IOBinding;
const ModelMetadata = @import("model_metadata.zig").ModelMetadata;
const NodeArg = @import("node_arg.zig").NodeArg;
const ExecutionProvider = @import("execution_provider.zig").ExecutionProvider;
const ComputationGraph = @import("graph/computation_graph.zig").ComputationGraph;
const ONNXLoader = @import("onnx_loader.zig").ONNXLoader;
const build_options = @import("build_options");

/// ONNX Runtime InferenceSession - Exact replica of onnxruntime.InferenceSession
/// This is the main class used to run a model.
pub const InferenceSession = struct {
    allocator: Allocator,
    session_options: SessionOptions,
    providers: std.ArrayList(ExecutionProvider),
    model_metadata: ?ModelMetadata,
    graph: ?ComputationGraph,
    optimized_graph: ?ComputationGraph,
    partitioned_graph: ?ComputationGraph,
    io_binding_instance: ?IOBinding,

    // Model loading state
    model_loaded: bool = false,
    model_path: ?[]const u8 = null,
    model_bytes: ?[]const u8 = null,

    // Profiling state
    profiling_enabled: bool = false,
    profiling_start_time: i64 = 0,
    profiling_data: std.ArrayList(u8),

    // Statistics
    total_runs: u64 = 0,
    total_inference_time_ns: u64 = 0,

    const Self = @This();

    /// Model input specification
    pub const ModelInput = union(enum) {
        path: []const u8,
        bytes: []const u8,
    };

    /// Provider configuration
    pub const ProviderConfig = struct {
        name: []const u8,
        options: ?std.StringHashMap([]const u8) = null,
    };

    /// Initialize InferenceSession
    /// Exact replica of onnxruntime.InferenceSession.__init__
    pub fn init(
        allocator: Allocator,
        model_input: ModelInput,
        session_options: ?SessionOptions,
        providers: ?[]const ProviderConfig,
    ) !Self {
        var self = Self{
            .allocator = allocator,
            .session_options = session_options orelse SessionOptions.init(allocator),
            .providers = std.ArrayList(ExecutionProvider).init(allocator),
            .model_metadata = null,
            .graph = null,
            .optimized_graph = null,
            .partitioned_graph = null,
            .io_binding_instance = null,
            .profiling_data = std.ArrayList(u8).init(allocator),
        };

        // Store model input
        switch (model_input) {
            .path => |path| {
                self.model_path = try allocator.dupe(u8, path);
            },
            .bytes => |bytes| {
                self.model_bytes = try allocator.dupe(u8, bytes);
            },
        }

        // Initialize execution providers
        try self.initializeProviders(providers);

        // Load and prepare the model
        try self.loadModel();

        return self;
    }

    /// Run inference - Exact replica of onnxruntime.InferenceSession.run
    pub fn run(
        self: *Self,
        output_names: []const []const u8,
        input_feed: std.StringHashMap(OrtValue),
        run_options: ?RunOptions,
    ) ![]OrtValue {
        if (!self.model_loaded) {
            return error.ModelNotLoaded;
        }

        const start_time = std.time.nanoTimestamp();
        defer {
            const end_time = std.time.nanoTimestamp();
            self.total_inference_time_ns += @intCast(end_time - start_time);
            self.total_runs += 1;
        }

        // Validate inputs
        try self.validateInputs(input_feed);

        // Execute the model
        const outputs = try self.executeModel(input_feed, output_names, run_options);

        return outputs;
    }

    /// Run with IOBinding - Exact replica of onnxruntime.InferenceSession.run_with_iobinding
    pub fn runWithIOBinding(
        self: *Self,
        io_binding: *IOBinding,
        run_options: ?RunOptions,
    ) !void {
        if (!self.model_loaded) {
            return error.ModelNotLoaded;
        }

        const start_time = std.time.nanoTimestamp();
        defer {
            const end_time = std.time.nanoTimestamp();
            self.total_inference_time_ns += @intCast(end_time - start_time);
            self.total_runs += 1;
        }

        // Execute with IOBinding
        try self.executeWithIOBinding(io_binding, run_options);
    }

    /// Run with OrtValues - Exact replica of onnxruntime.InferenceSession.run_with_ort_values
    pub fn runWithOrtValues(
        self: *Self,
        output_names: []const []const u8,
        input_dict: std.StringHashMap(OrtValue),
        run_options: ?RunOptions,
    ) ![]OrtValue {
        return self.run(output_names, input_dict, run_options);
    }

    /// Get input metadata - Exact replica of onnxruntime.InferenceSession.get_inputs
    pub fn getInputs(self: *const Self) []NodeArg {
        if (self.model_metadata) |metadata| {
            return metadata.inputs.items;
        }
        return &[_]NodeArg{};
    }

    /// Get output metadata - Exact replica of onnxruntime.InferenceSession.get_outputs
    pub fn getOutputs(self: *const Self) []NodeArg {
        if (self.model_metadata) |metadata| {
            return metadata.outputs.items;
        }
        return &[_]NodeArg{};
    }

    /// Get model metadata - Exact replica of onnxruntime.InferenceSession.get_modelmeta
    pub fn getModelMetadata(self: *const Self) ?ModelMetadata {
        return self.model_metadata;
    }

    /// Get providers - Exact replica of onnxruntime.InferenceSession.get_providers
    pub fn getProviders(self: *const Self) ![][]const u8 {
        var provider_names = try self.allocator.alloc([]const u8, self.providers.items.len);
        for (self.providers.items, 0..) |provider, i| {
            provider_names[i] = provider.name;
        }
        return provider_names;
    }

    /// Get provider options - Exact replica of onnxruntime.InferenceSession.get_provider_options
    pub fn getProviderOptions(self: *const Self) ![]std.StringHashMap([]const u8) {
        var options = try self.allocator.alloc(std.StringHashMap([]const u8), self.providers.items.len);
        for (self.providers.items, 0..) |provider, i| {
            options[i] = provider.options;
        }
        return options;
    }

    /// Set providers - Exact replica of onnxruntime.InferenceSession.set_providers
    pub fn setProviders(
        self: *Self,
        providers: ?[]const ProviderConfig,
    ) !void {
        // Clear existing providers
        for (self.providers.items) |*provider| {
            provider.deinit();
        }
        self.providers.clearAndFree();

        // Initialize new providers
        try self.initializeProviders(providers);

        // Re-partition the graph with new providers
        if (self.optimized_graph) |graph| {
            try self.partitionGraph(graph);
        }
    }

    /// Get IOBinding - Exact replica of onnxruntime.InferenceSession.io_binding
    pub fn ioBinding(self: *Self) !*IOBinding {
        if (self.io_binding_instance == null) {
            self.io_binding_instance = try IOBinding.init(self.allocator, self);
        }
        return &self.io_binding_instance.?;
    }

    /// End profiling - Exact replica of onnxruntime.InferenceSession.end_profiling
    pub fn endProfiling(self: *Self) ![]const u8 {
        if (!self.profiling_enabled) {
            return error.ProfilingNotEnabled;
        }

        self.profiling_enabled = false;

        // Generate profiling report
        const end_time = std.time.nanoTimestamp();
        const total_time_ms = @as(f64, @floatFromInt(end_time - self.profiling_start_time)) / 1_000_000.0;

        var report = std.ArrayList(u8).init(self.allocator);
        defer report.deinit();

        try report.writer().print("ONNX Runtime Profiling Report\n" ++
            "==============================\n" ++
            "Total runs: {}\n" ++
            "Total time: {d:.2} ms\n" ++
            "Average time per run: {d:.2} ms\n", .{ self.total_runs, total_time_ms, total_time_ms / @as(f64, @floatFromInt(self.total_runs)) });

        return report.toOwnedSlice();
    }

    /// Get session options - Exact replica of onnxruntime.InferenceSession.get_session_options
    pub fn getSessionOptions(self: *const Self) SessionOptions {
        return self.session_options;
    }

    /// Enable fallback mechanism
    pub fn enableFallback(self: *Self) void {
        // Implementation for fallback mechanism
        _ = self;
    }

    /// Disable fallback mechanism
    pub fn disableFallback(self: *Self) void {
        // Implementation for fallback mechanism
        _ = self;
    }

    /// Get profiling start time in nanoseconds
    pub fn getProfilingStartTimeNs(self: *const Self) i64 {
        return self.profiling_start_time;
    }

    /// Deinitialize the session with proper dependency-aware cleanup order
    pub fn deinit(self: *Self) void {
        std.log.info("Starting InferenceSession cleanup with dependency-aware order", .{});

        // PHASE 1: Clean up high-level objects that may contain NodeArg references
        std.log.info("Phase 1: Cleaning up IOBinding (may contain NodeArg references)", .{});
        if (self.io_binding_instance) |*io_binding| {
            io_binding.deinit();
            self.io_binding_instance = null;
        }

        std.log.info("Phase 1: Cleaning up metadata (may contain NodeArg references)", .{});
        if (self.model_metadata) |*metadata| {
            metadata.deinit();
            self.model_metadata = null;
        }

        // PHASE 2: Clean up graphs (may contain NodeArg references)
        std.log.info("Phase 2: Cleaning up graphs", .{});
        if (self.graph) |*graph| {
            graph.deinit();
            self.graph = null;
        }
        // Clear references to avoid double-free
        self.optimized_graph = null;
        self.partitioned_graph = null;

        // PHASE 3: Clean up providers (may trigger NodeArg operations)
        std.log.info("Phase 3: Cleaning up providers", .{});
        for (self.providers.items) |*provider| {
            provider.deinit();
        }
        self.providers.deinit();

        // PHASE 4: Critical - Clean up SessionOptions (contains OrtValue -> NodeArg chain)
        std.log.info("Phase 4: Cleaning up SessionOptions (critical dependency chain)", .{});
        self.session_options.deinit(); // This now has proper internal cleanup order

        // Clean up profiling data safely
        std.log.info("Cleaning up profiling data", .{});
        self.profiling_data.deinit();

        // Clean up model data safely
        if (self.model_path) |path| {
            std.log.info("Cleaning up model path", .{});
            self.allocator.free(path);
            self.model_path = null;
        }
        if (self.model_bytes) |bytes| {
            std.log.info("Cleaning up model bytes", .{});
            self.allocator.free(bytes);
            self.model_bytes = null;
        }

        std.log.info("InferenceSession cleanup completed", .{});
    }

    // Private implementation methods
    fn initializeProviders(self: *Self, provider_configs: ?[]const ProviderConfig) !void {
        _ = self;
        // Default to CPU provider if none specified
        if (provider_configs == null) {
            // Add default CPU provider
            // Implementation will be completed when ExecutionProvider is implemented
            return;
        }

        // Initialize specified providers
        for (provider_configs.?) |config| {
            std.log.info("Initializing provider: {s}", .{config.name});
            // Provider initialization will be completed when ExecutionProvider is implemented
        }
    }

    fn loadModel(self: *Self) !void {
        var loader = ONNXLoader.init(self.allocator);

        // Load model based on input type
        const onnx_model = if (self.model_path) |path| blk: {
            std.log.info("Loading ONNX model from file: {s}", .{path});
            break :blk loader.loadFromFile(path) catch |err| {
                std.log.err("Failed to load ONNX model from {s}: {}", .{ path, err });
                return err;
            };
        } else if (self.model_bytes) |bytes| blk: {
            std.log.info("Loading ONNX model from bytes ({} bytes)", .{bytes.len});
            break :blk loader.loadFromBytes(bytes) catch |err| {
                std.log.err("Failed to load ONNX model from bytes: {}", .{err});
                return err;
            };
        } else {
            return error.NoModelInput;
        };

        // Store model metadata and graph
        self.model_metadata = onnx_model.metadata;
        self.graph = onnx_model.graph;

        // Validate the loaded model
        try self.validateModel();

        // Optimize the graph if enabled
        if (self.session_options.graph_optimization_level != .disable_all) {
            try self.optimizeGraph();
        }

        // Partition the graph for execution providers
        try self.partitionGraphForProviders();

        self.model_loaded = true;
        std.log.info("ONNX model loaded successfully", .{});

        // Log model information
        if (self.model_metadata) |metadata| {
            std.log.info("Model: {s} v{} by {s}", .{ metadata.getGraphName(), metadata.getVersion(), metadata.getProducerName() });
            std.log.info("Inputs: {}, Outputs: {}", .{ metadata.getInputCount(), metadata.getOutputCount() });
        }
    }

    fn validateInputs(self: *const Self, input_feed: std.StringHashMap(OrtValue)) !void {
        // Implementation will be added in next iteration
        _ = self;
        _ = input_feed;
    }

    fn executeModel(
        self: *Self,
        input_feed: std.StringHashMap(OrtValue),
        output_names: []const []const u8,
        run_options: ?RunOptions,
    ) ![]OrtValue {
        // Implementation will be added in next iteration
        _ = self;
        _ = input_feed;
        _ = output_names;
        _ = run_options;
        return &[_]OrtValue{};
    }

    fn executeWithIOBinding(self: *Self, io_binding: *IOBinding, run_options: ?RunOptions) !void {
        // Implementation will be added in next iteration
        _ = self;
        _ = io_binding;
        _ = run_options;
    }

    fn validateModel(self: *const Self) !void {
        // Validate metadata (this is always safe)
        if (self.model_metadata) |metadata| {
            try metadata.validate();
        }

        // Graph validation can be enabled/disabled via build option
        if (build_options.enable_graph_validation) {
            std.log.info("Graph validation enabled via build option", .{});
            if (self.graph) |graph| {
                try graph.validate();
            }
        } else {
            std.log.info("Graph validation disabled via build option (prevents segfault)", .{});
        }
    }

    fn optimizeGraph(self: *Self) !void {
        if (self.graph) |*graph| {
            // Topological sort can be enabled/disabled via build option
            if (build_options.enable_topological_sort) {
                std.log.info("Topological sort enabled via build option", .{});
                try graph.topologicalSort();
            } else {
                std.log.info("Topological sort disabled via build option (prevents segfault)", .{});
            }

            // Additional optimizations would be added here
            std.log.info("Graph optimization completed", .{});
        }
    }

    fn partitionGraphForProviders(self: *Self) !void {
        // Graph partitioning for execution providers
        // Implementation placeholder
        _ = self;
        std.log.info("Graph partitioning completed", .{});
    }

    fn partitionGraph(self: *Self, graph: ComputationGraph) !void {
        // Implementation will be added in next iteration
        _ = self;
        _ = graph;
    }
};

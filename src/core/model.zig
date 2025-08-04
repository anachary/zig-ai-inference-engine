const std = @import("std");
const Tensor = @import("tensor.zig").DynamicTensor;

/// Model architecture types
pub const Architecture = enum {
    llama,
    gpt,
    bert,
    t5,
    unknown,
};

/// Model metadata
pub const Metadata = struct {
    name: []const u8,
    architecture: Architecture,
    vocab_size: u32,
    context_length: u32,
    embedding_dim: u32,
    num_layers: u32,
    num_heads: u32,
    num_kv_heads: ?u32, // For grouped-query attention
    intermediate_size: ?u32,
    rope_freq_base: ?f32,
    rope_scaling: ?f32,

    pub fn deinit(self: *Metadata, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
    }
};

/// Universal model interface
pub const Model = struct {
    metadata: Metadata,
    tensors: std.StringHashMap(Tensor),
    allocator: std.mem.Allocator,

    // Virtual function table for format-specific operations
    vtable: *const VTable,
    impl: *anyopaque,

    pub const VTable = struct {
        deinit: *const fn (impl: *anyopaque, allocator: std.mem.Allocator) void,
        getTensor: *const fn (impl: *anyopaque, name: []const u8) ?*Tensor,
        getMetadata: *const fn (impl: *anyopaque) *const Metadata,
    };

    pub fn init(allocator: std.mem.Allocator, metadata: Metadata, vtable: *const VTable, impl: *anyopaque) Model {
        return Model{
            .metadata = metadata,
            .tensors = std.StringHashMap(Tensor).init(allocator),
            .allocator = allocator,
            .vtable = vtable,
            .impl = impl,
        };
    }

    pub fn deinit(self: *Model) void {
        // Clean up tensors
        var iterator = self.tensors.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.tensors.deinit();

        // Clean up metadata
        var metadata_copy = self.metadata;
        metadata_copy.deinit(self.allocator);

        // Call format-specific cleanup
        self.vtable.deinit(self.impl, self.allocator);
    }

    pub fn getTensor(self: *Model, name: []const u8) ?*Tensor {
        if (self.tensors.getPtr(name)) |tensor| {
            return tensor;
        }

        // Try format-specific tensor loading
        return self.vtable.getTensor(self.impl, name);
    }

    pub fn getMetadata(self: *Model) *const Metadata {
        return self.vtable.getMetadata(self.impl);
    }

    pub fn addTensor(self: *Model, name: []const u8, tensor: Tensor) !void {
        try self.tensors.put(name, tensor);
    }

    pub fn hasTensor(self: *Model, name: []const u8) bool {
        return self.tensors.contains(name) or (self.vtable.getTensor(self.impl, name) != null);
    }

    pub fn listTensors(self: *Model, allocator: std.mem.Allocator) ![][]const u8 {
        var names = std.ArrayList([]const u8).init(allocator);
        defer names.deinit();

        var iterator = self.tensors.keyIterator();
        while (iterator.next()) |key| {
            try names.append(key.*);
        }

        return names.toOwnedSlice();
    }

    /// Get tensor with specific type checking
    pub fn getTypedTensor(self: *Model, comptime T: type, name: []const u8) ?[]T {
        if (self.getTensor(name)) |tensor| {
            return tensor.getTypedData(T);
        }
        return null;
    }

    /// Validate model integrity
    pub fn validate(self: *Model) !void {
        const required_tensors = switch (self.metadata.architecture) {
            .llama => &[_][]const u8{
                "token_embd.weight",
                "output_norm.weight",
                "output.weight",
            },
            .gpt => &[_][]const u8{
                "wte.weight",
                "ln_f.weight",
                "ln_f.bias",
            },
            else => &[_][]const u8{},
        };

        for (required_tensors) |tensor_name| {
            if (!self.hasTensor(tensor_name)) {
                std.log.err("Missing required tensor: {s}", .{tensor_name});
                return error.InvalidModel;
            }
        }
    }
};

/// Helper function to detect architecture from model name or tensors
pub fn detectArchitecture(model_name: []const u8, tensor_names: []const []const u8) Architecture {
    // Check model name first
    if (std.mem.indexOf(u8, model_name, "llama") != null) {
        return .llama;
    }
    if (std.mem.indexOf(u8, model_name, "gpt") != null) {
        return .gpt;
    }
    if (std.mem.indexOf(u8, model_name, "bert") != null) {
        return .bert;
    }

    // Check tensor names
    for (tensor_names) |name| {
        if (std.mem.indexOf(u8, name, "token_embd") != null) {
            return .llama;
        }
        if (std.mem.indexOf(u8, name, "wte") != null) {
            return .gpt;
        }
        if (std.mem.indexOf(u8, name, "embeddings") != null) {
            return .bert;
        }
    }

    return .unknown;
}

test "model metadata" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const metadata = Metadata{
        .name = try allocator.dupe(u8, "test-model"),
        .architecture = .llama,
        .vocab_size = 32000,
        .context_length = 2048,
        .embedding_dim = 4096,
        .num_layers = 32,
        .num_heads = 32,
        .num_kv_heads = null,
        .intermediate_size = null,
        .rope_freq_base = null,
        .rope_scaling = null,
    };

    var metadata_copy = metadata;
    defer metadata_copy.deinit(allocator);

    try testing.expect(metadata.architecture == .llama);
    try testing.expect(metadata.vocab_size == 32000);
}

test "architecture detection" {
    const arch1 = detectArchitecture("llama-2-7b", &[_][]const u8{});
    try std.testing.expect(arch1 == .llama);

    const arch2 = detectArchitecture("unknown", &[_][]const u8{"token_embd.weight"});
    try std.testing.expect(arch2 == .llama);

    const arch3 = detectArchitecture("unknown", &[_][]const u8{"wte.weight"});
    try std.testing.expect(arch3 == .gpt);
}

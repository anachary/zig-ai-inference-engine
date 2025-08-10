const std = @import("std");
const regs = @import("../../core/registries.zig");
const ir = @import("../../core/ir.zig");
const ll = @import("mod.zig");

fn init_thunk(allocator: std.mem.Allocator, model: *const ir.ModelDescriptor, model_path: []const u8, data_offset: usize) anyerror!*anyopaque {
    // Pass a stable copy-by-value of the model into the runtime to avoid dangling pointers
    var rt = try ll.runtime.LlamaRuntime.init(allocator, model, model_path, data_offset);
    return @ptrCast(rt);
}

fn forward_thunk(self: *anyopaque, tokens: []const u32, out_logits: []f32) anyerror!void {
    const rt: *ll.runtime.LlamaRuntime = @ptrCast(@alignCast(self));
    return rt.forward(tokens, out_logits);
}

fn deinit_thunk(self: *anyopaque) void {
    const rt: *ll.runtime.LlamaRuntime = @ptrCast(@alignCast(self));
    rt.deinit();
}

pub fn register() !void {
    const rt: regs.ArchitectureRuntime = .{
        .init = init_thunk,
        .forward = forward_thunk,
        .deinit = deinit_thunk,
    };
    try regs.registerArchitecture("llama", rt);
}

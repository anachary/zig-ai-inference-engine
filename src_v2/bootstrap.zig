const regs = @import("core/registries.zig");
const gguf_parser = @import("formats/gguf/parser.zig");
const gguf_tok = @import("tokenizers/gguf_vocab.zig");
const sampler_preset = @import("runtime/sampler.zig");
const tfms = @import("runtime/sampling/transforms.zig");
const sels = @import("runtime/sampling/selectors.zig");
const attn = @import("ops/attention.zig");
const rope = @import("ops/rope.zig");

const llama_reg = @import("models/llama/register.zig");
const std = @import("std");

var g_initialized: bool = false;

pub fn initAll(alloc: std.mem.Allocator) !void {
    if (g_initialized) return;
    regs.init(alloc);
    // Registrations
    try gguf_parser.register(alloc);
    try gguf_tok.register();
    try sampler_preset.registerGreedy();
    // Algorithm impl registrations
    try regs.registerNorm("rmsnorm", .{ .apply = @import("ops/normalization.zig").rmsnorm });
    try regs.registerRope("rope:llama", .{ .apply = rope.apply_rope_llama });
    try regs.registerAttention("attn:cpu:causal", .{ .compute = attn.attention_decode_once });
    try regs.registerActivation("act:swiglu", .{ .swiglu = struct {
        fn f(x: []f32, gate: []const f32) void {
            var i: usize = 0;
            while (i < x.len) : (i += 1) x[i] = (0.5 * x[i] * (1.0 + std.math.tanh(@as(f32, std.math.sqrt(2.0)) * (x[i] + 0.044715 * x[i] * x[i] * x[i])))) * gate[i];
        }
    }.f });

    try tfms.registerAll();
    try sels.registerAll();
    try llama_reg.register();
    g_initialized = true;
}

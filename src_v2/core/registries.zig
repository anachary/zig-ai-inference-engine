const std = @import("std");
const types = @import("types.zig");
const ir = @import("ir.zig");

// Interfaces (protocols)
pub const ParseResult = struct { md: ir.ModelDescriptor, data_offset: usize };

pub const FormatParser = struct {
    supports: *const fn (path: []const u8, head_bytes: []const u8) bool,
    parse: *const fn (allocator: std.mem.Allocator, path: []const u8) anyerror!ParseResult,
};

pub const ArchitectureRuntime = struct {
    init: *const fn (allocator: std.mem.Allocator, model: *const ir.ModelDescriptor, model_path: []const u8, data_offset: usize) anyerror!*anyopaque,
    forward: *const fn (self: *anyopaque, tokens: []const u32, out_logits: []f32) anyerror!void,
    deinit: *const fn (self: *anyopaque) void,
};

pub const Tokenizer = struct {
    tokenize: *const fn (allocator: std.mem.Allocator, input: []const u8) anyerror![]u32,
    detokenize: *const fn (allocator: std.mem.Allocator, tokens: []const u32) anyerror![]u8,
};

// High-level sampler preset (optional convenience)
pub const Sampler = struct {
    sample: *const fn (logits: []const f32, params: types.SamplingParams) u32,
};

// Sampling pipeline components
pub const SamplerCtx = struct {
    params: types.SamplingParams,
    recent: []const u32, // recent token IDs for penalties
    rng: ?*std.rand.Random, // multinomial needs RNG; null => deterministic fallback
};

pub const LogitTransform = struct {
    apply: *const fn (logits: []f32, ctx: SamplerCtx) void,
};

pub const Selector = struct {
    select: *const fn (logits: []const f32, ctx: SamplerCtx) u32,
};
// Algorithm registries (norm, rope, attention, activation)
pub const NormImpl = struct { apply: *const fn (x: []f32, eps: f32) void };
pub const RopeImpl = struct { apply: *const fn (q: []f32, k: []f32, head_dim: usize, pos: usize, rope_theta: f32) void };
pub const AttentionImpl = struct {
    // Computes attention output for single step t with KV cache
    compute: *const fn (
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        t: usize,
        q: []const f32,
        k_cache: []const f32,
        v_cache: []const f32,
        out: []f32,
    ) void,
};

pub const ActivationImpl = struct { swiglu: *const fn (x: []f32, gate: []const f32) void };

var g_norms: std.StringHashMap(NormImpl) = undefined;
var g_ropes: std.StringHashMap(RopeImpl) = undefined;
var g_attn: std.StringHashMap(AttentionImpl) = undefined;
var g_acts: std.StringHashMap(ActivationImpl) = undefined;

// Registries (simple runtime maps)
var g_format_parsers: std.StringHashMap(FormatParser) = undefined;
var g_architectures: std.StringHashMap(ArchitectureRuntime) = undefined;
var g_tokenizers: std.StringHashMap(Tokenizer) = undefined;
var g_samplers: std.StringHashMap(Sampler) = undefined; // presets
var g_logit_transforms: std.StringHashMap(LogitTransform) = undefined;
var g_selectors: std.StringHashMap(Selector) = undefined;

pub fn init(alloc: std.mem.Allocator) void {
    g_format_parsers = std.StringHashMap(FormatParser).init(alloc);
    g_architectures = std.StringHashMap(ArchitectureRuntime).init(alloc);
    g_tokenizers = std.StringHashMap(Tokenizer).init(alloc);
    g_samplers = std.StringHashMap(Sampler).init(alloc);
    g_logit_transforms = std.StringHashMap(LogitTransform).init(alloc);
    g_selectors = std.StringHashMap(Selector).init(alloc);
    g_norms = std.StringHashMap(NormImpl).init(alloc);
    g_ropes = std.StringHashMap(RopeImpl).init(alloc);
    g_attn = std.StringHashMap(AttentionImpl).init(alloc);
    g_acts = std.StringHashMap(ActivationImpl).init(alloc);
}

pub fn registerFormat(key: []const u8, parser: FormatParser) !void {
    try g_format_parsers.put(key, parser);
}
pub fn registerArchitecture(key: []const u8, arch: ArchitectureRuntime) !void {
    try g_architectures.put(key, arch);
}
pub fn registerTokenizer(key: []const u8, tok: Tokenizer) !void {
    try g_tokenizers.put(key, tok);
}
pub fn registerSampler(key: []const u8, s: Sampler) !void {
    try g_samplers.put(key, s);
}
pub fn registerLogitTransform(key: []const u8, t: LogitTransform) !void {
    try g_logit_transforms.put(key, t);
}
pub fn registerNorm(key: []const u8, n: NormImpl) !void {
    try g_norms.put(key, n);
}
pub fn registerRope(key: []const u8, r: RopeImpl) !void {
    try g_ropes.put(key, r);
}
pub fn registerAttention(key: []const u8, a: AttentionImpl) !void {
    try g_attn.put(key, a);
}
pub fn registerActivation(key: []const u8, a: ActivationImpl) !void {
    try g_acts.put(key, a);
}

pub fn getNorm(key: []const u8) ?NormImpl {
    return g_norms.get(key);
}
pub fn getRope(key: []const u8) ?RopeImpl {
    return g_ropes.get(key);
}
pub fn getAttention(key: []const u8) ?AttentionImpl {
    return g_attn.get(key);
}
pub fn getActivation(key: []const u8) ?ActivationImpl {
    return g_acts.get(key);
}

pub fn registerSelector(key: []const u8, s: Selector) !void {
    try g_selectors.put(key, s);
}

pub fn resolveFormat(head_bytes: []const u8) ?FormatParser {
    var it = g_format_parsers.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.supports("", head_bytes)) return entry.value_ptr.*;
    }
    return null;
}

pub fn getArchitecture(key: []const u8) ?ArchitectureRuntime {
    return g_architectures.get(key);
}
pub fn getTokenizer(key: []const u8) ?Tokenizer {
    return g_tokenizers.get(key);
}
pub fn getSampler(key: []const u8) ?Sampler {
    return g_samplers.get(key);
}
pub fn getLogitTransform(key: []const u8) ?LogitTransform {
    return g_logit_transforms.get(key);
}
pub fn getSelector(key: []const u8) ?Selector {
    return g_selectors.get(key);
}

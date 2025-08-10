const std = @import("std");
const regs = @import("registries.zig");
const ir = @import("ir.zig");
const types = @import("types.zig");
const bootstrap = @import("../bootstrap.zig");
const sampling_factory = @import("../runtime/sampling/factory.zig");
const tok = @import("../tokenizers/gguf_vocab.zig");

pub const RuntimeSession = struct {
    model: ir.ModelDescriptor,
    model_path: []const u8 = "",
    data_offset: usize = 0,
    rt_ptr: ?*anyopaque = null,
};

pub fn loadModel(allocator: std.mem.Allocator, path: []const u8) anyerror!RuntimeSession {
    // Use page_allocator for global registries to avoid GPA leak reports
    try bootstrap.initAll(std.heap.page_allocator);

    // read head bytes
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var head: [512]u8 = undefined;
    const n = try file.read(&head);
    const head_bytes = head[0..n];

    const parser = regs.resolveFormat(head_bytes) orelse return types.Error.InvalidFormat;
    const parsed = try parser.parse(allocator, path);
    std.debug.print("INFO: parsed tensors_len={} vocab={} layers={} hidden={} ctx={}\n", .{ parsed.md.tensors.items.len, parsed.md.tokenizer.vocab_size, parsed.md.num_layers, parsed.md.hidden_dim, parsed.md.context_length });
    // One-time histogram of ggml_type_id values for planning support
    var type_counts = std.AutoHashMap(u32, usize).init(allocator);
    defer type_counts.deinit();
    for (parsed.md.tensors.items) |tm| {
        const entry = try type_counts.getOrPut(tm.ggml_type_id);
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }
    var it = type_counts.iterator();
    std.debug.print("INFO: ggml types in model:", .{});
    while (it.next()) |e| {
        std.debug.print(" [type_id={} count={}]", .{ e.key_ptr.*, e.value_ptr.* });
    }
    std.debug.print("\n", .{});

    const model_path_copy = try allocator.dupe(u8, path);
    tok.setModelPath(model_path_copy);

    // Prepare session first so the model storage is stable in memory
    var session: RuntimeSession = .{ .model = parsed.md, .model_path = model_path_copy, .data_offset = parsed.data_offset, .rt_ptr = null };

    // Initialize architecture runtime and keep pointer to session.model (stable address)
    const arch = regs.getArchitecture("llama") orelse return types.Error.InvalidArchitecture;
    const rt_ptr = try arch.init(std.heap.page_allocator, &session.model, model_path_copy, parsed.data_offset);
    session.rt_ptr = rt_ptr;

    return session;
}

pub fn deinit(session: *RuntimeSession, allocator: std.mem.Allocator) void {
    // Deinit architecture runtime if provided
    if (session.rt_ptr) |ptr| {
        if (regs.getArchitecture("llama")) |arch| arch.deinit(ptr);
        session.rt_ptr = null;
    }
    // Free model path copy
    if (session.model_path.len != 0) allocator.free(session.model_path);
    // Free IR allocations
    ir.deinit(&session.model, allocator);
}

pub fn forward(session: *RuntimeSession, tokens: []const u32, out_logits: []f32) anyerror!void {
    // Use persistent runtime instance from session
    const arch = regs.getArchitecture("llama") orelse return types.Error.InvalidArchitecture;
    const rt_ptr = session.rt_ptr orelse return types.Error.InvalidArchitecture;
    return arch.forward(rt_ptr, tokens, out_logits);
}

pub fn generate(session: *RuntimeSession, prompt: []const u8, params: types.SamplingParams, writer: anytype) anyerror!void {
    // Build pipeline from params
    var gpa = std.heap.page_allocator;
    const pipeline = try sampling_factory.buildPipeline(gpa, params);
    defer gpa.free(pipeline.transforms);

    // Tokenize input
    const tokens = try tok.tokenize(gpa, prompt);
    defer gpa.free(tokens);

    // Forward to get logits for last token
    // Use output weight matrix size as vocab size since tokenizer vocab_size is 0
    const rt_ptr = session.rt_ptr orelse return types.Error.InvalidArchitecture;
    const runtime: *@import("../models/llama/runtime.zig").LlamaRuntime = @ptrCast(@alignCast(rt_ptr));
    const vocab_size = session.model.tensors.items[runtime.config.output_weight].shape[1];
    var logits = try gpa.alloc(f32, vocab_size);
    defer gpa.free(logits);
    try forward(session, tokens, logits);

    // Apply transforms in order
    var ctx: regs.SamplerCtx = .{ .params = params, .recent = tokens, .rng = null };
    for (pipeline.transforms) |t| t.apply(logits, ctx);

    // If multinomial is selected, convert logits to probs via softmax
    var use_softmax = true;
    _ = use_softmax;
    // Simple heuristic: multinomial needs probs; greedy does not
    // We cannot inspect function pointers easily; assume softmax needed unless greedy key used.
    // For now, compute softmax unconditionally and it won't affect greedy correctness.
    var maxv: f32 = -3.4e38;
    for (logits) |v| {
        if (v > maxv) maxv = v;
    }
    var sum: f32 = 0;
    for (logits, 0..) |v, i| {
        const e = @exp(v - maxv);
        logits[i] = e;
        sum += e;
    }
    for (logits, 0..) |v, i| logits[i] = v / sum;

    const token = pipeline.selector.select(logits, ctx);
    const out = try tok.detokenize(gpa, &[_]u32{token});
    defer gpa.free(out);

    try writer.print("{s}", .{out});
}

pub fn generateN(session: *RuntimeSession, prompt: []const u8, params: types.SamplingParams, max_tokens: usize, writer: anytype) anyerror!void {
    var gpa = std.heap.page_allocator;

    // Tokenize input into a growing context
    var ctx_tokens = try tok.tokenize(gpa, prompt);
    defer gpa.free(ctx_tokens);

    // Build pipeline from params
    const pipeline = try sampling_factory.buildPipeline(gpa, params);
    defer gpa.free(pipeline.transforms);

    // Logits buffer reused - use output weight matrix size as vocab size
    const rt_ptr = session.rt_ptr orelse return types.Error.InvalidArchitecture;
    const runtime: *@import("../models/llama/runtime.zig").LlamaRuntime = @ptrCast(@alignCast(rt_ptr));
    const vocab_size = session.model.tensors.items[runtime.config.output_weight].shape[1];
    var logits = try gpa.alloc(f32, vocab_size);
    defer gpa.free(logits);

    var step: usize = 0;
    while (step < max_tokens) : (step += 1) {
        // Forward on current context (measure forward-only time)
        const f0 = std.time.nanoTimestamp();
        try forward(session, ctx_tokens, logits);
        const f1 = std.time.nanoTimestamp();
        const fms = @divTrunc(f1 - f0, 1_000_000);

        // Apply transforms in order
        var sctx: regs.SamplerCtx = .{ .params = params, .recent = ctx_tokens, .rng = null };
        for (pipeline.transforms) |t| t.apply(logits, sctx);

        // Softmax (for multinomial; harmless for greedy)
        var maxv: f32 = -3.4e38;
        for (logits) |v| {
            if (v > maxv) maxv = v;
        }
        var sum: f32 = 0;
        for (logits, 0..) |v, i| {
            const e = @exp(v - maxv);
            logits[i] = e;
            sum += e;
        }
        for (logits, 0..) |v, i| logits[i] = v / sum;
        // Optional: per-token latency print
        const t0 = std.time.nanoTimestamp();

        const next_id = pipeline.selector.select(logits, sctx);
        // Write detokenized piece
        const piece = try tok.detokenize(gpa, &[_]u32{next_id});
        defer gpa.free(piece);
        try writer.print("{s}", .{piece});

        // Append to context
        var new_ctx = try gpa.alloc(u32, ctx_tokens.len + 1);
        const t1 = std.time.nanoTimestamp();
        const ms = @divTrunc(t1 - t0, 1_000_000);
        try writer.print("", .{}); // ensure something is written before timing print if buffered
        std.debug.print(" [t{d}: {d}ms fwd:{d}ms]", .{ step + 1, ms, fms });

        std.mem.copy(u32, new_ctx[0..ctx_tokens.len], ctx_tokens);
        new_ctx[ctx_tokens.len] = next_id;
        gpa.free(ctx_tokens);
        ctx_tokens = new_ctx;
    }
}

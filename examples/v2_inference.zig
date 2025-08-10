const std = @import("std");
const api = @import("src_v2");
const core = api.core;

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    // Defaults
    var model_path: []const u8 = "models/llama-2-7b-chat.gguf";
    var prompt: []const u8 = "Hello";
    var temperature: f32 = 0.0;
    var top_k: u32 = 0;
    var top_p: f32 = 1.0;
    var seed: u64 = 0;
    var max_tokens: usize = 8;
    var cache_mb: usize = 128;

    // Parse simple CLI args: --model, --prompt, --temperature, --max-tokens
    var args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);
    var i: usize = 1; // skip exe name
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--model")) {
            if (i + 1 >= args.len) return error.MissingModelPath;
            i += 1;
            model_path = args[i];
        } else if (std.mem.eql(u8, a, "--prompt")) {
            if (i + 1 >= args.len) return error.MissingPrompt;
            i += 1;
            prompt = args[i];
        } else if (std.mem.eql(u8, a, "--temperature")) {
            if (i + 1 >= args.len) return error.MissingTemperature;
            i += 1;
            temperature = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--top-k")) {
            if (i + 1 >= args.len) return error.MissingTopK;
            i += 1;
            top_k = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--top-p")) {
            if (i + 1 >= args.len) return error.MissingTopP;
            i += 1;
            top_p = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, a, "--seed")) {
            if (i + 1 >= args.len) return error.MissingSeed;
            i += 1;
            seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            if (i + 1 >= args.len) return error.MissingMaxTokens;
            i += 1;
            max_tokens = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "--cache-mb")) {
            if (i + 1 >= args.len) return error.MissingCacheMB;
            i += 1;
            cache_mb = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            std.debug.print("Usage: v2-inference [--model PATH] [--prompt TEXT] [--temperature F32] [--top-k N] [--top-p F32] [--seed U64] [--max-tokens N] [--cache-mb MB]\n", .{});
            return;
        }
    }

    // Apply weight cache size
    api.setWeightCacheCapMB(cache_mb);

    var session = try core.loadModel(gpa, model_path);

    var buf = std.ArrayList(u8).init(gpa);
    defer buf.deinit();
    var writer = buf.writer();

    // Measure per-token latency by wrapping writer to timestamp writes
    const t0 = std.time.nanoTimestamp();
    try core.generateN(&session, prompt, .{ .temperature = temperature, .top_k = top_k, .top_p = top_p, .seed = seed }, max_tokens, writer);
    const t1 = std.time.nanoTimestamp();

    std.debug.print("\nGenerated ({d} tokens) in {d} ms\n", .{ max_tokens, @divTrunc(t1 - t0, 1_000_000) });
    std.debug.print("Output: {s}\n", .{buf.items});
}

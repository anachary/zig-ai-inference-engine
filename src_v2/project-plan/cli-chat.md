## CLI Chat Design (src_v2)

Command
- zig-ai chat --model <path.gguf> [--temperature 0.7] [--top-k 40] [--max-tokens 256]

Flow
1) Parse args
2) session = loadModel(model_path)
3) prompt loop
   - read user input
   - tokens = tokenizer.tokenize(system + user prompt formatting if needed)
   - generate(session, tokens, params)
   - stream detokenized pieces to stdout

Streaming
- After each sampled token, detokenize incrementally
- Handle BOS/EOS correctly from tokenizer spec

Params
- Temperature, Top-K, Top-P (later), Repetition penalty (later)

Testing
- `zig build test` for chat module: simulate stdin, assert output contains known continuations for greedy

Note
- CLI lives in src_v2/cli, but depends only on public API in src_v2/core/api.zig


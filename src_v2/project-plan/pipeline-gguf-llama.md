## Pipeline Plan: GGUF + LLaMA (First E2E)

Goal
- Load llama-2-7b-chat.gguf, detect architecture from GGUF metadata, build IR, execute LLaMA forward pass, generate tokens, CLI chat.

Steps
1) Scaffolding
   - Create src_v2 skeleton folders (see architecture.md)
   - Add registries with stubs and unit tests (no logic yet)
2) GGUF Parser (MVP)
   - Read header, metadata_kv, tensor directory (names, shapes, ggml types, offsets)
   - Map metadata → ModelDescriptor:
     - architecture: "llama"
     - dims: n_layer, n_head, head_dim, hidden_dim, ffn_dim, context_length
     - rope params, norm type (RMSNorm), vocab size
     - tokenizer spec from embedded vocab
   - Build TensorMap for: token_embd.weight, blk.* tensors, output.weight
   - Write accurate errors if keys missing
3) Tokenizer (GGUF embedded)
   - Extract vocab from metadata (tokens array, merges if BPE)
   - Provide tokenize/detokenize compatible with llama tokenizer present
   - Verify against a few known strings
4) LLaMA Runtime
   - Implement runtime.llama with:
     - RMSNorm, RoPE, Multi-Head Attention (causal), SwiGLU FFN, residuals
     - KV cache (per layer K,V) for autoregressive decoding
   - Use ops kernels (matmul, attention, activations)
   - Accept TensorViews from WeightStore; dequantize per-use or lazily
5) Sampling (MVP)
   - Greedy and temperature + top-k; add top-p later
6) CLI Chat (MVP)
   - `zig-ai chat --model models/llama-2-7b-chat.gguf`
   - Stream tokens, stop at EOS or max_tokens
   - Params: temperature, top-k, max_tokens
7) Validation
   - Sanity checks on logits range, deterministic greedy path
   - Run short prompt and compare against stable baseline snippets
8) Performance passes (as time allows)
   - Preallocate buffers, SIMD matmul, fused QKV path, KV cache correctness

Deliverables
- Working llama pipeline using models/llama-2-7b-chat.gguf
- Unit tests for parser, tokenizer, ops, norm, attention, ffn, sampler
- CLI chat working demo


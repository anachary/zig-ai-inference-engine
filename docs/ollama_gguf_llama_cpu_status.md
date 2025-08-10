# Single-node CPU LLaMA (GGUF) Inference – Status and Pipeline (Ollama-style models)

This document summarizes the current end-to-end inference pipeline for LLaMA GGUF models (as distributed by Ollama), what each stage does, why it matters, and our implementation status. It also outlines what remains to reach production readiness and a short execution plan.

## Summary
- End-to-end decode on CPU works today for LLaMA GGUF models with the quant types we currently support.
- Tokenizer (GGUF-embedded) is implemented, leak-free, and stress-tested.
- Runtime implements full single-step forward with KV cache, RoPE (theta via GGUF when present), and attention (MHA/GQA).
- Dequantization uses a small LRU cache; CLI exposes cache size.
- Sampling supports greedy/top-k/top-p; per-token timing (total and forward-only) is printed.
- Remaining to be “production-ready”: expand quant coverage, pin dequantized weights per layer, golden correctness tests, penalties, packaging/API, and basic performance optimizations.

## Inference flow diagram

- Source: docs/diagrams/ollama_gguf_llama_flow.mmd
- PNG location (after export): docs/images/ollama_gguf_llama_flow.png

To generate the PNG locally, run (PowerShell):

- scripts: `docs/images/render-ollama-flow.ps1 -Format png`

![Ollama GGUF LLaMA Flow](images/ollama_gguf_llama_flow.png)

## Pipeline stages (what/why/status)

1) GGUF metadata parse
- What: Validate magic; read header + metadata (tensor count, tokenizer data, hyperparams: n_layer, n_head, n_kv_head, hidden/head_dim, context length, rope.theta, etc.).
- Why: Shapes, attention head mapping (GQA/MQA), tokenizer consistency, and RoPE behavior rely on correct metadata.
- Status: Implemented for required fields; rope.theta parsed when present; tolerant to some header versions. Coverage of optional keys partial (~80%).

2) Tokenizer load (from GGUF)
- What: Load tokenizer.ggml.tokens and tokenizer.ggml.merges; build piece→id and merge-rank maps. Arena mode available for fast teardown.
- Why: Produces the u32 ids the model expects; mismatched tokenizer => wrong ids => bad outputs.
- Status: Implemented, leak-free (unit + Unicode + stress tests); normalization to U+2581; BPE merges (~90%).

3) Session init & runtime config
- What: Configure hidden sizes, n_layer, n_head, n_kv_head, context length; allocate per-layer KV cache; set position.
- Why: Correct KV/attention shapes and GQA mapping depend on this.
- Status: Implemented for LLaMA; KV cache time-major; indexing validated (~85%).

4) Weight access & dequantization
- What: On-demand read of quantized tensors (embeddings, Wq/Wk/Wv/Wo/Gate/Up/Down, Wout); dequant to f32; small LRU cache.
- Why: Avoid repeated dequant; correctness depends on dequant kernels.
- Status: Implemented with LRU cache; supported types include f32, f16, q8_0, q4_k. Other common GGUF quants still to add (~70%).

5) Forward pass per token (decode)
- What: Embed → RMSNorm → Q/K/V → RoPE (theta from GGUF if present) → causal attention (uses cached K,V) → Wo + residual → RMSNorm → FFN (SwiGLU) + residual → logits.
- Why: Core inference path; determines quality.
- Status: Implemented for LLaMA; GQA/MQA via n_kv_head; numerics not yet golden-validated; performance baseline only (~70%).

6) KV cache
- What: Append new K,V at each layer for position t; attend over 0..t.
- Why: Decoding efficiency; correctness relies on consistent strides.
- Status: Implemented; per-token timing shows cache benefits (~85%).

7) Sampling
- What: Greedy/top-k/top-p (penalties to be added: repetition/presence/frequency).
- Why: Turns logits into next token id; affects user-visible behavior.
- Status: Implemented greedy/top-k/top-p (~80%).

8) Detokenization & streaming
- What: Map ids back to bytes; replace U+2581 with space for display; stream pieces.
- Why: User-visible output.
- Status: Implemented; per-token total and forward-only timing printed (~90%).

9) CLI & controls
- What: Example runner with flags: --prompt, --temperature, --top-k, --top-p, --seed, --max-tokens, --cache-mb.
- Why: Manual QA, demo, scripting.
- Status: Implemented (~80%).

10) Tests & validation
- What: Tokenizer unit, Unicode test (with safe fallback), stress test (arena), forward smoke (shape/logit length + 1-token generation); leak checks.
- Why: Baseline confidence.
- Status: Implemented and passing; missing golden logits and broader model coverage (~70%).

## What runs today with Ollama LLaMA GGUF
- CPU decode works end-to-end for LLaMA models using supported quants (f32/f16/q8_0/q4_k).
- If model uses unsupported quant (e.g., q4_0/q5_1/q6_k variants), load/dequant will fail until added.

## Readiness estimate (Ollama LLaMA, single-node CPU)
- Functional: 60–70% (works but quant coverage and golden numerics remain)
- Performance: 20–30% (baseline; needs weight pinning and simple SIMD)
- Robustness: ~60% (improve error messages, shape/key checks)
- Packaging/API: ~20% (shared library/ABI not finalized)

## What remains to reach “production-ready”
- Quant coverage: implement q4_0/q5_1/q6_k and other common GGUF formats with tests
- Golden correctness: tiny GGUF with known outputs; deterministic greedy checks across prompts
- Performance: pin dequantized per-layer weights; add SIMD matmul/attention; consider basic threading
- Sampling penalties: repetition/presence/frequency penalties + CLI flags
- Observability & errors: clear messages on unsupported quants/types; structured logs/metrics
- Packaging/API: stable C ABI + shared library target; versioning and docs

## Suggested short plan (2–3 weeks)
- Week 1
  - Add GGUF quant kernels used by Ollama (q4_0, q5_1, q6_k)
  - Add per-layer weight pinning cache; measure token-2+ latency improvement
- Week 2
  - Golden tests for a tiny LLaMA GGUF; deterministic greedy checks on 3–5 prompts
  - Implement repetition/presence/frequency penalties and CLI flags
- Week 3
  - Simple SIMD matmul kernel via registries (f32 baseline) + basic perf smoke harness
  - Harden error messages and capability checks (clear "unsupported quant" guidance)

## How to run (today)
- Build:
  - `zig build v2-inference`
- Run:
  - `zig-out/bin/v2-inference --prompt "Hello" --temperature 0.0 --top-k 0 --top-p 1.0 --max-tokens 16 --cache-mb 256`
- Tests:
  - `zig build v2-tokenizer-test`
  - `zig build v2-tokenizer-unicode-test`
  - `zig build v2-tokenizer-stress-test`
  - `zig build v2-smoke-forward-test`


## Gap Analysis: Path to LLaMA Chat (src_v2)

Overview
- Goal: End-to-end LLaMA chat via src_v2 with real inference (no fallbacks)
- Starting point: New modular architecture, GGUF parser, sampling pipeline, quantization, and weight store are in place
- This doc identifies remaining gaps, risks, and concrete next steps

Visual summary
- See gap diagram below (PNG). Source Mermaid file: images/src/llama-gap-analysis.mmd

![LLaMA Gap Analysis](images/llama-gap-analysis.png)

Current status by component
- Format & IR
  - GGUF detection + parse: DONE (header, metadata, tensor directory, offsets)
  - IR richness (norm, activation, attention kind, rope): DONE
  - Data offset surfaced in IR/TensorMeta: DONE (ggml_type_id in TensorMeta)
- Quantization & weights
  - Quant modules (f16, q8_0, q4_k_m) copied to src_v2: DONE
  - WeightStore (open, dequantizeInto): DONE (needs safe quantized_size derivation helper)
- Tokenizer
  - GGUF tokenizer loading (tokens, merges, special tokens): GAP (Not Started)
  - tokenize/detokenize parity with model: GAP (Not Started)
- Runtime (LLaMA)
  - Config mapping (tensor name -> view): GAP (Not Started)
  - Core ops glue (RMSNorm, RoPE, MHA, SwiGLU, residuals): GAP (Not Started)
  - KV cache: GAP (Not Started)
  - Forward() returning logits: GAP (Not Started)
- Sampling
  - Registries, transforms (temperature, top-k/top-p, penalties, min-p): DONE
  - Selectors (greedy, multinomial): DONE
  - Pipeline factory and integration in generate(): PARTIAL (uses fake logits until runtime wired)
- CLI
  - Chat CLI stub in src_v2: DONE (placeholder)
  - Wire to tokenizer + runtime + sampling for streaming: GAP (Blocked on tokenizer/runtime)
- Tests & validation
  - Parser smoke tests: GAP (Not Started)
  - Tokenizer tests: GAP (Not Started)
  - Ops and runtime unit tests: GAP (Not Started)
  - E2E smoke: GAP (Not Started)
- Performance & memory
  - On-demand dequant OK for MVP; caching/fusion later: GAP (Planned later)

Key risks
- Tokenizer correctness from GGUF metadata (BPE/SentencePiece nuances)
- GGML quant types variance across models (ensure at least q8_0, q4_k_m robust)
- Tensor naming variations across GGUF exports (map both token_embd/tok_embeddings, etc.)
- Memory pressure if naive dequant of large matrices each step (mitigate with caching)

Concrete next steps (MVP to first token)
1) Weight size helper: derive quantized_size from ggml_type_id + shape
2) Tokenizer: load vocab/merges, implement tokenize/detokenize; tests for a few strings
3) LLaMA runtime config: map tensor names to views (attn_q/k/v/o, norms, ffn)
4) LLaMA forward (single step): RMSNorm → QKV → RoPE → attention (causal) → output proj → pre-FFN norm → SwiGLU FFN → residuals → logits
5) Wire generate(): tokenize prompt, call forward, apply sampling, detokenize token; print/stream
6) E2E smoke: greedy single-token generation is deterministic and non-empty

Stretch (for chat usability)
- KV cache for fast multi-token decoding
- Streaming detokenization (partial piece handling)
- Basic stop conditions and EOS handling from tokenizer spec

Regenerate diagram
- Use: ./src_v2/project-plan/images/render-sequence-diagrams-advanced.ps1 -Format png -Theme dark -BackgroundColor "#0b1220" -UseDefaultConfig
- The Mermaid source is at images/src/llama-gap-analysis.mmd


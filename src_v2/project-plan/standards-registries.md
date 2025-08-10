## Standards: Registry-Per-Choice Rule

Rule
- If there is more than one valid algorithm or implementation for an activity, we expose it via a registry.
- Default behavior may be provided via a preset that selects from registries, but the underlying options remain pluggable.

Why
- Enforces SOLID (open/closed) with minimal coupling
- Enables format-/arch-agnostic composition
- Makes research-accurate alternatives easy to test and switch

Core registries (already planned/added)
- FormatParserRegistry (e.g., gguf, future formats)
- ArchitectureRegistry (llama, qwen, ...)
- TokenizerRegistry (gguf, sentencepiece, ...)
- SamplerRegistry (high-level presets)

New/extended registries to add
- Sampling pipeline
  - LogitTransformRegistry: temperature, repetition/presence/frequency penalties, top-k, top-p, min-p, logit bias
  - SelectorRegistry: greedy (argmax), multinomial, (future: beam)
  - SamplerPresetRegistry (optional): named strategies → pipeline assembly
- Ops/Kernels (algorithm-level)
  - ActivationRegistry: swiglu, gelu, silu, relu
  - NormRegistry: rmsnorm, layernorm
  - PositionalEncodingRegistry: rope (llama/ntk-aware/x_pos), alibi, none
  - AttentionKernelRegistry: mha/mqa/gqa variants and backend-optimized kernels
  - MatMulKernelRegistry: f32/f16/bf16/quant-aware + SIMD variants
  - QuantDequantRegistry: ggml quant families (Q4_K_M, Q5_K, Q8_0, ...)
- Runtime utilities
  - WeightLoaderRegistry: format-specific tensor loading/dequant strategies
  - KVCachePolicyRegistry: naive, paged, fused (future)

Guidelines
- Each registry has a small interface (struct of function pointers)
- Registration is explicit at bootstrap (no hidden singletons)
- Resolution uses stable string keys (e.g., "rope:llama", "attention:mqa")
- Architecture runtimes pull from registries; no hard-coded algorithms

Next steps (proposed)
1) Implement sampling registries (LogitTransform, Selector) + factory that assembles pipelines from SamplingParams
2) Add ActivationRegistry and NormRegistry; refactor ops to register implementations
3) Add PositionalEncodingRegistry (rope variants + alibi) with research-correct parameters
4) Introduce AttentionKernelRegistry and MatMulKernelRegistry stubs; wire pure Zig backends first

This standard complements research-first implementations (see standards-research-first.md).


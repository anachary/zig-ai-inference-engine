## Current Status (src_v2)

This document reflects the real, current state of src_v2. It supersedes any conflicting claims elsewhere in the repo.

### High-level
- Architecture: Modular, format/architecture-agnostic design in place (registries + IR)
- Target path: GGUF + LLaMA first, Qwen next
- Execution: Single-token forward path implemented with per-layer timing scaffolding

### Components
- Formats / GGUF
  - Parser: Implemented (header, metadata, tensor directory, offsets)
  - IR mapping: Implemented (ModelDescriptor with tensor map and rope theta)
- Quantization & Weight Access
  - Quant types: f32, f16, q8_0, q4_k family (initial) available
  - WeightStore: Implemented (open, dequantizeInto, embedding-column reads)
- Tokenizer
  - GGUF tokenizer: Partially wired via core; end-to-end parity not fully validated
  - Tokenize/detokenize: Working stubs in v2 path; needs golden tests for correctness
- Runtime (LLaMA)
  - Config build: Implemented
  - Forward pass: End-to-end structure present; per-layer timing variables and logging exist
  - KV cache: Implemented buffers; used for attention with time-major slices
  - Attention: Causal attention path invoked; numerics and performance need validation
  - FFN: SwiGLU path implemented using matmul + activation
  - Output projection: Implemented
- Ops
  - matmul: Pure Zig kernels in place; SIMD/threading optimizations pending
  - attention/norm/activation: Implemented and registered; require correctness/perf validation
- API / Runtime
  - core/api.zig exposes loadModel, forward, generate
  - Sampling: Greedy + top-k supported via pipeline; softmax applied
- CLI
  - Chat wiring exists (v2 chat); depends on tokenizer correctness and runtime stability

### Observability
- Per-token forward timing (fwd-only) printed in core/generate
- Per-layer detailed timing inside LLaMA runtime; now supports text or JSON lines

### Biggest gaps to production-readiness
- Tokenizer correctness: finalize GGUF tokenizer and add golden tests
- Quant coverage: add common GGUF quants (q4_0/q5_1/q6_k) with tests
- Performance: preallocate working buffers, avoid per-layer allocations, SIMD matmul/attention
- Validation: golden numerics for tiny models; deterministic greedy checks
- Packaging: stable C ABI DLL target and docs

### How to run (v2)
- Build: `zig build v2-inference`
- Run: `zig-out/bin/v2-inference --prompt "Hello" --temperature 0.0 --top-k 0 --top-p 1.0 --max-tokens 16`
- Enable JSON layer timings at runtime: call `setProfileMode(.json)` before forward/generate (temporary API)

### Next 1–2 weeks (proposed)
1) Observability
   - Emit JSONL per-layer timings guarded by a flag (done)
   - Add CLI flag to toggle profile mode and output file path
2) Correctness
   - Finish GGUF tokenizer and add unit + golden tests
   - Add tiny model smoke tests for forward numerics
3) Performance
   - Remove per-layer allocs for weights and temps; reuse buffers
   - Add basic SIMD to matmul kernels and/or thread the largest GEMMs
4) Polish
   - Stabilize C ABI, versioned DLL build, and user-facing docs


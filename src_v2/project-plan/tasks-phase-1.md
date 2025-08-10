## Tasks – Phase 1: Core, GGUF, LLaMA, CLI MVP

Outcomes
- src_v2 skeleton + registries
- GGUF parser → ModelDescriptor
- LLaMA runtime forward pass
- Tokenizer, basic sampler
- CLI chat MVP

Task List
1) Create src_v2 skeleton folders and placeholder files
   - core/{api.zig, types.zig, ir.zig, registries.zig, memory.zig}
   - formats/gguf/{parser.zig, tokenizer.zig}
   - models/{common/, llama/{config.zig,runtime.zig}}
   - ops/{matmul.zig,attention.zig,activations.zig,normalization.zig}
   - runtime/{session.zig,executor.zig,kv_cache.zig,sampler.zig}
   - tokenizers/{gguf_vocab.zig}
   - cli/{chat.zig}
2) Implement registries (runtime hash map)
   - Register GGUF parser; register LLaMA runtime; register GGUF tokenizer
3) GGUF parser MVP
   - Read header, metadata, tensor dir; map to IR; build TensorMap
   - Errors for missing metadata; unit tests with models/llama-2-7b-chat.gguf
4) Tokenizer MVP
   - Parse vocab from GGUF; simple tokenize/detokenize; tests
5) Ops MVP
   - matmul f32; attention f32 (no SIMD yet); RMSNorm; SiLU; SwiGLU gate
6) LLaMA runtime MVP
   - Forward for single step (no cache); logits for last token
   - Then add KV cache for autoregressive
7) Sampler MVP
   - greedy, temperature, top-k
8) CLI chat MVP
   - wire loadModel, generate loop, print tokens as text
9) Basic Bench + Validation
   - log timings; ensure deterministic greedy token for fixed prompt
10) Docs
   - update sequence-diagrams if any deviations; write quickstart


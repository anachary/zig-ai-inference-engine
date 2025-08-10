## Architecture: Modular, Format/Architecture-Agnostic Inference

Goals
- Clean separation: Format parsing → Model Descriptor (IR) → Runtime execution
- Registries for plug-and-play: formats, architectures, tokenizers, kernels, samplers
- Alignment with research papers for each architecture

Key Concepts
- ModelDescriptor (IR):
  - General: name, vocab_size, context_length, dtype, quantization
  - Architecture tag: e.g., llama, qwen
  - Layer config: num_layers, num_heads, head_dim, hidden_dim, ffn_dim, rope params, norm type
  - TensorMap: symbolic names → tensor metadata (shape, dtype, location/offset)
  - TokenizerSpec: vocab, merges/rules, special tokens
- RuntimeSession:
  - Holds weights (possibly lazily dequantized), KV cache, working buffers
  - Methods: forward(), logits(), sample(), generate()
- Registries:
  - FormatParserRegistry: .register("gguf", parser), resolve(file_bytes)
  - ArchitectureRegistry: .register("llama", impl), .register("qwen", impl)
  - TokenizerRegistry: maps tokenizer family to implementation
  - KernelRegistry: matmul, attention kernels for dtype/hw
  - SamplerRegistry: greedy, top-k, top-p, temperature

Module Boundaries (src_v2)
- core/
  - api.zig (public API surface)
  - types.zig (enums, errors)
  - ir.zig (ModelDescriptor, TensorMeta)
  - registries.zig (FormatParserRegistry, ArchitectureRegistry, ...)
  - memory.zig (alloc policy, arenas)
- formats/
  - gguf/
    - parser.zig (header+metadata+tensor dir → ModelDescriptor)
    - tokenizer.zig (vocab extraction from GGUF)
- models/
  - common/ (rope.zig, norm.zig, ffw.zig abstractions)
  - llama/
    - config.zig (derive from metadata)
    - runtime.zig (forward, kv cache)
  - qwen/
    - config.zig
    - runtime.zig
- ops/
  - matmul.zig, attention.zig, activations.zig, normalization.zig
- runtime/
  - session.zig (load(model_path) → FormatParser → IR → Arch runtime)
  - executor.zig (graph-ish execution coordination)
  - kv_cache.zig, sampler.zig
- tokenizers/
  - bpe.zig (if needed), gguf_vocab.zig
- cli/
  - chat.zig (wire to library)

Public API (draft)
- loadModel(allocator, path) → RuntimeSession
- forward(session, tokens[]) → logits[]
- generate(session, prompt, max_tokens, params) → stream tokens
- getMetadata(session) → ModelDescriptor summary

Error Handling
- Honest errors for unsupported ops/arch/quant types
- Validations at boundaries: parser → IR; IR → runtime config

Extensibility
- New format: implement FormatParser and register
- New architecture: implement ArchRuntime and register with key
- Alternative kernels: register MatMul/Attention for dtype/hw


## src_v2 Project Plan: Format- and Architecture-Agnostic Inference Library

Purpose
- Build a modular, plugin-style inference library in src_v2
- Format-agnostic (start with GGUF), architecture-agnostic (start with LLaMA → Qwen)
- Clean separation of concerns with SOLID-aligned interfaces and registries
- Deliver a working full pipeline and a CLI --chat mode using real model weights

Scope (Initial Milestones)
1) Phase 1: Bootstrapping and Core Architecture
   - Project skeleton, public API, registries, IR, and runtime session
2) Phase 2: GGUF parser + LLaMA architecture pipeline end-to-end
   - Detect architecture from GGUF metadata; run real forward pass; basic sampling
3) Phase 3: Extend to Qwen family
   - Detect Qwen via metadata; add differences (GQA/MQA, activation variances, etc.)
4) Phase 4: CLI chat experience
   - `zig-ai chat --model path.gguf` using tokenizer+sampler; streaming replies
5) Phase 5: Performance & Reliability
   - KV cache, quantization paths, SIMD kernels, tests, docs, and migration plan

Design Pillars
- Plugin modularity via registries: FormatParser, Architecture, Tokenizer, Sampler, Kernel Backend
- Strict boundary between Input Formats → Model Descriptor/IR → Runtime Execution
- Minimal, testable components; honest error messages; zero hidden fallbacks
- Practical usability: run the two provided GGUF models first

Folder Plan (src_v2)
- core/ (types, api, registries, IR)
- formats/ (gguf/ parser)
- models/ (llama/, qwen/, common/)
- ops/ (attention, matmul, activation, norm)
- runtime/ (session, executor, kv_cache, sampler)
- tokenizers/ (gguf tokenizer/vocab)
- cli/ (chat integration over the library)
- project-plan/ (this plan and technical docs)

Read Next
- architecture.md – high-level design, interfaces, IR
- registries-and-plugins.md – how plugins register and are resolved
- pipeline-gguf-llama.md – step-by-step plan to reach first E2E pipeline
- pipeline-gguf-qwen.md – extend to Qwen specifics
- sequence-diagrams.md – user and parsing flows (Mermaid)
- tasks-phase-1.md – actionable tasks checklists
- cli-chat.md – CLI chat mode design
- testing-and-migration.md – tests, validation, migration from src
- references.md – specs and research papers


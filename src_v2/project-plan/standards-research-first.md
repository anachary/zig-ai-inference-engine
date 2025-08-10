## Standards: Research-First Implementations

Principle
- When an algorithm has a known paper/spec, we implement it faithfully to that source, with doc-comments referencing the paper.

Scope
- Applies to attention (MHA/MQA/GQA), normalization (RMSNorm/LayerNorm), activations (SwiGLU, GELU, SiLU), positional encoding (RoPE variants), sampling, and kernel math.

Practices
- Add a top-of-file or function-level doc-comment with citation(s)
- Use variable/param names consistent with paper when reasonable
- Preserve numerical details (epsilon, scaling, masking)
- Unit tests include known-shape examples and corner cases from literature when available
- Prefer explicit errors over silent fallbacks when unsupported

Citations to prioritize
- Attention Is All You Need (Vaswani et al., 2017)
- RoPE/NTK-aware variants (Su et al., 2021 + community notes)
- LLaMA/LLaMA2 papers (Touvron et al.)
- Qwen2 Technical Report
- RMSNorm (Zhang & Sennrich, 2019), T5
- SwiGLU (Shazeer, 2020)

Change control
- Any deviation from papers must be documented with rationale (e.g., numeric stability, performance trade-offs)


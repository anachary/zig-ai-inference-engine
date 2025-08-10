## References (Specs and Research)

GGUF Format
- GGUF file format docs (llama.cpp repo)
- Tensor naming conventions in GGUF for LLaMA/Qwen

Transformer and Architectures
- Attention Is All You Need (Vaswani et al., 2017)
- LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
- LLaMA 2 (Touvron et al., 2023)
- Qwen2 Technical Report (Alibaba DAMO)
- RoPE: RoFormer (Su et al., 2021) and NTK-aware scaling notes (llama.cpp issues/discussions)
- RMSNorm: Zhang & Sennrich (2019), T5 paper
- SwiGLU: Shazeer (2020)

Quantization and Kernels
- ggml quantization formats (Q4_K_M, etc.) from ggml/llama.cpp
- Efficient attention kernels (FlashAttention paper) – for future work reference

Tokenizer
- GGUF-embedded tokenizer formats (BPE/SentencePiece variants used by LLaMA/Qwen)

Testing and Benchmarks
- ONNX Runtime style test categories for operator-level verification (inspiration)

Note
- Keep citations updated in code doc-comments where algorithms are implemented.


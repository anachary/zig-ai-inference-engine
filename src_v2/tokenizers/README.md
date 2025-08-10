Tokenizer plan (GGUF)
- Load tokenizer.ggml.tokens and tokenizer.ggml.merges from GGUF metadata/sections
- Implement tokenize/detokenize exactly per embedded data (no external deps)
- Provide BOS/EOS handling per spec
- Add tests on fixed strings


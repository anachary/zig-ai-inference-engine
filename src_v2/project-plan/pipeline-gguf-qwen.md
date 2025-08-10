## Pipeline Plan: Extend to Qwen (Next E2E)

Goal
- Detect Qwen from GGUF metadata and run inference with Qwen-specific differences.

Key Differences to Expect
- Potential Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)
- Slight differences in activation (SwiGLU variants), normalization usage
- Rope params and scaling may differ (NTK-aware variants)
- Vocabulary and tokenizer families differ

Steps
1) Parser Adjustments
   - Map general.architecture == "qwen" to Architecture.qwen
   - Extract qwen.* keys: n_layer, n_head, n_kv_head (for GQA), hidden_dim, ffn_dim, context_length, rope params
   - Validate tensor naming parity (blk.*.*) or map to Qwen names
2) Runtime.qwen
   - Support GQA/MQA: heads → kv_heads mapping during attention split
   - Confirm norm type; apply RMSNorm if specified, else LayerNorm
   - Use same ops kernels; only head partitioning and projections differ
3) Tokenizer
   - Provide Qwen tokenizer compatible with metadata
   - Validate with small strings
4) Tests
   - IR parsing tests for qwen metadata
   - Attention tests for GQA path (n_head != n_kv_head)
5) CLI
   - Auto-detect architecture and run seamlessly

Deliverables
- Qwen inference path with tests
- Head mapping correctness tests for GQA/MQA


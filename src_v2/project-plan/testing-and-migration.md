## Testing and Migration Plan

Testing Strategy
- Unit tests per module: parser, tokenizer, ops, runtime
- Golden tests for tokenization (fixtures)
- E2E smoke tests: load llama gguf → generate 8 tokens greedy → check token IDs
- Property tests (bounds, shapes) where feasible
- Benchmarks for matmul/attention

Safe-by-default Runs
- Parse-only tests do not allocate large buffers
- Minimal inference sequence lengths in CI

Migration from src → src_v2
- Keep src intact while building src_v2
- Copy reusable code into src_v2 with adjustments; do not import from src
- Once src_v2 LLaMA path is stable, create feature flag to switch CLI to src_v2
- Deprecate src after Qwen path solid and tests are green

Structure Parity
- Maintain similar directory names to ease future diffs
- Prefer clearer names in src_v2 (runtime/session vs inference)

Documentation
- Keep project-plan docs updated; link to research references


## Registries and Plugin Model

Objective
- Enable plug-and-play for formats, architectures, tokenizers, kernels, samplers

Interfaces (pseudo-Zig)
- FormatParser
  - supports(path: []const u8, head_bytes: []const u8) bool
  - parse(allocator, path: []const u8) !ModelDescriptor
- ArchitectureRuntime
  - init(allocator, ir: *const ModelDescriptor, weights: WeightStore) !Self
  - forward(self: *Self, tokens: []const u32, out_logits: []f32) !void
  - deinit(self: *Self) void
- Tokenizer
  - tokenize(input: []const u8) ![]u32
  - detokenize(tokens: []const u32) ![]u8
- Kernel
  - matmul(params...) !void
  - attention(params...) !void
- Sampler
  - sample(logits: []f32, params: SamplingParams) u32

Registry Patterns
- compile-time list or runtime hash map initialized at startup
- e.g., FormatParserRegistry.register("gguf", gguf.parser)
- resolveFormat(head_bytes) → key
- getArchitecture(ir.architecture_tag) → runtime impl

Initialization Flow
1) session.load(path)
2) read small head bytes → identify format via FormatParserRegistry
3) parser.parse(path) → ModelDescriptor + TensorMap + TokenizerSpec
4) tokenizer = TokenizerRegistry.get(spec.family)
5) arch = ArchitectureRegistry.get(ir.architecture)
6) weights = WeightStore.open(path, ir.tensor_map)
7) runtime = arch.init(ir, weights)

WeightStore
- Abstracts quantized data access and optional dequant cache
- get(name) → TensorView (shape, dtype, pointer/offset)

Notes
- Clear separation of parsing (no math) vs execution (pure math)
- Test registries independently with fakes/mocks


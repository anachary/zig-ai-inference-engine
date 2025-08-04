
src/
├── main.zig                 # Unified library entry point
├── core/                    # Core abstractions
│   ├── model.zig           # Universal Model interface
│   ├── tokenizer.zig       # Universal Tokenizer interface
│   ├── inference.zig       # Universal Inference interface
│   └── tensor.zig          # Universal Tensor type
├── formats/                 # ALL model formats
│   ├── mod.zig             # Format registry & detection
│   ├── gguf/               # GGUF (llama.cpp)
│   ├── onnx/               # ONNX
│   ├── safetensors/        # SafeTensors
│   ├── pytorch/            # PyTorch (.pth, .pt)
│   ├── tensorflow/         # TensorFlow (.pb, .tflite)
│   ├── huggingface/        # HuggingFace (.bin)
│   ├── mlx/                # Apple MLX
│   ├── coreml/             # Apple CoreML
│   └── common/             # Shared format utilities
├── tokenizers/              # ALL tokenizer types
│   ├── mod.zig             # Tokenizer registry
│   ├── bpe/                # Byte-Pair Encoding
│   ├── sentencepiece/      # SentencePiece
│   ├── tiktoken/           # OpenAI tiktoken
│   ├── wordpiece/          # BERT WordPiece
│   ├── unigram/            # Unigram
│   └── vocab/              # Universal vocabulary
├── inference/              # Universal inference engine
│   ├── mod.zig
│   ├── transformer/        # Transformer architectures
│   ├── diffusion/          # Diffusion models
│   ├── vision/             # Computer vision models
│   └── graph/              # Computation graph executor
├── compute/                # Hardware abstraction
├── memory/                 # Memory management
├── math/                   # Mathematical operations
└── utils/                  # Core utilities

build.zig
examples/                   # Format conversion examples
tests/                      # Comprehensive test suite


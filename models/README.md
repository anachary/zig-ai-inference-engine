# Models Directory

This directory contains AI models for testing the zig-ai-platform.

## Supported Formats

- **GGUF** (✅ Implemented) - llama.cpp format with quantization
- **ONNX** (🚧 Planned) - Industry standard format
- **SafeTensors** (🚧 Planned) - Modern safe format
- **PyTorch** (🚧 Planned) - Research format

## Example Models

For testing, you can download models from Hugging Face:

### Small Models (Good for Testing)
```bash
# Qwen2-0.5B (379 MB) - Fast inference
wget https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf

# TinyLlama (637 MB) - Very fast
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q4_k_m.gguf
```

### Larger Models (More Capable)
```bash
# Llama-2-7B (3.2 GB) - High quality
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.q4_k_m.gguf

# Mistral-7B (4.1 GB) - Latest architecture
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.q4_k_m.gguf
```

## Usage

Once you have models in this directory, you can use them with the CLI:

```bash
# Detect model format
zig build run -- detect models/your-model.gguf

# Start interactive chat
zig build run -- chat models/your-model.gguf
```

## Note

Model files are excluded from git due to their large size. Download them locally for testing.

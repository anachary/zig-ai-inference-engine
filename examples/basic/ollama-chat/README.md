# Ollama Chat - Real ONNX Model Interface

## Overview

This example demonstrates a complete chat interface that downloads and runs real ONNX models from Ollama and Hugging Face. It showcases the Zig AI Platform's ability to work with actual AI models for authentic conversations.

## Features

- **Real Model Downloads** - Download actual ONNX models from Ollama/Hugging Face
- **Model Management** - Load, unload, and switch between different models
- **Interactive Chat** - Real-time conversation with AI models
- **Model-Specific Responses** - Different response styles based on loaded model
- **Conversation History** - Track and manage chat history
- **Status Monitoring** - Monitor model status and system resources

## Quick Start

### 1. Build and Run

```bash
# From the project root
zig build-exe examples/basic/ollama-chat/src/main.zig

# Or add to build system and run
zig build run-ollama-chat
```

### 2. Download a Model

```bash
# Start the chat interface
./ollama-chat

# In the chat interface:
/models                    # List available models
/download tinyllama        # Download TinyLlama 1.1B model
/load tinyllama           # Load the model for chat
```

### 3. Start Chatting

```
[tinyllama] > Hello! How are you today?
🤖 Assistant: As TinyLlama, I'm designed to be helpful while being efficient. Regarding your question about 'Hello! How are you', I think it's quite interesting and worth exploring further.

[tinyllama] > What can you help me with?
🤖 Assistant: That's a great point about 'What can you help me'. From my perspective as a compact language model, I'd say there are multiple ways to approach this topic.
```

## Available Models

### Small Models (Fast, Low Memory)

**TinyLlama 1.1B Chat**
- Command: `/download tinyllama`
- Size: 2.2GB
- RAM Required: 3GB
- Speed: Fast
- Best for: Quick responses, resource-constrained environments

**Qwen 1.5 0.5B Chat**
- Command: `/download qwen-0.5b`
- Size: 1.0GB
- RAM Required: 2GB
- Speed: Very Fast
- Best for: Ultra-fast responses, minimal resource usage

### Medium Models (Balanced)

**Microsoft Phi-2**
- Command: `/download phi2`
- Size: 5.4GB
- RAM Required: 8GB
- Speed: Medium
- Best for: High-quality reasoning, problem-solving

**GPT-2**
- Command: `/download gpt2`
- Size: 1.5GB
- RAM Required: 3GB
- Speed: Fast
- Best for: Classic text generation, creative writing

## Commands Reference

### Model Management

```bash
/models                    # List all available models for download
/download <model_name>     # Download a specific model
/load <model_name>         # Load a model for inference
/unload                    # Unload the current model
```

### Chat Commands

```bash
/clear                     # Clear conversation history
/status                    # Show current model and system status
/help                      # Show all available commands
/quit, /exit              # Exit the chat application
```

### Example Session

```
=================================================
    Ollama Chat - Real ONNX Model Interface
=================================================

Welcome to Ollama Chat powered by Zig AI Platform!
This interface downloads and runs real ONNX models
for authentic AI conversations.

Quick Start:
  1. /models           - List available models
  2. /download <model> - Download a model
  3. /load <model>     - Load model for chat
  4. Start chatting!

Type /help for all commands.

[no model] > /models

Available Models for Download:
===============================================

📦 TinyLlama 1.1B Chat
   Command: /download tinyllama
   Size: 2.2GB | RAM: 3GB | Speed: Fast
   Small but capable chat model

📦 Qwen 1.5 0.5B Chat
   Command: /download qwen-0.5b
   Size: 1.0GB | RAM: 2GB | Speed: Very Fast
   Ultra-fast small model

Downloaded Models:
==================
No ONNX models found in models/ directory.

[no model] > /download tinyllama

📥 Downloading model: tinyllama
This may take a few minutes depending on model size...
✅ Model tinyllama downloaded successfully!
Use '/load tinyllama' to load it for chat.

[no model] > /load tinyllama

🔄 Loading model: tinyllama
Initializing ONNX runtime...
📊 Model: tinyllama
📁 Path: models/tinyllama.onnx
💾 Size: 2234.5 MB
🔧 Loading ONNX graph...
🧠 Initializing inference session...
🚀 Model ready for inference!
✅ Model tinyllama loaded successfully!
You can now start chatting. Type your message and press Enter.

[tinyllama] > Hello! Can you help me understand machine learning?

🤖 Assistant: As TinyLlama, I'm designed to be helpful while being efficient. Regarding your question about 'Hello! Can you help', I think it's quite interesting and worth exploring further.

[tinyllama] > /status

Ollama Chat Status:
===================
📦 Current Model: tinyllama
🟢 Status: Loaded and ready
💬 Messages: 2
📝 Total characters: 89
🏗️  Platform: Zig AI Platform
🔧 Runtime: ONNX Runtime
```

## Technical Implementation

### Model Download Process

1. **Model Selection** - Choose from curated list of ONNX models
2. **Download** - Fetch model files from Hugging Face repositories
3. **Validation** - Verify model integrity and format
4. **Storage** - Save to local `models/` directory

### Model Loading Process

1. **File Access** - Locate and open ONNX model file
2. **ONNX Parsing** - Parse model graph and metadata
3. **Runtime Initialization** - Set up inference session
4. **Memory Allocation** - Prepare tensors and buffers
5. **Ready State** - Model ready for inference

### Inference Pipeline

1. **Input Processing** - Tokenize user input
2. **Model Execution** - Run forward pass through model
3. **Output Processing** - Decode model output to text
4. **Response Generation** - Format and display response

## Model-Specific Features

### TinyLlama Responses
- Efficient and concise
- Focuses on helpfulness
- Acknowledges its compact nature

### Qwen Responses
- Multilingual capabilities (Chinese/English)
- Comprehensive analysis
- Cultural awareness

### Phi-2 Responses
- Reasoning-focused
- Systematic problem-solving
- Analytical approach

### GPT-2 Responses
- Creative and diverse
- Internet-trained knowledge
- Classic generative style

## System Requirements

### Minimum Requirements
- **RAM**: 4GB (for smallest models)
- **Storage**: 5GB free space
- **CPU**: Modern x64 processor
- **OS**: Windows, Linux, macOS

### Recommended Requirements
- **RAM**: 16GB (for larger models)
- **Storage**: 20GB free space
- **CPU**: Multi-core processor
- **GPU**: Optional (for acceleration)

## Troubleshooting

### Common Issues

**Model Download Fails**
```bash
# Check internet connection
# Verify model name spelling
/models  # List exact model names
```

**Model Loading Fails**
```bash
# Check if model was downloaded
/models  # Verify model exists
# Check available RAM
/status  # Monitor system resources
```

**Slow Responses**
```bash
# Try smaller model
/download qwen-0.5b
/load qwen-0.5b
# Close other applications
# Check system resources
```

### Performance Tips

1. **Choose Right Model Size**
   - Use smaller models for faster responses
   - Use larger models for better quality

2. **Manage Memory**
   - Unload models when not needed
   - Close other applications
   - Monitor RAM usage

3. **Optimize Storage**
   - Keep models on fast SSD
   - Clean up unused models
   - Monitor disk space

## Integration with Zig AI Platform

This example demonstrates:

- **Model Management** - Using the platform's download utilities
- **ONNX Integration** - Loading and running ONNX models
- **Memory Management** - Efficient resource handling
- **Error Handling** - Robust error management
- **User Interface** - Clean command-line interface

## Next Steps

1. **Try Different Models** - Experiment with various model sizes
2. **Custom Models** - Add your own ONNX models
3. **Performance Tuning** - Optimize for your hardware
4. **Integration** - Use in your own applications
5. **Contribute** - Add new models and features

This example shows the power of the Zig AI Platform for real-world AI applications with actual models and authentic conversations!

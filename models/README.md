# Models Directory

This directory is for storing AI model files that will be used by the Zig AI Inference Engine.

## Model Files

Model files are not included in the repository due to their large size. They should be downloaded separately or via the CLI tool.

### Supported Model Formats
- ONNX (.onnx)
- SafeTensors (.safetensors)
- PyTorch (.bin, .pt, .pth)
- TensorFlow (.h5, .pb)

### Download Models

Use the Zig AI CLI to download models:

```bash
zig build cli -- download --model gpt2
```

Or manually place your model files in this directory following the expected structure:

```
models/
├── model_name/
│   ├── model.onnx
│   ├── vocab.json
│   ├── tokenizer.json
│   └── config.json
```

## Note

Large model files are automatically ignored by Git (see .gitignore). This keeps the repository lightweight while allowing local model storage.

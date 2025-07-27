#!/usr/bin/env python3
"""
Download and convert GPT-2 model to ONNX format for zig-ai-platform
"""

import os
import sys
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import onnx
import onnxruntime as ort

def download_and_convert_gpt2(model_name="gpt2", output_dir="models"):
    """Download GPT-2 and convert to ONNX format"""
    
    print(f"🚀 Downloading {model_name} model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download model and tokenizer
    print("📥 Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded: {model_name}")
    print(f"   - Vocabulary size: {model.config.vocab_size}")
    print(f"   - Hidden size: {model.config.n_embd}")
    print(f"   - Number of layers: {model.config.n_layer}")
    print(f"   - Number of heads: {model.config.n_head}")
    
    # Create dummy input for ONNX export
    sequence_length = 10
    batch_size = 1
    dummy_input = torch.randint(0, model.config.vocab_size, (batch_size, sequence_length))
    
    print("🔄 Converting to ONNX format...")
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
    
    print(f"✅ ONNX model saved to: {onnx_path}")
    
    # Verify ONNX model
    print("🔍 Verifying ONNX model...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
        
        # Test with ONNX Runtime
        print("🧪 Testing ONNX model with sample input...")
        session = ort.InferenceSession(onnx_path)
        
        # Test input
        test_input = np.random.randint(0, 1000, (1, 5), dtype=np.int64)
        outputs = session.run(["logits"], {"input_ids": test_input})
        
        logits_shape = outputs[0].shape
        print(f"✅ ONNX inference test passed: output shape {logits_shape}")
        
        # Verify vocabulary size
        if logits_shape[-1] == model.config.vocab_size:
            print(f"✅ Vocabulary size matches: {logits_shape[-1]}")
        else:
            print(f"⚠️  Vocabulary size mismatch: expected {model.config.vocab_size}, got {logits_shape[-1]}")
            
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return False
    
    # Save tokenizer for reference
    tokenizer_path = os.path.join(output_dir, f"{model_name}_tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"💾 Tokenizer saved to: {tokenizer_path}")
    
    # Create model info file
    info_path = os.path.join(output_dir, f"{model_name}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Vocabulary size: {model.config.vocab_size}\n")
        f.write(f"Hidden size: {model.config.n_embd}\n")
        f.write(f"Number of layers: {model.config.n_layer}\n")
        f.write(f"Number of heads: {model.config.n_head}\n")
        f.write(f"Max position embeddings: {model.config.n_positions}\n")
        f.write(f"ONNX file: {model_name}.onnx\n")
        f.write(f"Tokenizer: {model_name}_tokenizer/\n")
    
    print(f"📋 Model info saved to: {info_path}")
    
    # Test tokenization
    print("\n🧪 Testing tokenization...")
    test_text = "Hello, how are you today?"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"   Text: '{test_text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Decoded: '{decoded}'")
    
    print(f"\n🎉 Successfully downloaded and converted {model_name}!")
    print(f"📁 Files created:")
    print(f"   - {onnx_path}")
    print(f"   - {tokenizer_path}/")
    print(f"   - {info_path}")
    
    return True

def main():
    """Main function"""
    print("🤖 GPT-2 ONNX Converter for zig-ai-platform")
    print("=" * 50)
    
    # Available models
    models = {
        "1": ("gpt2", "GPT-2 Small (124M parameters, ~500MB)"),
        "2": ("gpt2-medium", "GPT-2 Medium (355M parameters, ~1.5GB)"),
        "3": ("gpt2-large", "GPT-2 Large (774M parameters, ~3GB)"),
        "4": ("distilgpt2", "DistilGPT-2 (82M parameters, ~350MB)")
    }
    
    print("Available models:")
    for key, (model_name, description) in models.items():
        print(f"  {key}. {description}")
    
    # Get user choice
    choice = input("\nSelect model (1-4) [default: 1]: ").strip()
    if not choice:
        choice = "1"
    
    if choice not in models:
        print("❌ Invalid choice. Using GPT-2 Small.")
        choice = "1"
    
    model_name, description = models[choice]
    print(f"\n📦 Selected: {description}")
    
    # Convert model
    try:
        success = download_and_convert_gpt2(model_name)
        if success:
            print(f"\n🚀 Ready to test with zig-ai-platform:")
            print(f"   ./zig-ai.exe chat --model models/{model_name}.onnx")
        else:
            print("\n❌ Conversion failed. Please check the error messages above.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

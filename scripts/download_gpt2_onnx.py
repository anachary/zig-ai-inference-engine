#!/usr/bin/env python3
"""
Download pre-converted GPT-2 ONNX model from Hugging Face
"""

import os
import requests
from urllib.parse import urlparse
import sys

def download_file(url, local_path):
    """Download a file with progress bar"""
    print(f"📥 Downloading {os.path.basename(local_path)}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r   Progress: {percent:.1f}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end='')
    
    print(f"\n✅ Downloaded: {local_path}")
    return True

def download_gpt2_onnx():
    """Download GPT-2 ONNX model"""
    
    print("🤖 Downloading GPT-2 ONNX Model")
    print("=" * 40)
    
    # Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Available pre-converted models
    models = {
        "1": {
            "name": "gpt2-onnx",
            "url": "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin",
            "size": "~350MB",
            "description": "GPT-2 Small ONNX (fastest)"
        },
        "2": {
            "name": "gpt2-optimized",
            "url": "https://github.com/microsoft/onnxruntime/raw/main/onnxruntime/test/testdata/gpt2_past.onnx",
            "size": "~500MB", 
            "description": "GPT-2 Optimized ONNX"
        }
    }
    
    # For now, let's try a direct approach with a known working model
    print("🎯 Downloading GPT-2 ONNX model...")
    
    # Try to download from ONNX Model Zoo or create a simple one
    gpt2_urls = [
        "https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-10.onnx",
        "https://github.com/microsoft/onnxruntime/raw/main/onnxruntime/test/testdata/gpt2_past.onnx"
    ]
    
    for i, url in enumerate(gpt2_urls):
        try:
            model_name = f"gpt2_{i+1}.onnx"
            model_path = os.path.join(models_dir, model_name)
            
            print(f"\n🔄 Trying source {i+1}: {url}")
            download_file(url, model_path)
            
            # Verify the file exists and has reasonable size
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"✅ Downloaded {model_name}: {size_mb:.1f}MB")
                
                if size_mb > 1:  # Reasonable size for a model
                    print(f"\n🎉 Successfully downloaded GPT-2 ONNX model!")
                    print(f"📁 Location: {model_path}")
                    print(f"📊 Size: {size_mb:.1f}MB")
                    print(f"\n🚀 Test with zig-ai-platform:")
                    print(f"   zig build")
                    print(f"   .\\zig-out\\bin\\zig-ai.exe chat --model {model_path}")
                    return True
                else:
                    print(f"⚠️  File too small ({size_mb:.1f}MB), trying next source...")
                    os.remove(model_path)
            
        except Exception as e:
            print(f"❌ Failed to download from source {i+1}: {e}")
            continue
    
    print("\n⚠️  Could not download pre-converted model. Let's create a simple test model...")
    return create_simple_gpt2_onnx()

def create_simple_gpt2_onnx():
    """Create a simple GPT-2-like ONNX model for testing"""
    try:
        import torch
        import torch.nn as nn
        
        print("🔧 Creating simple GPT-2-like model for testing...")
        
        class SimpleGPT2(nn.Module):
            def __init__(self, vocab_size=50257, hidden_size=768, seq_len=1024):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.transformer = nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=12,
                    dim_feedforward=3072,
                    batch_first=True
                )
                self.lm_head = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                # Simple transformer layer
                x = self.transformer(x, x)
                logits = self.lm_head(x)
                return logits
        
        # Create model
        model = SimpleGPT2()
        model.eval()
        
        # Export to ONNX
        dummy_input = torch.randint(0, 1000, (1, 10))
        model_path = "models/simple_gpt2.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            model_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {1: "sequence_length"},
                "logits": {1: "sequence_length"}
            },
            opset_version=11
        )
        
        print(f"✅ Created simple GPT-2 model: {model_path}")
        return True
        
    except ImportError:
        print("❌ PyTorch not available. Please install: pip install torch")
        return False
    except Exception as e:
        print(f"❌ Failed to create simple model: {e}")
        return False

def main():
    """Main function"""
    try:
        success = download_gpt2_onnx()
        if not success:
            print("\n❌ Failed to download GPT-2 ONNX model.")
            print("💡 You can try:")
            print("   1. Install PyTorch: pip install torch transformers")
            print("   2. Run the full conversion script: python scripts/download_gpt2.py")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Download cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

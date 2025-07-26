#!/usr/bin/env python3
"""
Download a simple ONNX model for testing real inference
"""

import os
import urllib.request
import sys

def download_file(url, filename):
    """Download a file with progress indication"""
    print(f"Downloading {filename}...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end="")
        else:
            print(f"\rDownloaded: {downloaded} bytes", end="")
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False

def main():
    """Download test models"""
    print("🚀 Downloading ONNX test models for real inference testing")
    
    # Create models directory
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"📁 Created {models_dir} directory")
    
    # Simple models for testing
    models = [
        {
            "name": "mnist_classifier.onnx",
            "url": "https://github.com/onnx/models/raw/main/Computer_Vision/mnist_cnn_onnx/model/mnist.onnx",
            "description": "Simple MNIST digit classifier (26KB)"
        },
        {
            "name": "simple_add.onnx",
            "url": "https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/test/testdata/add.onnx",
            "description": "Simple addition operator test (1KB)"
        }
    ]
    
    success_count = 0
    
    for model in models:
        print(f"\n📦 {model['description']}")
        filepath = os.path.join(models_dir, model["name"])
        
        # Skip if already exists
        if os.path.exists(filepath):
            print(f"✅ {model['name']} already exists, skipping")
            success_count += 1
            continue
            
        if download_file(model["url"], filepath):
            success_count += 1
            
            # Verify file size
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"📊 File size: {size_mb:.1f} MB")
    
    print(f"\n🎉 Downloaded {success_count}/{len(models)} models successfully")
    
    if success_count > 0:
        print("\n🔧 Next steps:")
        print("1. Run: zig build test-inference")
        print("2. Test with: zig build cli -- pipeline --model models/mnist_classifier.onnx --prompt 'test'")
        print("3. Check real inference: zig run test_real_model.zig")
    
    return success_count == len(models)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/bin/bash
# Download TinyLLM Models for IoT Demo
# This script downloads lightweight LLM models suitable for Raspberry Pi

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_success() { echo -e "${GREEN}$1${NC}"; }
print_error() { echo -e "${RED}$1${NC}"; }
print_warning() { echo -e "${YELLOW}$1${NC}"; }
print_info() { echo -e "${CYAN}$1${NC}"; }

print_header() {
    echo ""
    echo -e "${CYAN}$(printf '=%.0s' {1..60})${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..60})${NC}"
    echo ""
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
CACHE_DIR="$HOME/.cache/tinyllm"

# Model configurations
declare -A MODELS=(
    ["tinyllama-1.1b"]="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ["phi-2-2.7b"]="https://huggingface.co/microsoft/phi-2"
    ["stablelm-zephyr-3b"]="https://huggingface.co/stabilityai/stablelm-zephyr-3b"
    ["qwen1.5-0.5b"]="https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat"
)

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    local missing_deps=()
    
    # Check Python and pip
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("python3-pip")
    fi
    
    # Check git (for huggingface downloads)
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    # Check curl/wget
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_info "Please install them first:"
        print_info "  sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
    
    print_success "✓ All dependencies available"
}

# Install Python packages for LLM inference
install_llm_packages() {
    print_header "Installing LLM Packages"
    
    print_info "Installing transformers and related packages..."
    
    # Core packages for LLM inference
    pip3 install --user \
        transformers>=4.35.0 \
        torch>=2.0.0 \
        tokenizers>=0.14.0 \
        accelerate>=0.24.0 \
        sentencepiece>=0.1.99 \
        protobuf>=3.20.0
    
    # Optional: Install optimized packages for ARM/Pi
    if [[ $(uname -m) == "aarch64" || $(uname -m) == "armv7l" ]]; then
        print_info "Installing ARM-optimized packages..."
        pip3 install --user \
            onnxruntime>=1.16.0 \
            optimum>=1.14.0
    fi
    
    print_success "✓ LLM packages installed"
}

# Download model from Hugging Face
download_model() {
    local model_name=$1
    local model_url=$2
    local model_dir="$MODELS_DIR/$model_name"
    
    print_info "Downloading $model_name..."
    
    # Create model directory
    mkdir -p "$model_dir"
    
    # Check if model already exists
    if [ -f "$model_dir/config.json" ]; then
        print_warning "Model $model_name already exists, skipping download"
        return 0
    fi
    
    # Download using git (Hugging Face standard)
    if command -v git-lfs &> /dev/null; then
        print_info "Using git-lfs for large file support..."
        cd "$MODELS_DIR"
        git clone "$model_url" "$model_name"
    else
        # Fallback: download essential files only
        print_warning "git-lfs not available, downloading essential files only..."
        
        local files=(
            "config.json"
            "tokenizer.json"
            "tokenizer_config.json"
            "special_tokens_map.json"
            "vocab.txt"
        )
        
        for file in "${files[@]}"; do
            local file_url="$model_url/resolve/main/$file"
            if curl -f -L "$file_url" -o "$model_dir/$file" 2>/dev/null; then
                print_success "✓ Downloaded $file"
            else
                print_warning "⚠ Could not download $file (may not exist)"
            fi
        done
        
        # Try to download a small model file
        local model_files=("pytorch_model.bin" "model.safetensors")
        for model_file in "${model_files[@]}"; do
            local file_url="$model_url/resolve/main/$model_file"
            print_info "Attempting to download $model_file..."
            if curl -f -L "$file_url" -o "$model_dir/$model_file" 2>/dev/null; then
                print_success "✓ Downloaded $model_file"
                break
            fi
        done
    fi
    
    # Verify download
    if [ -f "$model_dir/config.json" ]; then
        print_success "✓ Model $model_name downloaded successfully"
        
        # Get model size
        local size=$(du -sh "$model_dir" | cut -f1)
        print_info "  Model size: $size"
    else
        print_error "❌ Failed to download $model_name"
        return 1
    fi
}

# Create model configuration
create_model_config() {
    print_header "Creating Model Configuration"
    
    cat > "$MODELS_DIR/model_config.yaml" << 'EOF'
# TinyLLM Model Configuration for IoT Demo

models:
  tinyllama-1.1b:
    name: "TinyLlama 1.1B Chat"
    parameters: 1100000000
    memory_requirement_mb: 2200
    use_cases: ["smart-home", "general-chat"]
    inference_time_ms: 2000
    quantization: "fp16"
    context_length: 2048
    
  qwen1.5-0.5b:
    name: "Qwen 1.5 0.5B Chat"
    parameters: 500000000
    memory_requirement_mb: 1000
    use_cases: ["smart-home", "lightweight-chat"]
    inference_time_ms: 1500
    quantization: "fp16"
    context_length: 2048
    
  phi-2-2.7b:
    name: "Microsoft Phi-2 2.7B"
    parameters: 2700000000
    memory_requirement_mb: 5400
    use_cases: ["industrial-iot", "complex-reasoning"]
    inference_time_ms: 3500
    quantization: "fp16"
    context_length: 2048
    
  stablelm-zephyr-3b:
    name: "StableLM Zephyr 3B"
    parameters: 3000000000
    memory_requirement_mb: 6000
    use_cases: ["retail-edge", "customer-service"]
    inference_time_ms: 4000
    quantization: "fp16"
    context_length: 4096

optimization:
  enable_quantization: true
  enable_caching: true
  max_cache_size_mb: 512
  enable_batching: false  # Disabled for edge devices
  max_sequence_length: 512  # Reduced for Pi
EOF
    
    print_success "✓ Created model configuration"
}

# Create LLM inference wrapper
create_llm_wrapper() {
    print_header "Creating LLM Inference Wrapper"
    
    mkdir -p "$SCRIPT_DIR/src/llm"
    
    cat > "$SCRIPT_DIR/src/llm/tinyllm_inference.py" << 'EOF'
#!/usr/bin/env python3
"""
TinyLLM Inference Wrapper for IoT Demo
Provides real LLM inference using lightweight models
"""

import os
import time
import torch
import logging
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TextStreamer,
    BitsAndBytesConfig
)
import yaml

logger = logging.getLogger(__name__)

class TinyLLMInference:
    """Lightweight LLM inference for IoT devices"""
    
    def __init__(self, model_name: str = "qwen1.5-0.5b", device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.config = self._load_config()
        
        # Performance settings for Pi
        self.max_length = 512
        self.temperature = 0.7
        self.do_sample = True
        
    def _get_device(self, device: str) -> str:
        """Determine optimal device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        config_path = os.path.join(os.path.dirname(__file__), "../../models/model_config.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Model config not found, using defaults")
            return {"models": {}}
    
    def load_model(self) -> bool:
        """Load the specified model"""
        model_path = os.path.join(os.path.dirname(__file__), f"../../models/{self.model_name}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model path not found: {model_path}")
            return False
        
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}")
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100) -> Dict[str, Any]:
        """Generate response using the loaded model"""
        if self.model is None or self.tokenizer is None:
            return {
                "response": "Model not loaded. Please load a model first.",
                "error": "Model not available",
                "processing_time_ms": 0
            }
        
        start_time = time.time()
        
        try:
            # Format prompt for chat models
            if "chat" in self.model_name.lower():
                formatted_prompt = f"<|user|>\n{prompt}<|assistant|>\n"
            else:
                formatted_prompt = prompt
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - max_new_tokens
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "response": response,
                "processing_time_ms": round(processing_time, 2),
                "model_used": self.model_name,
                "device": self.device,
                "tokens_generated": len(outputs[0]) - inputs['input_ids'].shape[1]
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Generation failed: {e}")
            return {
                "response": f"Generation failed: {str(e)}",
                "error": str(e),
                "processing_time_ms": round(processing_time, 2)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        model_config = self.config.get("models", {}).get(self.model_name, {})
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": model_config.get("parameters", "unknown"),
            "memory_requirement_mb": model_config.get("memory_requirement_mb", "unknown"),
            "context_length": model_config.get("context_length", "unknown"),
            "status": "loaded"
        }

# Example usage
if __name__ == "__main__":
    # Test the inference
    llm = TinyLLMInference("qwen1.5-0.5b")
    
    if llm.load_model():
        print("Model loaded successfully!")
        
        test_prompts = [
            "What is the weather like?",
            "Turn on the lights",
            "Check system status"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            result = llm.generate_response(prompt)
            print(f"Response: {result['response']}")
            print(f"Time: {result['processing_time_ms']:.1f}ms")
    else:
        print("Failed to load model")
EOF
    
    print_success "✓ Created LLM inference wrapper"
}

# Show available models
show_models() {
    print_header "Available TinyLLM Models"
    
    print_info "Recommended models for Raspberry Pi:"
    echo ""
    
    for model in "${!MODELS[@]}"; do
        case $model in
            "qwen1.5-0.5b")
                print_success "✓ $model - Ultra lightweight (500M params, ~1GB RAM)"
                print_info "  Best for: Smart home, basic chat"
                ;;
            "tinyllama-1.1b")
                print_success "✓ $model - Lightweight (1.1B params, ~2GB RAM)"
                print_info "  Best for: General purpose, good quality"
                ;;
            "phi-2-2.7b")
                print_warning "⚠ $model - Medium (2.7B params, ~5GB RAM)"
                print_info "  Best for: Industrial IoT, complex reasoning"
                ;;
            "stablelm-zephyr-3b")
                print_warning "⚠ $model - Large (3B params, ~6GB RAM)"
                print_info "  Best for: Retail, customer service"
                ;;
        esac
        echo ""
    done
    
    print_info "Recommendation for Raspberry Pi 4 (4GB): qwen1.5-0.5b or tinyllama-1.1b"
    print_info "Recommendation for Raspberry Pi 4 (8GB): phi-2-2.7b"
}

# Main download function
download_selected_models() {
    local models_to_download=("$@")
    
    if [ ${#models_to_download[@]} -eq 0 ]; then
        # Default: download lightweight models
        models_to_download=("qwen1.5-0.5b" "tinyllama-1.1b")
        print_info "No models specified, downloading default lightweight models..."
    fi
    
    print_header "Downloading Selected Models"
    
    for model in "${models_to_download[@]}"; do
        if [[ -n "${MODELS[$model]}" ]]; then
            download_model "$model" "${MODELS[$model]}"
        else
            print_error "Unknown model: $model"
            print_info "Available models: ${!MODELS[*]}"
        fi
    done
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [MODELS...]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help"
    echo "  -l, --list     List available models"
    echo "  -a, --all      Download all models"
    echo "  --install-deps Install Python dependencies only"
    echo ""
    echo "Models:"
    echo "  qwen1.5-0.5b      Ultra lightweight (500M params)"
    echo "  tinyllama-1.1b    Lightweight (1.1B params)"
    echo "  phi-2-2.7b        Medium (2.7B params)"
    echo "  stablelm-zephyr-3b Large (3B params)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Download default models"
    echo "  $0 qwen1.5-0.5b            # Download specific model"
    echo "  $0 -a                       # Download all models"
    echo "  $0 --list                   # Show available models"
}

# Main execution
main() {
    case "${1:-}" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--list)
            show_models
            exit 0
            ;;
        --install-deps)
            check_dependencies
            install_llm_packages
            exit 0
            ;;
        -a|--all)
            check_dependencies
            install_llm_packages
            create_model_config
            create_llm_wrapper
            download_selected_models "${!MODELS[@]}"
            ;;
        *)
            check_dependencies
            install_llm_packages
            create_model_config
            create_llm_wrapper
            download_selected_models "$@"
            ;;
    esac
    
    print_header "Download Complete!"
    print_success "🎉 TinyLLM models ready for IoT demo!"
    echo ""
    print_info "Next steps:"
    print_info "1. Test the models: python3 src/llm/tinyllm_inference.py"
    print_info "2. Update Pi simulator to use real LLM"
    print_info "3. Start the demo: ./start-iot-demo.sh"
    echo ""
    print_info "Models location: $MODELS_DIR"
}

# Create directories
mkdir -p "$MODELS_DIR"
mkdir -p "$CACHE_DIR"

# Run main function
main "$@"

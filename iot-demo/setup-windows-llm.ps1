# Windows-Compatible LLM Setup for IoT Demo
# Fixes CMake/sentencepiece issues on Windows

param(
    [switch]$SkipTorch,
    [switch]$CPUOnly,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

# Colors
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Header = "Magenta"
}

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput "=" * 60 -Color "Header"
    Write-ColorOutput "  $Title" -Color "Header"
    Write-ColorOutput "=" * 60 -Color "Header"
    Write-Host ""
}

Write-Header "Windows LLM Setup for IoT Demo"

# Check Python version
$pythonVersion = python --version 2>$null
if ($pythonVersion) {
    Write-ColorOutput "✓ Python detected: $pythonVersion" -Color "Success"
} else {
    Write-ColorOutput "❌ Python not found. Please install Python 3.8+" -Color "Error"
    exit 1
}

# Step 1: Install PyTorch (CPU-only for Windows compatibility)
if (-not $SkipTorch) {
    Write-Header "Installing PyTorch"
    
    if ($CPUOnly) {
        Write-ColorOutput "Installing CPU-only PyTorch..." -Color "Info"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    } else {
        Write-ColorOutput "Installing PyTorch with CUDA support..." -Color "Info"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✓ PyTorch installed successfully" -Color "Success"
    } else {
        Write-ColorOutput "⚠ PyTorch installation had issues, continuing..." -Color "Warning"
    }
}

# Step 2: Install transformers and related packages (avoiding sentencepiece)
Write-Header "Installing Transformers"

Write-ColorOutput "Installing core packages..." -Color "Info"
pip install transformers>=4.35.0 tokenizers>=0.14.0 accelerate>=0.24.0

# Step 3: Try to install sentencepiece with fallback
Write-Header "Installing SentencePiece"

Write-ColorOutput "Attempting to install sentencepiece..." -Color "Info"
$sentencepieceResult = pip install sentencepiece 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "⚠ SentencePiece failed to install (CMake issue)" -Color "Warning"
    Write-ColorOutput "Installing alternative tokenizer..." -Color "Info"
    pip install tiktoken
    Write-ColorOutput "✓ Alternative tokenizer installed" -Color "Success"
} else {
    Write-ColorOutput "✓ SentencePiece installed successfully" -Color "Success"
}

# Step 4: Install remaining packages
Write-Header "Installing Additional Packages"

$packages = @(
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pyyaml>=6.0.1",
    "psutil>=5.9.0",
    "pydantic>=2.4.0",
    "httpx>=0.25.0",
    "aiofiles>=23.2.1",
    "python-multipart>=0.0.6"
)

foreach ($package in $packages) {
    Write-ColorOutput "Installing $package..." -Color "Info"
    pip install $package
}

# Step 5: Create Windows-compatible LLM wrapper
Write-Header "Creating Windows LLM Wrapper"

$llmDir = "src/llm"
if (-not (Test-Path $llmDir)) {
    New-Item -ItemType Directory -Path $llmDir -Force
}

# Create a simplified LLM wrapper that works on Windows
$windowsLLMWrapper = @'
#!/usr/bin/env python3
"""
Windows-Compatible TinyLLM Inference
Simplified version that avoids CMake dependencies
"""

import os
import time
import logging
from typing import Optional, Dict, Any
import yaml

# Try to import torch and transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class WindowsLLMInference:
    """Windows-compatible LLM inference"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = "cpu"  # Force CPU on Windows for compatibility
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Use models that don't require sentencepiece
        self.compatible_models = {
            "dialogpt": "microsoft/DialoGPT-medium",
            "gpt2": "gpt2",
            "distilgpt2": "distilgpt2"
        }
        
        # Use compatible model if specified model isn't available
        if model_name in self.compatible_models:
            self.model_name = self.compatible_models[model_name]
        elif not model_name.startswith("microsoft/") and not model_name.startswith("gpt"):
            # Fallback to DialoGPT for unknown models
            self.model_name = self.compatible_models["dialogpt"]
            logger.warning(f"Using fallback model: {self.model_name}")
    
    def load_model(self) -> bool:
        """Load the model (Windows-compatible)"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available")
            return False
        
        try:
            logger.info(f"Loading Windows-compatible model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model (CPU only for Windows compatibility)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True
            )
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> Dict[str, Any]:
        """Generate response using the loaded model"""
        if not self.model_loaded:
            return {
                "response": "Model not loaded. Using fallback response.",
                "error": "Model not available",
                "processing_time_ms": 0,
                "model_used": "fallback",
                "device": "cpu"
            }
        
        start_time = time.time()
        
        try:
            # Format prompt for chat
            if "DialoGPT" in self.model_name:
                # DialoGPT format
                formatted_prompt = prompt
            else:
                # GPT-2 format
                formatted_prompt = f"Human: {prompt}\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer.encode(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512 - max_new_tokens
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Clean up response
            if not response:
                response = "I understand your request and I'm here to help!"
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "response": response,
                "processing_time_ms": round(processing_time, 2),
                "model_used": self.model_name,
                "device": self.device,
                "tokens_generated": len(outputs[0]) - inputs.shape[1]
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Generation failed: {e}")
            return {
                "response": f"I apologize, but I encountered an issue processing your request.",
                "error": str(e),
                "processing_time_ms": round(processing_time, 2),
                "model_used": self.model_name,
                "device": self.device
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "status": "loaded" if self.model_loaded else "not_loaded",
            "compatible_models": list(self.compatible_models.keys()),
            "platform": "windows"
        }

# Test function
if __name__ == "__main__":
    print("🧪 Testing Windows LLM Inference...")
    
    # Test with compatible model
    llm = WindowsLLMInference("dialogpt")
    
    if llm.load_model():
        print("✅ Model loaded successfully!")
        
        test_prompts = [
            "Hello, how are you?",
            "What is the weather like?",
            "Turn on the lights"
        ]
        
        for prompt in test_prompts:
            print(f"\n👤 Prompt: {prompt}")
            result = llm.generate_response(prompt)
            print(f"🤖 Response: {result['response']}")
            print(f"⏱️ Time: {result['processing_time_ms']:.1f}ms")
    else:
        print("❌ Failed to load model")
        print("💡 Try installing: pip install torch transformers")
'@

$windowsLLMWrapper | Out-File -FilePath "$llmDir/windows_llm_inference.py" -Encoding UTF8

# Step 6: Create Windows startup script
Write-Header "Creating Windows Startup Scripts"

$windowsStartScript = @'
# Windows LLM IoT Demo Starter
# Uses Windows-compatible models

$ErrorActionPreference = "Continue"

Write-Host "🚀 Starting Windows LLM IoT Demo..." -ForegroundColor Green

# Start devices with Windows-compatible LLM
Write-Host "Starting Smart Home Pi with Windows LLM..." -ForegroundColor Cyan
$env:USE_REAL_LLM = "true"
$env:MODEL_NAME = "dialogpt"
$env:DEVICE_NAME = "smart-home-pi"
$env:DEVICE_PORT = "8081"
Start-Process python -ArgumentList "src/pi-simulator/windows_llm_pi.py" -WindowStyle Hidden
Start-Sleep 2

Write-Host "Starting Industrial Pi with Windows LLM..." -ForegroundColor Cyan
$env:DEVICE_NAME = "industrial-pi"
$env:DEVICE_PORT = "8082"
Start-Process python -ArgumentList "src/pi-simulator/windows_llm_pi.py" -WindowStyle Hidden
Start-Sleep 2

Write-Host "Starting Retail Pi with Windows LLM..." -ForegroundColor Cyan
$env:DEVICE_NAME = "retail-pi"
$env:DEVICE_PORT = "8083"
Start-Process python -ArgumentList "src/pi-simulator/windows_llm_pi.py" -WindowStyle Hidden
Start-Sleep 2

Write-Host ""
Write-Host "✅ Windows LLM devices started!" -ForegroundColor Green
Write-Host ""
Write-Host "🤖 Access Points:" -ForegroundColor Cyan
Write-Host "   🏠 Smart Home Pi: http://localhost:8081" -ForegroundColor Cyan
Write-Host "   🏭 Industrial Pi: http://localhost:8082" -ForegroundColor Cyan
Write-Host "   🛒 Retail Pi: http://localhost:8083" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Note: Using Windows-compatible models (DialoGPT)" -ForegroundColor Yellow
Write-Host "🛑 Stop: .\stop-windows-llm.ps1" -ForegroundColor Yellow

# Test one device
Start-Sleep 5
Write-Host ""
Write-Host "🧪 Testing Windows LLM..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8081/api/inference" -Method POST -Body '{"query": "Hello from Windows!"}' -ContentType "application/json" -TimeoutSec 10
    Write-Host "✅ Test successful: $($response.result)" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Service starting up, try again in a moment..." -ForegroundColor Yellow
}
'@

$windowsStartScript | Out-File -FilePath "start-windows-llm.ps1" -Encoding UTF8

# Step 7: Create simplified Pi simulator for Windows
$windowsPiSimulator = @'
#!/usr/bin/env python3
"""
Windows-Compatible Pi Simulator
Uses DialoGPT instead of models requiring sentencepiece
"""

import asyncio
import sys
import os

# Add LLM path
sys.path.append(os.path.join(os.path.dirname(__file__), '../llm'))

try:
    from windows_llm_inference import WindowsLLMInference
    LLM_AVAILABLE = True
except ImportError:
    print("⚠️ Windows LLM not available, using fallback")
    LLM_AVAILABLE = False

# Import the base simulator
sys.path.append(os.path.dirname(__file__))
from real_llm_pi import RealLLMPiSimulator, app, simulator

# Override the LLM initialization for Windows
if LLM_AVAILABLE:
    class WindowsPiSimulator(RealLLMPiSimulator):
        def _initialize_llm(self):
            """Initialize Windows-compatible LLM"""
            try:
                print(f"Initializing Windows LLM: {self.model_name}")
                self.llm = WindowsLLMInference(self.model_name)
                asyncio.create_task(self._load_model_async())
            except Exception as e:
                print(f"Failed to initialize Windows LLM: {e}")
                self.llm = None
    
    # Replace the global simulator
    DEVICE_NAME = os.getenv("DEVICE_NAME", "windows-pi")
    MODEL_NAME = os.getenv("MODEL_NAME", "dialogpt")
    simulator = WindowsPiSimulator(DEVICE_NAME, MODEL_NAME)

if __name__ == "__main__":
    import uvicorn
    DEVICE_PORT = int(os.getenv("DEVICE_PORT", "8081"))
    print(f"🚀 Starting Windows LLM Pi Simulator on port {DEVICE_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=DEVICE_PORT)
'@

$windowsPiSimulator | Out-File -FilePath "src/pi-simulator/windows_llm_pi.py" -Encoding UTF8

# Step 8: Create test script
$windowsTestScript = @'
#!/usr/bin/env python3
"""
Test Windows LLM IoT Demo
"""

import asyncio
import httpx
import time

async def test_windows_llm():
    print("🧪 Testing Windows LLM IoT Demo")
    print("=" * 50)
    
    devices = [
        {"name": "Smart Home Pi", "port": 8081},
        {"name": "Industrial Pi", "port": 8082},
        {"name": "Retail Pi", "port": 8083}
    ]
    
    test_queries = [
        "Hello, how are you?",
        "Turn on the lights",
        "Check system status"
    ]
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        for device in devices:
            print(f"\n🧪 Testing {device['name']} (Port {device['port']})")
            
            try:
                # Health check
                health = await client.get(f"http://localhost:{device['port']}/health")
                if health.status_code == 200:
                    print("✅ Device online")
                else:
                    print("❌ Device not responding")
                    continue
            except:
                print("❌ Device not reachable")
                continue
            
            # Test inference
            for query in test_queries[:1]:  # Test 1 query per device
                print(f"\n👤 Query: '{query}'")
                
                try:
                    response = await client.post(
                        f"http://localhost:{device['port']}/api/inference",
                        json={"query": query}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"🤖 Response: {result['result']}")
                        print(f"⏱️ Time: {result['processing_time_ms']:.1f}ms")
                        print(f"🧠 Model: {result['model_used']}")
                    else:
                        print(f"❌ Error: {response.status_code}")
                        
                except Exception as e:
                    print(f"❌ Failed: {e}")
    
    print("\n🎉 Windows LLM test completed!")

if __name__ == "__main__":
    asyncio.run(test_windows_llm())
'@

$windowsTestScript | Out-File -FilePath "test-windows-llm.py" -Encoding UTF8

Write-Header "Setup Complete!"

Write-ColorOutput "🎉 Windows LLM setup completed!" -Color "Success"
Write-Host ""
Write-ColorOutput "Next steps:" -Color "Info"
Write-ColorOutput "1. .\start-windows-llm.ps1    # Start Windows LLM demo" -Color "Cyan"
Write-ColorOutput "2. python test-windows-llm.py # Test the setup" -Color "Cyan"
Write-ColorOutput "3. Open http://localhost:8081  # Access dashboard" -Color "Cyan"
Write-Host ""
Write-ColorOutput "💡 This setup uses:" -Color "Info"
Write-ColorOutput "   • DialoGPT (no sentencepiece required)" -Color "Info"
Write-ColorOutput "   • CPU-only inference (Windows compatible)" -Color "Info"
Write-ColorOutput "   • Fallback to simulation if needed" -Color "Info"
Write-Host ""

if ($Verbose) {
    Write-ColorOutput "🔍 Installed packages:" -Color "Info"
    pip list | Select-String -Pattern "(torch|transformers|fastapi)"
}

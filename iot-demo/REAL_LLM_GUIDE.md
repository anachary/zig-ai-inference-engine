# Real LLM Integration Guide for IoT Demo

## 🤖 Overview

This guide shows how to integrate real TinyLLM models into the IoT demo for actual AI inference instead of simulated responses.

## 🚀 Quick Start

### 1. Setup Real LLM Environment
```bash
cd iot-demo
chmod +x setup-real-llm.sh
./setup-real-llm.sh
```

### 2. Start Real LLM Demo
```bash
./start-real-llm-demo.sh
```

### 3. Test Real AI Inference
```bash
python3 test-real-llm.py
```

### 4. Access Interactive Dashboards
- 🏠 Smart Home Pi: http://localhost:8081
- 🏭 Industrial Pi: http://localhost:8082  
- 🛒 Retail Pi: http://localhost:8083

## 📋 Available Models

### Recommended for Raspberry Pi

| Model | Size | RAM Usage | Speed | Best For |
|-------|------|-----------|-------|----------|
| **Qwen 1.5 0.5B** | 500M params | ~1GB | Fast | Smart Home, Basic Chat |
| **TinyLlama 1.1B** | 1.1B params | ~2GB | Medium | General Purpose |
| **Phi-2 2.7B** | 2.7B params | ~5GB | Slower | Industrial, Complex Tasks |

### Download Specific Models
```bash
# Download lightweight model (recommended)
./download-tinyllm.sh qwen1.5-0.5b

# Download multiple models
./download-tinyllm.sh qwen1.5-0.5b tinyllama-1.1b

# Download all models
./download-tinyllm.sh --all

# List available models
./download-tinyllm.sh --list
```

## 🔧 Configuration

### Environment Variables
```bash
# Enable/disable real LLM
export USE_REAL_LLM=true

# Select model
export MODEL_NAME=qwen1.5-0.5b

# Device configuration
export DEVICE_NAME=smart-home-pi
export DEVICE_PORT=8081
```

### Model Configuration
Edit `models/model_config.yaml`:
```yaml
models:
  qwen1.5-0.5b:
    name: "Qwen 1.5 0.5B Chat"
    parameters: 500000000
    memory_requirement_mb: 1000
    use_cases: ["smart-home", "lightweight-chat"]
    inference_time_ms: 1500
    quantization: "fp16"
    context_length: 2048
```

## 🧪 Testing Real LLM

### Manual Testing
```bash
# Test specific device
curl -X POST http://localhost:8081/api/inference \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn on the living room lights", "max_tokens": 50}'

# Check model status
curl http://localhost:8081/api/model-info

# Get device metrics
curl http://localhost:8081/api/metrics
```

### Automated Testing
```bash
# Run comprehensive tests
python3 test-real-llm.py

# Test specific scenarios
python3 -c "
import asyncio
import httpx

async def test():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://localhost:8081/api/inference',
            json={'query': 'What is the weather like today?'}
        )
        result = response.json()
        print(f'Response: {result[\"result\"]}')
        print(f'Type: {result[\"inference_type\"]}')
        print(f'Time: {result[\"processing_time_ms\"]}ms')

asyncio.run(test())
"
```

## 📊 Performance Comparison

### Real LLM vs Simulated

| Metric | Simulated | Real LLM (Qwen 0.5B) | Real LLM (TinyLlama 1.1B) |
|--------|-----------|----------------------|---------------------------|
| Response Time | 1-2 seconds | 2-5 seconds | 3-8 seconds |
| Memory Usage | ~100MB | ~1GB | ~2GB |
| CPU Usage | 10-20% | 60-90% | 70-95% |
| Response Quality | Fixed templates | Dynamic AI | Higher quality AI |
| Offline Capability | ✅ | ✅ | ✅ |

### Resource Requirements

#### Minimum Requirements
- **RAM**: 4GB (for Qwen 0.5B)
- **Storage**: 2GB free space
- **CPU**: ARM Cortex-A72 or better

#### Recommended Requirements  
- **RAM**: 8GB (for multiple models)
- **Storage**: 5GB free space
- **CPU**: ARM Cortex-A76 or x86_64

## 🎯 Use Case Examples

### Smart Home Assistant
```bash
# Real LLM responses
curl -X POST http://localhost:8081/api/inference \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn on the bedroom lights and set them to 50% brightness"}'

# Expected: Contextual response about smart lighting control
```

### Industrial Monitoring
```bash
# Sensor analysis
curl -X POST http://localhost:8082/api/inference \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze this vibration data: frequency 15Hz, amplitude 2.3mm"}'

# Expected: Technical analysis of sensor readings
```

### Retail Customer Service
```bash
# Product assistance
curl -X POST http://localhost:8083/api/inference \
  -H "Content-Type: application/json" \
  -d '{"query": "I need help finding organic vegetables and checking current promotions"}'

# Expected: Helpful retail assistance response
```

## 🔍 Monitoring and Debugging

### Check Model Loading Status
```bash
# Check if model is loaded
curl http://localhost:8081/api/status | jq '.llm_status'

# Get detailed model info
curl http://localhost:8081/api/model-info | jq '.'
```

### Monitor Resource Usage
```bash
# Real-time metrics
watch -n 2 'curl -s http://localhost:8081/api/metrics | jq ".cpu_usage_percent, .memory_usage_percent, .temperature_celsius"'

# Check inference history
curl http://localhost:8081/api/history | jq '.history[-5:]'
```

### Debug Model Issues
```bash
# Check logs
python3 src/pi-simulator/real_llm_pi.py

# Test LLM directly
python3 src/llm/tinyllm_inference.py

# Verify model files
ls -la models/qwen1.5-0.5b/
```

## 🚨 Troubleshooting

### Model Won't Load
```bash
# Check model files exist
ls models/qwen1.5-0.5b/config.json

# Verify Python packages
pip3 list | grep -E "(torch|transformers)"

# Check available memory
free -h

# Try smaller model
export MODEL_NAME=qwen1.5-0.5b
```

### Slow Inference
```bash
# Check CPU usage
htop

# Monitor temperature
vcgencmd measure_temp  # On Raspberry Pi

# Reduce max_tokens
curl -X POST http://localhost:8081/api/inference \
  -d '{"query": "test", "max_tokens": 20}'
```

### Memory Issues
```bash
# Check memory usage
cat /proc/meminfo

# Enable swap (if needed)
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Use quantization
export QUANTIZATION=true
```

## 🔄 Fallback Behavior

The system automatically falls back to simulated responses if:
- Model fails to load
- Inference times out
- Memory issues occur
- LLM packages not installed

Check fallback status:
```bash
curl http://localhost:8081/api/status | jq '.llm_status'
# Possible values: "loaded", "loading", "failed", "not_available"
```

## 🎛 Advanced Configuration

### Custom Model Integration
```python
# Add custom model to download-tinyllm.sh
MODELS["custom-model"]="https://huggingface.co/your/custom-model"

# Update model_config.yaml
custom-model:
  name: "Custom Model"
  parameters: 1000000000
  memory_requirement_mb: 2000
  use_cases: ["custom-application"]
```

### Performance Tuning
```python
# Edit src/llm/tinyllm_inference.py
class TinyLLMInference:
    def __init__(self, model_name: str = "qwen1.5-0.5b"):
        # Adjust these for your hardware
        self.max_length = 256  # Reduce for faster inference
        self.temperature = 0.3  # Lower for more deterministic
        self.do_sample = False  # Disable for speed
```

### Multi-Model Setup
```bash
# Run different models on different devices
MODEL_NAME=qwen1.5-0.5b DEVICE_PORT=8081 python3 src/pi-simulator/real_llm_pi.py &
MODEL_NAME=tinyllama-1.1b DEVICE_PORT=8082 python3 src/pi-simulator/real_llm_pi.py &
MODEL_NAME=phi-2-2.7b DEVICE_PORT=8083 python3 src/pi-simulator/real_llm_pi.py &
```

## 📈 Production Deployment

### Optimization for Production
1. **Model Quantization**: Use INT8 or INT4 quantization
2. **Model Caching**: Pre-load models at startup
3. **Request Batching**: Batch multiple requests (if supported)
4. **Resource Monitoring**: Monitor CPU, memory, temperature
5. **Graceful Degradation**: Fallback to simulated responses

### Scaling Considerations
- **Horizontal Scaling**: Multiple Pi devices with load balancing
- **Model Distribution**: Different models for different use cases
- **Edge Caching**: Cache frequent responses
- **Cloud Fallback**: Route complex queries to cloud

## 🎉 Success Indicators

You'll know real LLM is working when:
- ✅ `inference_type: "real_llm"` in API responses
- ✅ Dynamic, contextual AI responses
- ✅ Model loading logs in console
- ✅ Higher memory usage (~1-2GB)
- ✅ Realistic inference times (2-8 seconds)

## 📞 Support

- **Model Issues**: Check Hugging Face model pages
- **Performance**: Monitor system resources
- **Integration**: Review API documentation at `/docs`
- **Debugging**: Enable verbose logging in Python scripts

Happy AI inferencing on the edge! 🤖🍓

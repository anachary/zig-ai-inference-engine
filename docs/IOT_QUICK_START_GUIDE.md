# 🚀 IoT Quick Start: Deploy AI Model in 15 Minutes

## 📋 Overview

Get your AI model running on an IoT device in just 15 minutes! This guide covers the fastest path to deploy a small, optimized model on a Raspberry Pi or similar edge device.

## ⚡ Prerequisites

- **Hardware**: Raspberry Pi 4 (4GB+ RAM) or similar ARM device
- **OS**: Raspberry Pi OS or Ubuntu 20.04+
- **Network**: Internet connection for initial setup
- **Storage**: 16GB+ SD card

## 🏃‍♂️ Quick Setup Steps

### Step 1: Prepare Your Device (3 minutes)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-pip git docker.io
sudo usermod -aG docker $USER

# Reboot to apply changes
sudo reboot
```

### Step 2: Download and Setup Zig AI (2 minutes)

```bash
# Clone repository
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform

# Install IoT dependencies
pip3 install -r requirements-iot.txt
```

### Step 3: Download Pre-optimized Model (3 minutes)

```bash
# Create model directory
mkdir -p models/iot

# Download a pre-optimized model (example: TinyBERT for text classification)
wget https://github.com/anachary/zig-ai-platform/releases/download/v1.0.0/tinybert-iot.onnx \
  -O models/iot/tinybert-iot.onnx

# Verify download
ls -lh models/iot/
```

### Step 4: Deploy with Docker (5 minutes)

```bash
# Build IoT container
docker build -f docker/Dockerfile.iot-arm64 -t zig-ai-iot:latest .

# Run the container
docker run -d \
  --name zig-ai-iot \
  --restart unless-stopped \
  -p 8080:8080 \
  -v $(pwd)/models/iot:/app/models:ro \
  -e ZIG_AI_MODE=iot \
  -e ZIG_AI_DEVICE_TYPE=raspberry-pi \
  zig-ai-iot:latest

# Check if running
docker ps
```

### Step 5: Test Your Deployment (2 minutes)

```bash
# Test inference endpoint
curl -X POST http://localhost:8080/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a great product!", "task": "sentiment"}'

# Expected response:
# {"result": "positive", "confidence": 0.95, "latency_ms": 23}

# Check device status
curl http://localhost:8080/health

# View logs
docker logs zig-ai-iot
```

## 🎯 What You've Deployed

### Model Specifications
- **Model**: TinyBERT optimized for IoT
- **Size**: ~14MB (vs 440MB original BERT)
- **Latency**: <50ms on Raspberry Pi 4
- **Memory**: <256MB RAM usage
- **Accuracy**: 92% of original BERT performance

### Capabilities
- **Text Classification**: Sentiment analysis, intent recognition
- **Offline Operation**: No internet required after deployment
- **Auto-restart**: Container restarts automatically on failure
- **Health Monitoring**: Built-in health check endpoint

## 📊 Performance Monitoring

### Check System Resources

```bash
# CPU and memory usage
docker stats zig-ai-iot

# Device temperature (Raspberry Pi)
vcgencmd measure_temp

# Disk usage
df -h
```

### Performance Benchmarking

```bash
# Run 100 inference requests
for i in {1..100}; do
  curl -s -X POST http://localhost:8080/v1/inference \
    -H "Content-Type: application/json" \
    -d '{"text": "Test message '$i'", "task": "sentiment"}' \
    | jq '.latency_ms'
done | awk '{sum+=$1; count++} END {print "Average latency:", sum/count, "ms"}'
```

## 🔧 Customization Options

### Use Your Own Model

```bash
# Stop current container
docker stop zig-ai-iot

# Replace model file
cp your-optimized-model.onnx models/iot/custom-model.onnx

# Update configuration
cat > config/iot-config.yaml << EOF
model:
  path: "/app/models/custom-model.onnx"
  task: "your-task-type"
  input_format: "text"  # or "image", "audio"
  
device:
  max_memory_mb: 512
  threads: 4
  optimization_level: 3
EOF

# Restart with new configuration
docker run -d \
  --name zig-ai-iot-custom \
  --restart unless-stopped \
  -p 8080:8080 \
  -v $(pwd)/models/iot:/app/models:ro \
  -v $(pwd)/config:/app/config:ro \
  zig-ai-iot:latest
```

### Enable GPU Acceleration (NVIDIA Jetson)

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run with GPU support
docker run -d \
  --name zig-ai-iot-gpu \
  --restart unless-stopped \
  --runtime=nvidia \
  -p 8080:8080 \
  -v $(pwd)/models/iot:/app/models:ro \
  -e ZIG_AI_ENABLE_GPU=true \
  zig-ai-iot:latest
```

## 🌐 Connect to Cloud (Optional)

### Enable Telemetry

```bash
# Create cloud configuration
cat > config/cloud-config.yaml << EOF
cloud:
  enabled: true
  endpoint: "https://your-iot-hub.azure.com"
  device_id: "rpi-$(hostname)"
  api_key: "your-api-key"
  
telemetry:
  interval_seconds: 60
  metrics:
    - inference_count
    - average_latency
    - cpu_usage
    - memory_usage
    - temperature
EOF

# Restart with cloud connectivity
docker stop zig-ai-iot
docker run -d \
  --name zig-ai-iot-cloud \
  --restart unless-stopped \
  -p 8080:8080 \
  -v $(pwd)/models/iot:/app/models:ro \
  -v $(pwd)/config:/app/config:ro \
  -e ZIG_AI_CLOUD_ENABLED=true \
  zig-ai-iot:latest
```

## 🔄 Model Updates

### Over-the-Air Updates

```bash
# Enable automatic model updates
cat > config/update-config.yaml << EOF
updates:
  enabled: true
  check_interval: "24h"
  endpoint: "https://api.zig-ai.com/models"
  auto_install: false  # Set to true for automatic updates
  backup_previous: true
EOF

# The system will check for updates daily and notify via logs
docker logs -f zig-ai-iot-cloud | grep "UPDATE"
```

## 🚨 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check Docker logs
docker logs zig-ai-iot

# Common fixes:
# 1. Insufficient memory
free -h

# 2. Model file missing
ls -la models/iot/

# 3. Port already in use
sudo netstat -tlnp | grep 8080
```

#### High Memory Usage
```bash
# Monitor memory usage
watch -n 1 'docker stats zig-ai-iot --no-stream'

# Reduce memory usage
docker stop zig-ai-iot
docker run -d \
  --name zig-ai-iot-optimized \
  --restart unless-stopped \
  -p 8080:8080 \
  -v $(pwd)/models/iot:/app/models:ro \
  -e ZIG_AI_MAX_MEMORY_MB=256 \
  -e ZIG_AI_OPTIMIZATION_LEVEL=3 \
  zig-ai-iot:latest
```

#### Slow Inference
```bash
# Check CPU usage
top

# Optimize for performance
docker stop zig-ai-iot
docker run -d \
  --name zig-ai-iot-performance \
  --restart unless-stopped \
  --cpus="3.0" \
  -p 8080:8080 \
  -v $(pwd)/models/iot:/app/models:ro \
  -e ZIG_AI_THREADS=4 \
  -e ZIG_AI_OPTIMIZATION=performance \
  zig-ai-iot:latest
```

## 📱 Example Applications

### Smart Doorbell

```python
# smart_doorbell.py
import requests
import cv2
import time

def detect_person_at_door():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Save frame temporarily
        cv2.imwrite('/tmp/doorbell_frame.jpg', frame)
        
        # Send to AI model
        with open('/tmp/doorbell_frame.jpg', 'rb') as f:
            response = requests.post(
                'http://localhost:8080/v1/inference',
                files={'image': f},
                data={'task': 'person_detection'}
            )
            
        result = response.json()
        
        if result.get('person_detected', False):
            print("🚪 Person detected at door!")
            # Trigger notification, recording, etc.
            
        time.sleep(2)  # Check every 2 seconds

# Run the doorbell
detect_person_at_door()
```

### Voice-Controlled Light Switch

```python
# voice_light_control.py
import speech_recognition as sr
import requests
import RPi.GPIO as GPIO

# Setup GPIO for light control
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)  # Light relay pin

def control_lights():
    r = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        r.adjust_for_ambient_noise(source)
    
    print("🎤 Voice control ready. Say 'lights on' or 'lights off'")
    
    while True:
        try:
            with mic as source:
                audio = r.listen(source, timeout=1)
                
            text = r.recognize_google(audio).lower()
            
            # Send to AI for intent recognition
            response = requests.post(
                'http://localhost:8080/v1/inference',
                json={'text': text, 'task': 'intent_classification'}
            )
            
            result = response.json()
            intent = result.get('intent', '')
            
            if 'lights_on' in intent:
                GPIO.output(18, GPIO.HIGH)
                print("💡 Lights turned ON")
            elif 'lights_off' in intent:
                GPIO.output(18, GPIO.LOW)
                print("💡 Lights turned OFF")
                
        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            print(f"Error: {e}")

# Run voice control
control_lights()
```

## 🎉 Next Steps

### Expand Your Deployment
1. **Add More Models**: Deploy multiple specialized models
2. **Scale to Multiple Devices**: Use Docker Swarm or K3s
3. **Implement Edge Analytics**: Process data locally before sending to cloud
4. **Add Security**: Enable HTTPS, authentication, and encryption

### Learn More
- [Complete IoT Deployment Guide](./IOT_EDGE_DEPLOYMENT_GUIDE.md)
- [Performance Optimization](./LLM_PERFORMANCE_OPTIMIZATION.md)
- [Troubleshooting Guide](./LLM_TROUBLESHOOTING_GUIDE.md)

---

🎯 **Congratulations!** You now have a fully functional AI inference system running on your IoT device. The model is optimized for edge deployment and ready for real-world applications!

# 🔌 IoT Edge Device Deployment Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying small, optimized AI models on IoT edge devices using the Zig AI platform. Perfect for resource-constrained environments where low latency, offline operation, and minimal power consumption are critical.

## 🎯 Target Use Cases

### Edge AI Applications
- **Smart Cameras**: Real-time object detection and classification
- **Industrial IoT**: Predictive maintenance and anomaly detection
- **Smart Home**: Voice commands and gesture recognition
- **Autonomous Vehicles**: Real-time decision making
- **Medical Devices**: Patient monitoring and diagnostic assistance
- **Agricultural IoT**: Crop monitoring and pest detection

### Performance Targets
- **Inference Latency**: <50ms for real-time applications
- **Memory Usage**: <512MB RAM for model and runtime
- **Power Consumption**: <5W for battery-powered devices
- **Model Size**: <100MB for efficient storage and updates
- **CPU Usage**: <80% to allow for other system processes

## 🔧 Supported IoT Platforms

### ARM-based Devices
- **Raspberry Pi 4/5** (4GB+ RAM recommended)
- **NVIDIA Jetson Nano/Xavier** (GPU acceleration)
- **Qualcomm Snapdragon** (mobile processors)
- **ARM Cortex-A** series processors

### x86-based Edge Devices
- **Intel NUC** (compact form factor)
- **Intel Atom/Celeron** processors
- **AMD Ryzen Embedded** series

### Microcontrollers (Limited Support)
- **ESP32** (for very small models)
- **Arduino** (basic inference only)
- **STM32** (with sufficient memory)

## 📦 Model Optimization for IoT

### 1. Model Quantization

```yaml
# iot-model-config.yaml
model:
  name: "TinyBERT-IoT"
  base_model: "distilbert-base-uncased"
  target_device: "raspberry-pi-4"
  
quantization:
  enabled: true
  # INT8 quantization for 4x size reduction
  precision: "int8"
  # Dynamic quantization for better accuracy
  dynamic: true
  # Calibration for better quantization
  calibration_samples: 1000
  
optimization:
  # Remove unnecessary layers
  layer_pruning: true
  # Reduce vocabulary size
  vocab_pruning: true
  # Optimize for inference
  inference_only: true
```

### 2. Model Pruning and Distillation

```bash
# Model optimization script
#!/bin/bash
# optimize-for-iot.sh

echo "🔧 Optimizing model for IoT deployment..."

# Install optimization tools
pip install torch torchvision onnx onnxruntime
pip install neural-compressor # Intel optimization toolkit

# Convert and optimize model
python scripts/optimize_model.py \
  --input-model ./models/original-model.onnx \
  --output-model ./models/iot-optimized-model.onnx \
  --target-device raspberry-pi \
  --quantization int8 \
  --pruning 0.3 \
  --optimization-level 3

echo "✅ Model optimization completed"
echo "📊 Original size: $(du -h ./models/original-model.onnx | cut -f1)"
echo "📊 Optimized size: $(du -h ./models/iot-optimized-model.onnx | cut -f1)"
```

### 3. Model Architecture Selection

```yaml
# Recommended model architectures for IoT
architectures:
  # Text Processing
  text:
    - name: "TinyBERT"
      size: "14MB"
      use_case: "Text classification, sentiment analysis"
      latency: "10ms"
    
    - name: "DistilBERT"
      size: "65MB"
      use_case: "Question answering, NLU"
      latency: "25ms"
  
  # Computer Vision
  vision:
    - name: "MobileNetV3"
      size: "5MB"
      use_case: "Image classification"
      latency: "15ms"
    
    - name: "EfficientNet-B0"
      size: "20MB"
      use_case: "Object detection"
      latency: "30ms"
    
    - name: "YOLOv5n"
      size: "7MB"
      use_case: "Real-time object detection"
      latency: "20ms"
  
  # Audio Processing
  audio:
    - name: "Wav2Vec2-Base"
      size: "95MB"
      use_case: "Speech recognition"
      latency: "40ms"
    
    - name: "SpeechT5"
      size: "45MB"
      use_case: "Text-to-speech"
      latency: "35ms"
```

## 🏗️ IoT Device Setup

### Raspberry Pi 4/5 Setup

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install dependencies
sudo apt install -y python3-pip python3-venv git cmake build-essential
sudo apt install -y libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev

# 3. Enable GPU memory split (for Pi with GPU)
sudo raspi-config
# Advanced Options -> Memory Split -> 128

# 4. Install optimized libraries
pip3 install numpy==1.21.0  # Optimized for ARM
pip3 install onnxruntime==1.12.0  # ARM-optimized build
pip3 install opencv-python-headless==4.6.0.66

# 5. Clone and setup Zig AI
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform
```

### NVIDIA Jetson Setup

```bash
# 1. Install JetPack SDK
sudo apt update
sudo apt install nvidia-jetpack

# 2. Install CUDA-optimized libraries
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install onnxruntime-gpu

# 3. Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# 4. Verify GPU access
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Intel NUC Setup

```bash
# 1. Install Intel OpenVINO toolkit
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_ubuntu20_2023.0.0.10926.b4452d56304_x86_64.tgz
tar -xf l_openvino_toolkit_ubuntu20_2023.0.10926.b4452d56304_x86_64.tgz
cd l_openvino_toolkit_ubuntu20_2023.0.0.10926.b4452d56304_x86_64
sudo ./install.sh

# 2. Setup environment
source /opt/intel/openvino_2023/setupvars.sh
echo 'source /opt/intel/openvino_2023/setupvars.sh' >> ~/.bashrc

# 3. Install optimized runtime
pip3 install openvino-dev[onnx,tensorflow2]
```

## 🚀 IoT Deployment Architecture

### Edge-Only Deployment

```yaml
# iot-edge-config.yaml
deployment:
  type: "edge-only"
  description: "Fully autonomous edge deployment"
  
  device:
    type: "raspberry-pi-4"
    memory: "4GB"
    storage: "32GB"
    
  runtime:
    framework: "onnxruntime"
    optimization: "cpu"
    threads: 4
    
  model:
    path: "/opt/zig-ai/models/iot-model.onnx"
    cache_size: "256MB"
    batch_size: 1
    
  networking:
    offline_mode: true
    telemetry_endpoint: "https://iot-hub.azure.com"
    update_check_interval: "24h"
```

### Edge-Cloud Hybrid

```yaml
# iot-hybrid-config.yaml
deployment:
  type: "edge-cloud-hybrid"
  description: "Edge inference with cloud fallback"
  
  edge:
    primary_model: "lightweight-model.onnx"
    confidence_threshold: 0.8
    max_latency: "50ms"
    
  cloud:
    fallback_endpoint: "https://zig-ai-coordinator-service:8080/v1/inference"
    fallback_conditions:
      - low_confidence: true
      - complex_query: true
      - edge_failure: true
    
  data_sync:
    enabled: true
    sync_interval: "1h"
    compression: true
```

## 📱 Container Deployment for IoT

### Docker Setup for ARM Devices

```dockerfile
# Dockerfile.iot-arm64
FROM arm64v8/ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-iot.txt .
RUN pip3 install --no-cache-dir -r requirements-iot.txt

# Copy optimized model and application
COPY models/iot-optimized-model.onnx /app/models/
COPY src/iot/ /app/src/
COPY config/iot-config.yaml /app/config/

WORKDIR /app

# Set resource limits
ENV OMP_NUM_THREADS=4
ENV MALLOC_ARENA_MAX=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import requests; requests.get('http://localhost:8080/health')"

EXPOSE 8080

CMD ["python3", "src/iot_inference_server.py"]
```

### Docker Compose for IoT Stack

```yaml
# docker-compose.iot.yml
version: '3.8'

services:
  iot-inference:
    build:
      context: .
      dockerfile: Dockerfile.iot-arm64
    container_name: zig-ai-iot
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - /dev:/dev  # For hardware access
    environment:
      - ZIG_AI_MODE=iot
      - ZIG_AI_DEVICE_TYPE=raspberry-pi
      - ZIG_AI_LOG_LEVEL=info
    devices:
      - /dev/gpiomem:/dev/gpiomem  # GPIO access
    privileged: false
    security_opt:
      - no-new-privileges:true
    mem_limit: 1g
    cpus: '2.0'
    
  iot-monitor:
    image: prom/node-exporter:latest
    container_name: iot-monitor
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'

networks:
  default:
    driver: bridge
```

### Kubernetes Deployment for Edge Clusters

```yaml
# iot-edge-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zig-ai-iot-edge
  namespace: iot-edge
  labels:
    app: zig-ai-iot
    tier: edge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: zig-ai-iot
  template:
    metadata:
      labels:
        app: zig-ai-iot
    spec:
      nodeSelector:
        kubernetes.io/arch: arm64
        device-type: raspberry-pi
      containers:
      - name: iot-inference
        image: zigai/iot-inference:arm64-v1.0.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "2"
        env:
        - name: ZIG_AI_MODE
          value: "iot"
        - name: ZIG_AI_DEVICE_TYPE
          value: "raspberry-pi"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
        - name: device-access
          mountPath: /dev
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: model-storage
        configMap:
          name: iot-model-config
      - name: device-access
        hostPath:
          path: /dev
---
apiVersion: v1
kind: Service
metadata:
  name: zig-ai-iot-service
  namespace: iot-edge
spec:
  selector:
    app: zig-ai-iot
  ports:
  - port: 8080
    targetPort: 8080
    nodePort: 30080
  type: NodePort
```

## ⚡ Performance Optimization

### CPU Optimization

```bash
# CPU performance tuning for IoT devices
#!/bin/bash
# iot-performance-tuning.sh

echo "🔧 Optimizing IoT device performance..."

# 1. Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 2. Disable CPU frequency scaling
sudo systemctl disable ondemand

# 3. Optimize memory settings
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf

# 4. Set process priorities
echo 'zig-ai-iot soft priority 10' | sudo tee -a /etc/security/limits.conf
echo 'zig-ai-iot hard priority 15' | sudo tee -a /etc/security/limits.conf

# 5. Optimize network settings
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf

sudo sysctl -p
echo "✅ Performance optimization completed"

### Memory Optimization

```python
# iot_memory_optimizer.py
import gc
import psutil
import threading
import time

class IoTMemoryOptimizer:
    def __init__(self, max_memory_mb=512):
        self.max_memory_mb = max_memory_mb
        self.monitoring = True

    def start_monitoring(self):
        """Start memory monitoring thread"""
        monitor_thread = threading.Thread(target=self._monitor_memory)
        monitor_thread.daemon = True
        monitor_thread.start()

    def _monitor_memory(self):
        """Monitor and optimize memory usage"""
        while self.monitoring:
            memory_usage = psutil.virtual_memory().percent

            if memory_usage > 80:  # High memory usage
                self._optimize_memory()

            time.sleep(10)  # Check every 10 seconds

    def _optimize_memory(self):
        """Perform memory optimization"""
        # Force garbage collection
        gc.collect()

        # Clear model cache if needed
        if hasattr(self, 'model_cache'):
            self.model_cache.clear()

        print(f"🧹 Memory optimized. Current usage: {psutil.virtual_memory().percent}%")

# Usage in IoT application
optimizer = IoTMemoryOptimizer(max_memory_mb=512)
optimizer.start_monitoring()
```

### Power Management

```bash
# iot-power-management.sh
#!/bin/bash

echo "🔋 Configuring power management for IoT device..."

# 1. Enable power saving mode
sudo systemctl enable systemd-sleep

# 2. Configure CPU frequency scaling
echo 'powersave' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 3. Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-powersave@wlan0

# 4. Configure USB power management
echo 'auto' | sudo tee /sys/bus/usb/devices/*/power/control

# 5. Set up wake-on-LAN for remote management
sudo ethtool -s eth0 wol g

echo "✅ Power management configured"
```

## 📊 Monitoring and Telemetry

### IoT Device Monitoring

```python
# iot_telemetry.py
import json
import time
import psutil
import requests
from datetime import datetime

class IoTTelemetry:
    def __init__(self, device_id, endpoint_url):
        self.device_id = device_id
        self.endpoint_url = endpoint_url
        self.metrics = {}

    def collect_metrics(self):
        """Collect device metrics"""
        self.metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'device_id': self.device_id,
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'temperature': self._get_temperature(),
            'inference_count': self._get_inference_count(),
            'average_latency': self._get_average_latency(),
            'model_accuracy': self._get_model_accuracy(),
            'power_consumption': self._get_power_consumption()
        }

    def _get_temperature(self):
        """Get device temperature (Raspberry Pi specific)"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read()) / 1000.0
                return temp
        except:
            return None

    def _get_inference_count(self):
        """Get inference count from application metrics"""
        # Implementation depends on your metrics collection
        return 0

    def _get_average_latency(self):
        """Get average inference latency"""
        # Implementation depends on your metrics collection
        return 0.0

    def _get_model_accuracy(self):
        """Get model accuracy metrics"""
        # Implementation depends on your validation system
        return 0.0

    def _get_power_consumption(self):
        """Get power consumption (if available)"""
        # Implementation depends on hardware capabilities
        return None

    def send_telemetry(self):
        """Send telemetry data to cloud endpoint"""
        try:
            response = requests.post(
                self.endpoint_url,
                json=self.metrics,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send telemetry: {e}")
            return False

# Usage
telemetry = IoTTelemetry("rpi-001", "https://iot-hub.azure.com/telemetry")

while True:
    telemetry.collect_metrics()
    telemetry.send_telemetry()
    time.sleep(60)  # Send every minute
```

### Prometheus Metrics for IoT

```yaml
# iot-prometheus-config.yml
global:
  scrape_interval: 30s
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'iot-devices'
    static_configs:
      - targets: ['192.168.1.100:9100']  # IoT device IP
    scrape_interval: 15s
    metrics_path: /metrics

  - job_name: 'iot-inference'
    static_configs:
      - targets: ['192.168.1.100:8080']
    scrape_interval: 10s
    metrics_path: /metrics

rule_files:
  - "iot_alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## 🔄 Model Updates and Management

### Over-the-Air (OTA) Updates

```python
# iot_model_updater.py
import os
import hashlib
import requests
import shutil
from pathlib import Path

class IoTModelUpdater:
    def __init__(self, model_dir="/opt/zig-ai/models", update_endpoint="https://api.zig-ai.com/models"):
        self.model_dir = Path(model_dir)
        self.update_endpoint = update_endpoint
        self.current_version = self._get_current_version()

    def check_for_updates(self):
        """Check if model updates are available"""
        try:
            response = requests.get(f"{self.update_endpoint}/latest")
            latest_version = response.json()['version']

            if latest_version != self.current_version:
                return latest_version
            return None
        except Exception as e:
            print(f"Failed to check for updates: {e}")
            return None

    def download_model(self, version):
        """Download new model version"""
        try:
            download_url = f"{self.update_endpoint}/{version}/download"
            response = requests.get(download_url, stream=True)

            temp_path = self.model_dir / f"model-{version}.tmp"

            with open(temp_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

            # Verify checksum
            if self._verify_checksum(temp_path, version):
                return temp_path
            else:
                temp_path.unlink()
                return None

        except Exception as e:
            print(f"Failed to download model: {e}")
            return None

    def _verify_checksum(self, file_path, version):
        """Verify downloaded file checksum"""
        try:
            response = requests.get(f"{self.update_endpoint}/{version}/checksum")
            expected_checksum = response.text.strip()

            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            return sha256_hash.hexdigest() == expected_checksum
        except:
            return False

    def install_model(self, temp_path, version):
        """Install new model version"""
        try:
            # Backup current model
            current_model = self.model_dir / "current_model.onnx"
            backup_model = self.model_dir / f"backup_model_{self.current_version}.onnx"

            if current_model.exists():
                shutil.copy2(current_model, backup_model)

            # Install new model
            shutil.move(temp_path, current_model)

            # Update version file
            with open(self.model_dir / "version.txt", 'w') as f:
                f.write(version)

            self.current_version = version
            return True

        except Exception as e:
            print(f"Failed to install model: {e}")
            return False

    def rollback(self):
        """Rollback to previous model version"""
        try:
            backup_files = list(self.model_dir.glob("backup_model_*.onnx"))
            if backup_files:
                latest_backup = max(backup_files, key=os.path.getctime)
                current_model = self.model_dir / "current_model.onnx"

                shutil.copy2(latest_backup, current_model)
                print("Model rolled back successfully")
                return True
            return False
        except Exception as e:
            print(f"Failed to rollback: {e}")
            return False

    def _get_current_version(self):
        """Get current model version"""
        try:
            with open(self.model_dir / "version.txt", 'r') as f:
                return f.read().strip()
        except:
            return "unknown"

# Usage
updater = IoTModelUpdater()

# Check for updates every hour
import schedule
schedule.every().hour.do(lambda: updater.check_for_updates())
```

## 🛡️ Security for IoT Deployments

### Secure Communication

```python
# iot_security.py
import ssl
import jwt
import hashlib
import hmac
from cryptography.fernet import Fernet

class IoTSecurity:
    def __init__(self, device_key, api_key):
        self.device_key = device_key
        self.api_key = api_key
        self.cipher = Fernet(device_key)

    def encrypt_data(self, data):
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode())

    def decrypt_data(self, encrypted_data):
        """Decrypt received data"""
        return self.cipher.decrypt(encrypted_data).decode()

    def generate_device_token(self, device_id):
        """Generate JWT token for device authentication"""
        payload = {
            'device_id': device_id,
            'exp': time.time() + 3600  # 1 hour expiry
        }
        return jwt.encode(payload, self.api_key, algorithm='HS256')

    def verify_message_integrity(self, message, signature):
        """Verify message integrity using HMAC"""
        expected_signature = hmac.new(
            self.device_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    def create_secure_context(self):
        """Create SSL context for secure communication"""
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED
        return context

## 🧪 Testing and Validation

### Performance Benchmarking

```python
# iot_benchmark.py
import time
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class IoTBenchmark:
    def __init__(self, inference_function):
        self.inference_function = inference_function
        self.results = []

    def run_latency_test(self, num_requests=100):
        """Test inference latency"""
        latencies = []

        for i in range(num_requests):
            start_time = time.time()
            result = self.inference_function(f"test_input_{i}")
            end_time = time.time()

            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)

        return {
            'mean_latency': statistics.mean(latencies),
            'median_latency': statistics.median(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'min_latency': min(latencies),
            'max_latency': max(latencies)
        }

    def run_throughput_test(self, duration_seconds=60, max_workers=4):
        """Test inference throughput"""
        start_time = time.time()
        request_count = 0

        def make_request():
            nonlocal request_count
            self.inference_function(f"throughput_test_{request_count}")
            request_count += 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while time.time() - start_time < duration_seconds:
                executor.submit(make_request)

        throughput = request_count / duration_seconds
        return {
            'requests_per_second': throughput,
            'total_requests': request_count,
            'test_duration': duration_seconds
        }

    def run_memory_test(self, num_requests=1000):
        """Test memory usage during inference"""
        import psutil

        initial_memory = psutil.virtual_memory().used
        peak_memory = initial_memory

        for i in range(num_requests):
            self.inference_function(f"memory_test_{i}")
            current_memory = psutil.virtual_memory().used
            peak_memory = max(peak_memory, current_memory)

        memory_increase = peak_memory - initial_memory
        return {
            'initial_memory_mb': initial_memory / 1024 / 1024,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'memory_increase_mb': memory_increase / 1024 / 1024
        }

# Usage example
def dummy_inference(input_text):
    time.sleep(0.01)  # Simulate inference time
    return {"result": "processed"}

benchmark = IoTBenchmark(dummy_inference)
latency_results = benchmark.run_latency_test()
throughput_results = benchmark.run_throughput_test()
memory_results = benchmark.run_memory_test()

print(f"Average latency: {latency_results['mean_latency']:.2f}ms")
print(f"Throughput: {throughput_results['requests_per_second']:.2f} RPS")
print(f"Memory increase: {memory_results['memory_increase_mb']:.2f}MB")
```

### Hardware Stress Testing

```bash
#!/bin/bash
# iot-stress-test.sh

echo "🔥 Starting IoT device stress test..."

# 1. CPU stress test
echo "Testing CPU performance..."
stress-ng --cpu 4 --timeout 60s --metrics-brief

# 2. Memory stress test
echo "Testing memory performance..."
stress-ng --vm 2 --vm-bytes 256M --timeout 60s --metrics-brief

# 3. I/O stress test
echo "Testing I/O performance..."
stress-ng --io 2 --timeout 60s --metrics-brief

# 4. Temperature monitoring during stress
echo "Monitoring temperature during stress..."
for i in {1..60}; do
    temp=$(cat /sys/class/thermal/thermal_zone0/temp)
    echo "Temperature: $((temp/1000))°C"
    sleep 1
done

echo "✅ Stress test completed"
```

## 🚨 Troubleshooting IoT Deployments

### Common Issues and Solutions

#### Issue 1: High Memory Usage

**Symptoms:**
- Device becomes unresponsive
- Out of memory errors
- Slow inference times

**Solutions:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Optimize memory settings
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf

# Enable memory monitoring
python3 iot_memory_optimizer.py
```

#### Issue 2: Model Loading Failures

**Symptoms:**
- "Model not found" errors
- Corrupted model files
- Version mismatch errors

**Solutions:**
```bash
# Verify model integrity
md5sum /opt/zig-ai/models/current_model.onnx

# Check model format
python3 -c "import onnx; model = onnx.load('current_model.onnx'); print('Model loaded successfully')"

# Rollback to previous version
python3 -c "from iot_model_updater import IoTModelUpdater; updater = IoTModelUpdater(); updater.rollback()"
```

#### Issue 3: Network Connectivity Issues

**Symptoms:**
- Failed telemetry uploads
- Model update failures
- Cloud communication errors

**Solutions:**
```bash
# Check network connectivity
ping -c 4 8.8.8.8
curl -I https://api.zig-ai.com/health

# Test DNS resolution
nslookup api.zig-ai.com

# Check firewall settings
sudo ufw status
sudo iptables -L
```

### Diagnostic Tools

```bash
#!/bin/bash
# iot-diagnostics.sh

echo "🔍 IoT Device Diagnostics"
echo "========================"

echo "📊 System Information:"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "Uptime: $(uptime -p)"

echo -e "\n💾 Memory Information:"
free -h

echo -e "\n💽 Disk Usage:"
df -h

echo -e "\n🌡️ Temperature:"
if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    temp=$(cat /sys/class/thermal/thermal_zone0/temp)
    echo "CPU Temperature: $((temp/1000))°C"
fi

echo -e "\n🔌 USB Devices:"
lsusb

echo -e "\n🌐 Network Interfaces:"
ip addr show

echo -e "\n🏃 Running Processes:"
ps aux --sort=-%cpu | head -10

echo -e "\n📦 Docker Containers:"
if command -v docker &> /dev/null; then
    docker ps
fi

echo -e "\n🔧 Zig AI Status:"
if systemctl is-active --quiet zig-ai-iot; then
    echo "✅ Zig AI service is running"
else
    echo "❌ Zig AI service is not running"
fi
```

## 📱 Real-World Examples

### Smart Camera Application

```python
# smart_camera_app.py
import cv2
import numpy as np
from zig_ai_iot import IoTInferenceEngine

class SmartCamera:
    def __init__(self, model_path, camera_id=0):
        self.engine = IoTInferenceEngine(model_path)
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def detect_objects(self):
        """Real-time object detection"""
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)

            # Run inference
            start_time = time.time()
            results = self.engine.infer(input_tensor)
            inference_time = time.time() - start_time

            # Post-process results
            annotated_frame = self.draw_detections(frame, results)

            # Display FPS and inference time
            fps_text = f"FPS: {1/inference_time:.1f} | Latency: {inference_time*1000:.1f}ms"
            cv2.putText(annotated_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Smart Camera', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()

    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize and normalize
        resized = cv2.resize(frame, (416, 416))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def draw_detections(self, frame, results):
        """Draw detection boxes on frame"""
        for detection in results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']

            if confidence > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

# Usage
camera = SmartCamera("/opt/zig-ai/models/yolov5n.onnx")
camera.detect_objects()
```

### Voice Assistant for IoT

```python
# voice_assistant_iot.py
import speech_recognition as sr
import pyttsx3
from zig_ai_iot import IoTInferenceEngine

class VoiceAssistant:
    def __init__(self, model_path):
        self.engine = IoTInferenceEngine(model_path)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts = pyttsx3.init()

        # Configure TTS for IoT device
        self.tts.setProperty('rate', 150)
        self.tts.setProperty('volume', 0.8)

    def listen_and_respond(self):
        """Main voice assistant loop"""
        print("🎤 Voice assistant started. Say 'wake up' to activate.")

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                # Listen for wake word
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)

                text = self.recognizer.recognize_google(audio).lower()

                if "wake up" in text:
                    self.tts.say("How can I help you?")
                    self.tts.runAndWait()

                    # Listen for command
                    with self.microphone as source:
                        audio = self.recognizer.listen(source, timeout=5)

                    command = self.recognizer.recognize_google(audio)
                    response = self.process_command(command)

                    self.tts.say(response)
                    self.tts.runAndWait()

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"Error: {e}")

    def process_command(self, command):
        """Process voice command using AI model"""
        # Use IoT inference engine for intent recognition
        intent_result = self.engine.infer({
            "text": command,
            "task": "intent_classification"
        })

        intent = intent_result['intent']
        confidence = intent_result['confidence']

        if confidence < 0.7:
            return "I'm not sure what you mean. Can you try again?"

        # Handle different intents
        if intent == "weather":
            return self.get_weather()
        elif intent == "time":
            return self.get_time()
        elif intent == "control_device":
            return self.control_device(command)
        else:
            return "I can help with weather, time, or device control."

    def get_weather(self):
        """Get weather information"""
        # Implement weather API call or local sensor reading
        return "The current temperature is 22 degrees Celsius."

    def get_time(self):
        """Get current time"""
        from datetime import datetime
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}"

    def control_device(self, command):
        """Control IoT devices"""
        # Implement device control logic
        if "lights" in command.lower():
            if "on" in command.lower():
                return "Turning on the lights."
            elif "off" in command.lower():
                return "Turning off the lights."
        return "Device control command processed."

# Usage
assistant = VoiceAssistant("/opt/zig-ai/models/intent_classifier.onnx")
assistant.listen_and_respond()
```

## 📋 Deployment Checklist

### Pre-deployment Checklist
- [ ] IoT device meets minimum hardware requirements
- [ ] Operating system updated and configured
- [ ] Model optimized for target device (quantized, pruned)
- [ ] Container images built for target architecture
- [ ] Network connectivity tested
- [ ] Security certificates and keys configured
- [ ] Monitoring and telemetry endpoints configured

### Post-deployment Checklist
- [ ] Model loading successful
- [ ] Inference latency within acceptable limits
- [ ] Memory usage optimized
- [ ] Temperature monitoring active
- [ ] Telemetry data flowing to cloud
- [ ] OTA update mechanism tested
- [ ] Backup and recovery procedures verified
- [ ] Performance benchmarks established

### Production Readiness Checklist
- [ ] Security hardening completed
- [ ] Automated monitoring and alerting configured
- [ ] Model update pipeline tested
- [ ] Disaster recovery plan documented
- [ ] Performance optimization completed
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Compliance requirements met

## 🔗 Additional Resources

### Hardware Vendors
- [Raspberry Pi Foundation](https://www.raspberrypi.org/)
- [NVIDIA Jetson](https://developer.nvidia.com/embedded/jetson-developer-kits)
- [Intel NUC](https://www.intel.com/content/www/us/en/products/details/nuc.html)

### Software Tools
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [PyTorch Mobile](https://pytorch.org/mobile/)

### Optimization Tools
- [Neural Compressor](https://github.com/intel/neural-compressor)
- [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)
- [TensorRT](https://developer.nvidia.com/tensorrt)

---

## 🎯 Summary

This guide provides a comprehensive approach to deploying AI models on IoT edge devices. The Zig AI platform enables efficient inference on resource-constrained devices while maintaining high performance and reliability.

**Key Benefits:**
- ✅ **Low Latency**: <50ms inference for real-time applications
- ✅ **Resource Efficient**: Optimized for devices with <1GB RAM
- ✅ **Offline Capable**: Full functionality without internet connectivity
- ✅ **Secure**: End-to-end encryption and secure model updates
- ✅ **Scalable**: From single devices to large IoT deployments

For cloud-scale deployments, see the [Massive LLM Deployment Guide](./MASSIVE_LLM_DEPLOYMENT_GUIDE.md).
```
```

# Distributed Model Sharding Deployment Guide

## 🎯 Overview

This guide explains how to deploy and run large language models (like GPT-3) across multiple VMs using horizontal model sharding in the Zig AI Inference Engine.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
├─────────────────────────────────────────────────────────────┤
│                 Coordinator Node                            │
│              (zig-ai-platform)                              │
├─────────────────────────────────────────────────────────────┤
│  Shard 1 VM     │  Shard 2 VM     │  Shard 3 VM            │
│  Layers 0-11    │  Layers 12-23   │  Layers 24-35          │
│  ┌───────────┐  │  ┌───────────┐  │  ┌───────────┐         │
│  │model-server│  │  │model-server│  │  │model-server│         │
│  │inference  │  │  │inference  │  │  │inference  │         │
│  │tensor-core│  │  │tensor-core│  │  │tensor-core│         │
│  └───────────┘  │  └───────────┘  │  └───────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

**Hardware Requirements:**
- **Coordinator Node**: 16GB RAM, 8 CPU cores
- **Shard VMs**: 32-64GB RAM each, 16+ CPU cores, optional GPU
- **Network**: 10Gbps+ between VMs for optimal performance

**Software Requirements:**
- Zig 0.11+ on all nodes
- Linux/Ubuntu 20.04+ (recommended)
- Docker (optional, for containerized deployment)

### 2. Build the System

```bash
# Clone the repository
git clone https://github.com/anachary/zig-ai-inference-engine.git
cd zig-ai-inference-engine

# Build all components
zig build build-all

# Build distributed components specifically
zig build distributed
```

### 3. Prepare Model Shards

```bash
# Create model sharding configuration
cat > distributed_config.json << EOF
{
  "model_path": "models/gpt3-175b.onnx",
  "total_layers": 96,
  "shards_count": 8,
  "max_shard_memory_mb": 32768,
  "replication_factor": 2,
  "load_balancing_strategy": "least_loaded"
}
EOF

# Split large model into shards (if not already done)
zig build run -- shard-model --config distributed_config.json
```

## 🖥️ VM Setup

### Coordinator Node Setup

```bash
# 1. Install coordinator
sudo mkdir -p /opt/zig-ai/coordinator
sudo cp zig-out/bin/zig-ai-platform /opt/zig-ai/coordinator/

# 2. Create coordinator configuration
cat > /opt/zig-ai/coordinator/config.yaml << EOF
coordinator:
  listen_port: 8080
  max_connections: 1000
  health_check_interval: 5s
  
distributed:
  shards_count: 8
  replication_factor: 2
  failover_timeout: 30s
  
memory:
  max_pool_size_gb: 64
  enable_compression: true
  
logging:
  level: info
  file: /var/log/zig-ai/coordinator.log
EOF

# 3. Create systemd service
sudo tee /etc/systemd/system/zig-ai-coordinator.service << EOF
[Unit]
Description=Zig AI Distributed Coordinator
After=network.target

[Service]
Type=simple
User=zig-ai
WorkingDirectory=/opt/zig-ai/coordinator
ExecStart=/opt/zig-ai/coordinator/zig-ai-platform coordinator --config config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 4. Start coordinator
sudo systemctl enable zig-ai-coordinator
sudo systemctl start zig-ai-coordinator
```

### Shard VM Setup

```bash
# 1. Install shard server on each VM
sudo mkdir -p /opt/zig-ai/shard
sudo cp zig-out/bin/zig-model-server /opt/zig-ai/shard/

# 2. Create shard configuration (customize per VM)
cat > /opt/zig-ai/shard/config.yaml << EOF
shard:
  shard_id: 1  # Unique per VM
  coordinator_address: "10.0.1.100:8080"
  listen_port: 8080
  
model:
  shard_path: "/data/model_shard_1.onnx"
  layers_start: 0
  layers_end: 12
  
resources:
  max_memory_gb: 32
  worker_threads: 16
  enable_gpu: true
  
health:
  check_interval: 10s
  timeout: 30s
EOF

# 3. Create systemd service
sudo tee /etc/systemd/system/zig-ai-shard.service << EOF
[Unit]
Description=Zig AI Model Shard Server
After=network.target

[Service]
Type=simple
User=zig-ai
WorkingDirectory=/opt/zig-ai/shard
ExecStart=/opt/zig-ai/shard/zig-model-server shard --config config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 4. Start shard server
sudo systemctl enable zig-ai-shard
sudo systemctl start zig-ai-shard
```

## 🐳 Docker Deployment

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  coordinator:
    build: 
      context: .
      dockerfile: docker/Dockerfile.coordinator
    ports:
      - "8080:8080"
    environment:
      - ZIG_AI_MODE=coordinator
      - ZIG_AI_SHARDS_COUNT=4
    volumes:
      - ./config:/config
      - ./logs:/logs
    networks:
      - zig-ai-network

  shard-1:
    build:
      context: .
      dockerfile: docker/Dockerfile.shard
    environment:
      - ZIG_AI_MODE=shard
      - ZIG_AI_SHARD_ID=1
      - ZIG_AI_COORDINATOR=coordinator:8080
      - ZIG_AI_LAYERS_START=0
      - ZIG_AI_LAYERS_END=24
    volumes:
      - ./models/shard_1:/data
    depends_on:
      - coordinator
    networks:
      - zig-ai-network

  shard-2:
    build:
      context: .
      dockerfile: docker/Dockerfile.shard
    environment:
      - ZIG_AI_MODE=shard
      - ZIG_AI_SHARD_ID=2
      - ZIG_AI_COORDINATOR=coordinator:8080
      - ZIG_AI_LAYERS_START=24
      - ZIG_AI_LAYERS_END=48
    volumes:
      - ./models/shard_2:/data
    depends_on:
      - coordinator
    networks:
      - zig-ai-network

  # Add more shards as needed...

networks:
  zig-ai-network:
    driver: bridge
```

### Deploy with Docker

```bash
# Build and start all services
docker-compose up -d

# Scale shards
docker-compose up -d --scale shard-1=2 --scale shard-2=2

# Monitor logs
docker-compose logs -f coordinator
docker-compose logs -f shard-1
```

## ☸️ Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s/coordinator-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zig-ai-coordinator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: zig-ai-coordinator
  template:
    metadata:
      labels:
        app: zig-ai-coordinator
    spec:
      containers:
      - name: coordinator
        image: zig-ai/coordinator:latest
        ports:
        - containerPort: 8080
        env:
        - name: ZIG_AI_MODE
          value: "coordinator"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
---
apiVersion: v1
kind: Service
metadata:
  name: zig-ai-coordinator-service
spec:
  selector:
    app: zig-ai-coordinator
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

```yaml
# k8s/shard-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zig-ai-shard
spec:
  replicas: 4  # Number of shards
  selector:
    matchLabels:
      app: zig-ai-shard
  template:
    metadata:
      labels:
        app: zig-ai-shard
    spec:
      containers:
      - name: shard
        image: zig-ai/shard:latest
        ports:
        - containerPort: 8080
        env:
        - name: ZIG_AI_MODE
          value: "shard"
        - name: ZIG_AI_COORDINATOR
          value: "zig-ai-coordinator-service:8080"
        resources:
          requests:
            memory: "32Gi"
            cpu: "16"
            nvidia.com/gpu: 1
          limits:
            memory: "64Gi"
            cpu: "32"
            nvidia.com/gpu: 1
```

### Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Scale shards
kubectl scale deployment zig-ai-shard --replicas=8

# Monitor deployment
kubectl get pods -l app=zig-ai-coordinator
kubectl get pods -l app=zig-ai-shard

# Check logs
kubectl logs -f deployment/zig-ai-coordinator
kubectl logs -f deployment/zig-ai-shard
```

## 🔧 Configuration

### Model Configuration

```yaml
# models/distributed_config.yaml
model:
  name: "GPT-3-175B"
  format: "onnx"
  total_parameters: 175000000000
  total_layers: 96
  vocab_size: 50257
  
sharding:
  strategy: "layer_wise"
  shards_count: 8
  layers_per_shard: 12
  overlap_layers: 1  # For smooth transitions
  
memory:
  max_shard_memory_gb: 32
  enable_quantization: true
  quantization_bits: 16
  
performance:
  batch_size: 1
  max_sequence_length: 2048
  enable_kv_cache: true
```

### Network Configuration

```yaml
# network/config.yaml
network:
  coordinator_port: 8080
  shard_base_port: 8081
  health_check_port: 9090
  metrics_port: 9091
  
communication:
  protocol: "http"  # or "grpc"
  compression: "gzip"
  timeout_seconds: 30
  retry_count: 3
  
load_balancing:
  strategy: "least_loaded"
  health_check_interval: 5
  failover_timeout: 30
```

## 📊 Monitoring

### Prometheus Metrics

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'zig-ai-coordinator'
    static_configs:
      - targets: ['coordinator:9091']
  
  - job_name: 'zig-ai-shards'
    static_configs:
      - targets: ['shard-1:9091', 'shard-2:9091', 'shard-3:9091', 'shard-4:9091']
```

### Grafana Dashboard

Key metrics to monitor:
- **Inference latency** per shard
- **Memory usage** across VMs
- **Network bandwidth** between shards
- **GPU utilization** (if applicable)
- **Error rates** and failover events
- **Request throughput**

## 🛡️ Security

### Network Security

```bash
# Firewall rules (iptables)
# Allow coordinator access
sudo iptables -A INPUT -p tcp --dport 8080 -s 10.0.1.0/24 -j ACCEPT

# Allow inter-shard communication
sudo iptables -A INPUT -p tcp --dport 8081:8088 -s 10.0.1.0/24 -j ACCEPT

# Block external access to shards
sudo iptables -A INPUT -p tcp --dport 8081:8088 -j DROP
```

### TLS Configuration

```yaml
# security/tls.yaml
tls:
  enabled: true
  cert_file: "/etc/ssl/certs/zig-ai.crt"
  key_file: "/etc/ssl/private/zig-ai.key"
  ca_file: "/etc/ssl/certs/ca.crt"
  
authentication:
  enabled: true
  method: "jwt"
  secret_key: "${JWT_SECRET}"
  token_expiry: "24h"
```

## 🚀 Usage Examples

### Basic Inference

```bash
# Single inference request
curl -X POST http://coordinator:8080/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Batch Inference

```bash
# Batch processing
curl -X POST http://coordinator:8080/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"prompt": "Explain quantum computing", "max_tokens": 50},
      {"prompt": "What is machine learning?", "max_tokens": 50}
    ]
  }'
```

### Health Check

```bash
# Check system health
curl http://coordinator:8080/api/v1/health

# Check individual shard
curl http://shard-1:8080/api/v1/health
```

## 🔍 Troubleshooting

### Common Issues

1. **Shard Connection Failures**
   ```bash
   # Check network connectivity
   ping shard-1
   telnet shard-1 8080
   
   # Check shard logs
   journalctl -u zig-ai-shard -f
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   free -h
   cat /proc/meminfo
   
   # Check shard memory allocation
   curl http://shard-1:8080/api/v1/stats
   ```

3. **Performance Issues**
   ```bash
   # Check CPU usage
   top -p $(pgrep zig-model-server)
   
   # Monitor network bandwidth
   iftop -i eth0
   ```

## 📈 Performance Tuning

### Optimization Tips

1. **Network Optimization**
   - Use 10Gbps+ networking
   - Enable TCP window scaling
   - Tune network buffers

2. **Memory Optimization**
   - Use memory pools
   - Enable compression for transfers
   - Implement smart caching

3. **GPU Optimization**
   - Use tensor parallelism
   - Optimize memory transfers
   - Enable mixed precision

This distributed sharding system enables running massive models like GPT-3 across multiple VMs with fault tolerance, load balancing, and high performance!

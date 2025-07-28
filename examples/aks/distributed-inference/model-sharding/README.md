# Distributed Model Sharding on AKS

## 🎯 Overview

This example demonstrates horizontal model sharding across multiple nodes in Azure Kubernetes Service (AKS) using the Zig AI Platform. Perfect for running large models like Mistral 7B that don't fit on a single node.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AKS Cluster                              │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Shard 0       │   Shard 1       │   Shard 2               │
│  (Layers 0-7)   │  (Layers 8-15)  │  (Layers 16-23)         │
│                 │                 │                         │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────────┐ │
│ │ Node Pool 1 │ │ │ Node Pool 2 │ │ │ Node Pool 3         │ │
│ │ 4 vCPUs     │ │ │ 4 vCPUs     │ │ │ 4 vCPUs             │ │
│ │ 16GB RAM    │ │ │ 16GB RAM    │ │ │ 16GB RAM            │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────────────┘ │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
                    ┌─────────────┐
                    │ Load        │
                    │ Balancer    │
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │ Client      │
                    │ Requests    │
                    └─────────────┘
```

## 🔧 Prerequisites

- **Azure CLI** installed and configured
- **kubectl** configured for your AKS cluster
- **Helm 3** installed
- **Docker** for building container images

## 🚀 Quick Start

### 1. Create AKS Cluster

```bash
# Create resource group
az group create --name zig-ai-rg --location eastus

# Create AKS cluster with multiple node pools
az aks create \
  --resource-group zig-ai-rg \
  --name zig-ai-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group zig-ai-rg --name zig-ai-cluster
```

### 2. Deploy Model Sharding

```bash
# Clone the repository
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform/examples/aks/distributed-inference/model-sharding/

# Build and push container images
./scripts/build_and_push.sh

# Deploy using Helm
helm install zig-ai-sharding ./helm-chart/
```

### 3. Test Distributed Inference

```bash
# Port forward to access the service
kubectl port-forward svc/zig-ai-coordinator 8080:80

# Test inference
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of distributed AI is", "max_tokens": 50}'
```

## 📁 Project Structure

```
model-sharding/
├── src/
│   ├── coordinator/              # Coordination service
│   │   ├── main.zig             # Main coordinator
│   │   ├── shard_manager.zig    # Shard management
│   │   └── load_balancer.zig    # Request routing
│   ├── worker/                  # Worker node service
│   │   ├── main.zig             # Worker main
│   │   ├── model_shard.zig      # Model shard handler
│   │   └── inference.zig        # Inference engine
│   └── common/
│       ├── protocol.zig         # Communication protocol
│       ├── model_config.zig     # Model configuration
│       └── metrics.zig          # Performance metrics
├── docker/
│   ├── coordinator.Dockerfile   # Coordinator container
│   ├── worker.Dockerfile        # Worker container
│   └── base.Dockerfile          # Base image
├── helm-chart/
│   ├── Chart.yaml              # Helm chart metadata
│   ├── values.yaml             # Default values
│   └── templates/              # Kubernetes templates
├── k8s/
│   ├── coordinator.yaml        # Coordinator deployment
│   ├── worker.yaml             # Worker deployment
│   ├── service.yaml            # Service definitions
│   └── configmap.yaml          # Configuration
├── scripts/
│   ├── build_and_push.sh       # Build and push images
│   ├── deploy.sh               # Deployment script
│   └── benchmark.sh            # Performance testing
└── README.md                   # This file
```

## 🔧 Implementation Details

### Coordinator Service

```zig
// src/coordinator/main.zig
const std = @import("std");
const ai = @import("implementations");
const protocol = @import("../common/protocol.zig");

pub const Coordinator = struct {
    allocator: std.mem.Allocator,
    shard_manager: ShardManager,
    load_balancer: LoadBalancer,
    http_server: std.http.Server,
    
    pub fn init(allocator: std.mem.Allocator) !Coordinator {
        return Coordinator{
            .allocator = allocator,
            .shard_manager = try ShardManager.init(allocator),
            .load_balancer = try LoadBalancer.init(allocator),
            .http_server = std.http.Server.init(allocator, .{}),
        };
    }
    
    pub fn start(self: *Coordinator, port: u16) !void {
        try self.http_server.listen(std.net.Address.parseIp("0.0.0.0", port) catch unreachable);
        
        std.log.info("Coordinator started on port {}", .{port});
        
        while (true) {
            var response = try self.http_server.accept(.{});
            defer response.deinit();
            
            try self.handleRequest(&response);
        }
    }
    
    fn handleRequest(self: *Coordinator, response: *std.http.Server.Response) !void {
        const request_body = try response.reader().readAllAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(request_body);
        
        // Parse inference request
        const request = try protocol.InferenceRequest.parse(request_body);
        
        // Route to appropriate shards
        const result = try self.distributeInference(request);
        
        // Send response
        try response.headers.append("content-type", "application/json");
        try response.do();
        try response.writeAll(result);
    }
    
    fn distributeInference(self: *Coordinator, request: protocol.InferenceRequest) ![]const u8 {
        // Get available shards
        const shards = try self.shard_manager.getAvailableShards();
        
        // Distribute computation across shards
        var results = std.ArrayList(protocol.ShardResult).init(self.allocator);
        defer results.deinit();
        
        for (shards) |shard| {
            const shard_request = protocol.ShardRequest{
                .shard_id = shard.id,
                .input_tokens = request.input_tokens,
                .layer_range = shard.layer_range,
            };
            
            const result = try self.sendToShard(shard, shard_request);
            try results.append(result);
        }
        
        // Combine results
        return self.combineResults(results.items);
    }
};
```

### Worker Node Service

```zig
// src/worker/main.zig
const std = @import("std");
const ai = @import("implementations");
const protocol = @import("../common/protocol.zig");

pub const Worker = struct {
    allocator: std.mem.Allocator,
    model_shard: ModelShard,
    inference_engine: ai.ExecutionEngine,
    shard_id: u32,
    
    pub fn init(allocator: std.mem.Allocator, shard_id: u32, model_path: []const u8) !Worker {
        // Initialize AI platform for worker
        var platform = try ai.utils.createTransformerPlatform(allocator);
        
        // Load model shard
        const model_shard = try ModelShard.init(allocator, model_path, shard_id);
        
        return Worker{
            .allocator = allocator,
            .model_shard = model_shard,
            .inference_engine = platform.getFramework().execution_engine,
            .shard_id = shard_id,
        };
    }
    
    pub fn processRequest(self: *Worker, request: protocol.ShardRequest) !protocol.ShardResult {
        // Process tokens through this shard's layers
        const start_time = std.time.nanoTimestamp();
        
        // Run inference on assigned layers
        const output = try self.model_shard.forward(request.input_tokens, request.layer_range);
        
        const end_time = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        
        return protocol.ShardResult{
            .shard_id = self.shard_id,
            .output_tokens = output,
            .processing_time_ms = duration_ms,
            .memory_used_mb = self.getMemoryUsage(),
        };
    }
    
    fn getMemoryUsage(self: *Worker) f64 {
        // Get current memory usage
        const stats = self.inference_engine.getExecutionStats();
        return @as(f64, @floatFromInt(stats.total_memory_used)) / (1024.0 * 1024.0);
    }
};
```

### Model Shard Implementation

```zig
// src/worker/model_shard.zig
const std = @import("std");
const ai = @import("implementations");

pub const ModelShard = struct {
    allocator: std.mem.Allocator,
    platform: *ai.AIPlatform,
    graph: ai.Graph,
    layer_range: LayerRange,
    
    pub const LayerRange = struct {
        start: u32,
        end: u32,
    };
    
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, shard_id: u32) !ModelShard {
        var platform = try ai.utils.createTransformerPlatform(allocator);
        
        // Load only the layers for this shard
        var graph = platform.createGraph();
        try loadModelShard(&graph, model_path, shard_id);
        
        return ModelShard{
            .allocator = allocator,
            .platform = &platform,
            .graph = graph,
            .layer_range = calculateLayerRange(shard_id),
        };
    }
    
    pub fn forward(self: *ModelShard, input_tokens: []const u32, layer_range: LayerRange) ![]f32 {
        // Create input tensor
        const batch_size = 1;
        const seq_len = input_tokens.len;
        const hidden_dim = 4096; // Mistral 7B hidden dimension
        
        const input_shape = [_]usize{ batch_size, seq_len, hidden_dim };
        var input_tensor = try ai.utils.createTensor(self.allocator, &input_shape, .f32);
        defer input_tensor.deinit();
        
        // Convert tokens to embeddings (if this is the first shard)
        if (layer_range.start == 0) {
            try self.tokenToEmbedding(input_tokens, &input_tensor);
        }
        
        // Set graph inputs
        try self.graph.setInput(0, input_tensor);
        
        // Execute the shard
        try self.platform.executeGraph(&self.graph);
        
        // Get output
        const output_tensor = self.graph.getOutput(0) orelse return error.NoOutput;
        return output_tensor.getData(f32);
    }
    
    fn tokenToEmbedding(self: *ModelShard, tokens: []const u32, output: *ai.Tensor) !void {
        // Convert token IDs to embeddings using the embedding layer
        const embedding_weights = try self.getEmbeddingWeights();
        const hidden_dim = 4096;
        
        const output_data = output.getMutableData(f32);
        
        for (tokens, 0..) |token_id, i| {
            const embedding_offset = token_id * hidden_dim;
            const output_offset = i * hidden_dim;
            
            for (0..hidden_dim) |j| {
                output_data[output_offset + j] = embedding_weights[embedding_offset + j];
            }
        }
    }
    
    fn calculateLayerRange(shard_id: u32) LayerRange {
        const total_layers = 32; // Mistral 7B has 32 layers
        const layers_per_shard = total_layers / 3; // Assuming 3 shards
        
        return LayerRange{
            .start = shard_id * layers_per_shard,
            .end = if (shard_id == 2) total_layers else (shard_id + 1) * layers_per_shard,
        };
    }
};
```

## 🐳 Docker Configuration

### Base Image (`docker/base.Dockerfile`)
```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Zig
RUN curl -L https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz | tar -xJ \
    && mv zig-linux-x86_64-0.11.0 /opt/zig \
    && ln -s /opt/zig/zig /usr/local/bin/zig

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Build the application
RUN zig build -Doptimize=ReleaseFast
```

### Coordinator Image (`docker/coordinator.Dockerfile`)
```dockerfile
FROM zig-ai-base:latest

# Expose coordinator port
EXPOSE 8080

# Run coordinator
CMD ["./zig-out/bin/coordinator"]
```

### Worker Image (`docker/worker.Dockerfile`)
```dockerfile
FROM zig-ai-base:latest

# Copy model files
COPY models/ /app/models/

# Expose worker port
EXPOSE 8081

# Run worker
CMD ["./zig-out/bin/worker"]
```

## ☸️ Kubernetes Deployment

### Coordinator Deployment (`k8s/coordinator.yaml`)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zig-ai-coordinator
  labels:
    app: zig-ai-coordinator
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
        image: zigai/coordinator:latest
        ports:
        - containerPort: 8080
        env:
        - name: WORKER_ENDPOINTS
          value: "zig-ai-worker-0:8081,zig-ai-worker-1:8081,zig-ai-worker-2:8081"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: zig-ai-coordinator
spec:
  selector:
    app: zig-ai-coordinator
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Worker StatefulSet (`k8s/worker.yaml`)
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: zig-ai-worker
spec:
  serviceName: zig-ai-worker
  replicas: 3
  selector:
    matchLabels:
      app: zig-ai-worker
  template:
    metadata:
      labels:
        app: zig-ai-worker
    spec:
      containers:
      - name: worker
        image: zigai/worker:latest
        ports:
        - containerPort: 8081
        env:
        - name: SHARD_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: MODEL_PATH
          value: "/app/models/mistral-7b-shard-$(SHARD_ID).onnx"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2000m"
          limits:
            memory: "16Gi"
            cpu: "4000m"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
  volumeClaimTemplates:
  - metadata:
      name: model-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
```

## 📊 Performance Monitoring

### Metrics Collection
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'zig-ai-coordinator'
      static_configs:
      - targets: ['zig-ai-coordinator:8080']
    - job_name: 'zig-ai-workers'
      static_configs:
      - targets: ['zig-ai-worker-0:8081', 'zig-ai-worker-1:8081', 'zig-ai-worker-2:8081']
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Zig AI Distributed Inference",
    "panels": [
      {
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage by Shard",
        "type": "graph",
        "targets": [
          {
            "expr": "memory_usage_bytes{job=\"zig-ai-workers\"}",
            "legendFormat": "Shard {{instance}}"
          }
        ]
      },
      {
        "title": "Throughput",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(inference_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      }
    ]
  }
}
```

## 🧪 Testing and Benchmarking

### Load Testing Script (`scripts/benchmark.sh`)
```bash
#!/bin/bash

# Configuration
COORDINATOR_URL="http://localhost:8080"
CONCURRENT_REQUESTS=10
TOTAL_REQUESTS=100

echo "🚀 Starting distributed inference benchmark..."

# Install dependencies
if ! command -v hey &> /dev/null; then
    echo "Installing hey load testing tool..."
    go install github.com/rakyll/hey@latest
fi

# Prepare test payload
cat > test_payload.json << EOF
{
  "prompt": "The future of distributed artificial intelligence systems will revolutionize",
  "max_tokens": 100,
  "temperature": 0.7
}
EOF

# Run load test
echo "Running load test with $CONCURRENT_REQUESTS concurrent requests..."
hey -n $TOTAL_REQUESTS -c $CONCURRENT_REQUESTS -m POST \
    -H "Content-Type: application/json" \
    -D test_payload.json \
    $COORDINATOR_URL/generate

# Clean up
rm test_payload.json

echo "✅ Benchmark completed!"
```

## 🔧 Scaling and Optimization

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: zig-ai-coordinator-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: zig-ai-coordinator
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Cluster Autoscaler
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
  namespace: kube-system
data:
  nodes.max: "10"
  nodes.min: "3"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
```

This example demonstrates real-world distributed inference deployment on AKS, showing how to scale large AI models across multiple nodes efficiently! 🚀

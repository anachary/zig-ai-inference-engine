# AKS Deployment Example: GPT-3 Scale Distributed Inference

## 🎯 Overview

This example demonstrates deploying a GPT-3 scale model (175B parameters) across Azure Kubernetes Service (AKS) using horizontal model sharding.

## 🏗️ Architecture on AKS

```
┌─────────────────────────────────────────────────────────────┐
│                Azure Load Balancer                         │
│              (External IP: 20.x.x.x)                       │
├─────────────────────────────────────────────────────────────┤
│                    AKS Cluster                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Coordinator Pod                          │   │
│  │         (Standard_D8s_v3 node)                      │   │
│  │    - Request routing                                │   │
│  │    - Load balancing                                 │   │
│  │    - Health monitoring                              │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│  │ Shard Pod 1 │ │ Shard Pod 2 │ │ Shard Pod 3 │         │
│  │ Layers 0-11 │ │Layers 12-23 │ │Layers 24-35 │         │
│  │NC24s_v3 GPU │ │NC24s_v3 GPU │ │NC24s_v3 GPU │         │
│  │   32GB RAM  │ │   32GB RAM  │ │   32GB RAM  │         │
│  └─────────────┘ └─────────────┘ └─────────────┘         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│  │ Shard Pod 4 │ │ Shard Pod 5 │ │ Shard Pod 6 │         │
│  │Layers 36-47 │ │Layers 48-59 │ │Layers 60-71 │         │
│  │NC24s_v3 GPU │ │NC24s_v3 GPU │ │NC24s_v3 GPU │         │
│  └─────────────┘ └─────────────┘ └─────────────┘         │
│  ┌─────────────┐ ┌─────────────┐                         │
│  │ Shard Pod 7 │ │ Shard Pod 8 │                         │
│  │Layers 72-83 │ │Layers 84-95 │                         │
│  │NC24s_v3 GPU │ │NC24s_v3 GPU │                         │
│  └─────────────┘ └─────────────┘                         │
├─────────────────────────────────────────────────────────────┤
│ Azure Container Registry │ Azure Files │ Azure Monitor     │
│     (Model Images)        │ (5TB Models)│  (Monitoring)     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Step-by-Step Deployment

### 1. Prerequisites Setup

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set subscription
az account set --subscription "your-subscription-id"

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/
```

### 2. Deploy Infrastructure

```bash
# Clone the repository
git clone https://github.com/anachary/zig-ai-inference-engine.git
cd zig-ai-inference-engine

# Set environment variables
export RESOURCE_GROUP="zig-ai-production"
export CLUSTER_NAME="zig-ai-aks-prod"
export LOCATION="eastus"
export ACR_NAME="zigairegistry$(date +%s)"

# Run the automated deployment script
chmod +x deploy/aks/scripts/deploy-to-aks.sh
./deploy/aks/scripts/deploy-to-aks.sh
```

### 3. Manual Step-by-Step (Alternative)

#### Create Resource Group and ACR
```bash
# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Premium \
  --admin-enabled true
```

#### Create AKS Cluster
```bash
# Create AKS cluster with GPU support
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --location $LOCATION \
  --node-count 3 \
  --node-vm-size Standard_D8s_v3 \
  --enable-addons monitoring \
  --attach-acr $ACR_NAME \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 15 \
  --kubernetes-version 1.28.0 \
  --enable-managed-identity \
  --generate-ssh-keys

# Add GPU node pool
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name gpunodes \
  --node-count 8 \
  --node-vm-size Standard_NC24s_v3 \
  --enable-cluster-autoscaler \
  --min-count 4 \
  --max-count 16 \
  --node-taints nvidia.com/gpu=true:NoSchedule \
  --labels accelerator=nvidia-tesla-v100
```

#### Configure kubectl
```bash
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME
```

### 4. Deploy GPU Support

```bash
# Install NVIDIA GPU Operator
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

helm install --wait gpu-operator \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --set driver.enabled=true
```

### 5. Build and Push Images

```bash
# Login to ACR
az acr login --name $ACR_NAME
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)

# Build images
docker build -f deploy/aks/Dockerfile.coordinator -t $ACR_LOGIN_SERVER/zig-ai/coordinator:v1.0 .
docker build -f deploy/aks/Dockerfile.shard -t $ACR_LOGIN_SERVER/zig-ai/shard:v1.0 .

# Push images
docker push $ACR_LOGIN_SERVER/zig-ai/coordinator:v1.0
docker push $ACR_LOGIN_SERVER/zig-ai/shard:v1.0
```

### 6. Deploy Application

```bash
# Create namespace
kubectl create namespace zig-ai

# Deploy with Helm
helm install zig-ai ./deploy/aks/helm/zig-ai \
  --namespace zig-ai \
  --set global.imageRegistry=$ACR_LOGIN_SERVER \
  --set coordinator.replicas=1 \
  --set shard.replicas=8 \
  --set shard.resources.requests.nvidia\.com/gpu=1 \
  --set shard.resources.limits.nvidia\.com/gpu=1 \
  --set storage.modelStorage.size=5Ti \
  --wait --timeout=20m
```

## 📊 Resource Requirements

### Cluster Specifications
- **Total Nodes**: 9-17 (auto-scaling)
- **Coordinator Node**: 1x Standard_D8s_v3 (8 vCPUs, 32GB RAM)
- **GPU Nodes**: 8x Standard_NC24s_v3 (24 vCPUs, 448GB RAM, 4x Tesla V100)
- **Total GPUs**: 32x Tesla V100
- **Total Memory**: 3.6TB RAM
- **Storage**: 5TB Azure Files Premium

### Cost Estimation (East US)
- **GPU Nodes**: ~$6,000/month (8x NC24s_v3)
- **CPU Nodes**: ~$200/month (1x D8s_v3)
- **Storage**: ~$500/month (5TB Premium)
- **Network**: ~$100/month
- **Total**: ~$6,800/month

## 🔧 Configuration Examples

### Production Values (values-production.yaml)
```yaml
global:
  imageRegistry: "zigairegistry.azurecr.io"

coordinator:
  replicas: 1
  resources:
    requests:
      memory: "16Gi"
      cpu: "8"
    limits:
      memory: "32Gi"
      cpu: "16"

shard:
  replicas: 8
  resources:
    requests:
      memory: "64Gi"
      cpu: "20"
      nvidia.com/gpu: 1
    limits:
      memory: "128Gi"
      cpu: "24"
      nvidia.com/gpu: 1
  
  nodeSelector:
    accelerator: nvidia-tesla-v100
  
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule

autoscaling:
  enabled: true
  minReplicas: 4
  maxReplicas: 16
  targetCPUUtilizationPercentage: 70

storage:
  modelStorage:
    enabled: true
    storageClass: "azurefile-premium"
    size: "5Ti"
```

## 🚀 Usage Examples

### 1. Basic Inference Request
```bash
# Get external IP
EXTERNAL_IP=$(kubectl get service zig-ai-coordinator-service -n zig-ai -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test inference
curl -X POST http://$EXTERNAL_IP/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the future of artificial intelligence?",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### 2. Batch Processing
```bash
curl -X POST http://$EXTERNAL_IP/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "prompt": "Explain quantum computing",
        "max_tokens": 100
      },
      {
        "prompt": "What is machine learning?",
        "max_tokens": 100
      },
      {
        "prompt": "Describe neural networks",
        "max_tokens": 100
      }
    ]
  }'
```

### 3. Health Check
```bash
curl http://$EXTERNAL_IP/api/v1/health
```

## 📈 Monitoring and Scaling

### View Cluster Status
```bash
# Check pods
kubectl get pods -n zig-ai -o wide

# Check GPU usage
kubectl top nodes

# Check HPA status
kubectl get hpa -n zig-ai

# View logs
kubectl logs -f deployment/zig-ai-coordinator -n zig-ai
kubectl logs -f deployment/zig-ai-shard -n zig-ai
```

### Manual Scaling
```bash
# Scale shards
kubectl scale deployment zig-ai-shard --replicas=12 -n zig-ai

# Scale cluster nodes
az aks nodepool scale \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name gpunodes \
  --node-count 12
```

### Monitoring Dashboard
```bash
# Access Grafana (if installed)
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Access Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

## 🔍 Troubleshooting

### Common Issues

1. **GPU Not Available**
```bash
# Check GPU operator
kubectl get pods -n gpu-operator

# Check node labels
kubectl get nodes --show-labels | grep nvidia
```

2. **Image Pull Errors**
```bash
# Check ACR integration
az aks check-acr --name $CLUSTER_NAME --resource-group $RESOURCE_GROUP --acr $ACR_NAME

# Verify image exists
az acr repository list --name $ACR_NAME
```

3. **Pod Scheduling Issues**
```bash
# Check node resources
kubectl describe nodes

# Check pod events
kubectl describe pod <pod-name> -n zig-ai
```

4. **Performance Issues**
```bash
# Check resource usage
kubectl top pods -n zig-ai

# Check network connectivity
kubectl exec -it <coordinator-pod> -n zig-ai -- ping <shard-pod-ip>
```

## 🧹 Cleanup

```bash
# Delete application
helm uninstall zig-ai -n zig-ai

# Delete namespace
kubectl delete namespace zig-ai

# Delete AKS cluster
az aks delete --name $CLUSTER_NAME --resource-group $RESOURCE_GROUP --yes

# Delete resource group (optional)
az group delete --name $RESOURCE_GROUP --yes
```

## 🎯 Performance Expectations

### Latency
- **Single inference**: 2-5 seconds
- **Batch inference**: 1-3 seconds per item
- **Cold start**: 30-60 seconds

### Throughput
- **Concurrent requests**: 50-100
- **Tokens per second**: 400-800
- **Daily capacity**: 10M+ tokens

### Scaling
- **Auto-scale trigger**: 70% CPU/GPU
- **Scale-up time**: 2-5 minutes
- **Scale-down time**: 10-15 minutes

This AKS deployment provides enterprise-grade scalability, monitoring, and reliability for running GPT-3 scale models in production!

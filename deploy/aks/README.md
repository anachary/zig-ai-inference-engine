# Azure Kubernetes Service (AKS) Deployment Guide

## 🎯 Overview

Deploy the Zig AI Distributed Inference Engine on Azure Kubernetes Service (AKS) for scalable, production-ready AI model serving with horizontal model sharding.

## 🏗️ AKS Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure Load Balancer                     │
├─────────────────────────────────────────────────────────────┤
│                    AKS Cluster                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Coordinator Pod                      │   │
│  │            (zig-ai-platform)                        │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│  │  Shard Pod  │ │  Shard Pod  │ │  Shard Pod  │         │
│  │  Layers 0-11│ │ Layers 12-23│ │ Layers 24-35│         │
│  │  GPU Node   │ │  GPU Node   │ │  GPU Node   │         │
│  └─────────────┘ └─────────────┘ └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  Azure Container Registry │ Azure Storage │ Azure Monitor  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Azure CLI** installed and configured
2. **kubectl** installed
3. **Helm 3.x** installed
4. **Docker** for building images
5. **Azure subscription** with sufficient quota

### 1. Create AKS Cluster

```bash
# Set variables
export RESOURCE_GROUP="zig-ai-rg"
export CLUSTER_NAME="zig-ai-aks"
export LOCATION="eastus"
export ACR_NAME="zigairegistry"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
az acr create --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME --sku Premium

# Create AKS cluster with GPU support
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --location $LOCATION \
  --node-count 3 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-addons monitoring \
  --attach-acr $ACR_NAME \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10 \
  --kubernetes-version 1.28.0
```

### 2. Configure kubectl

```bash
# Get AKS credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

# Verify connection
kubectl get nodes
```

### 3. Install NVIDIA GPU Operator

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

# Install GPU operator
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator
```

## 🐳 Build and Push Images

### Build Docker Images

```bash
# Build coordinator image
docker build -f deploy/aks/Dockerfile.coordinator -t $ACR_NAME.azurecr.io/zig-ai/coordinator:latest .

# Build shard image
docker build -f deploy/aks/Dockerfile.shard -t $ACR_NAME.azurecr.io/zig-ai/shard:latest .

# Login to ACR
az acr login --name $ACR_NAME

# Push images
docker push $ACR_NAME.azurecr.io/zig-ai/coordinator:latest
docker push $ACR_NAME.azurecr.io/zig-ai/shard:latest
```

## ⚙️ Deploy with Helm

### Install Zig AI Helm Chart

```bash
# Add custom Helm repository (if available)
helm repo add zig-ai-aks ./deploy/aks/helm/zig-ai-aks
helm repo update

# Install with custom values
helm install zig-ai-aks zig-ai-aks/zig-ai-aks \
  --namespace zig-ai \
  --create-namespace \
  --values deploy/aks/values-production.yaml \
  --set image.registry=$ACR_NAME.azurecr.io \
  --set coordinator.replicas=1 \
  --set shard.replicas=8 \
  --set shard.resources.requests.nvidia\.com/gpu=1
```

### Custom Values for Production

```yaml
# deploy/aks/values-production.yaml
global:
  imageRegistry: "zigairegistry.azurecr.io"
  storageClass: "managed-premium"
  
coordinator:
  replicas: 1
  image:
    repository: "zig-ai/coordinator"
    tag: "latest"
  resources:
    requests:
      memory: "8Gi"
      cpu: "4"
    limits:
      memory: "16Gi"
      cpu: "8"
  service:
    type: LoadBalancer
    port: 8080

shard:
  replicas: 8
  image:
    repository: "zig-ai/shard"
    tag: "latest"
  resources:
    requests:
      memory: "32Gi"
      cpu: "16"
      nvidia.com/gpu: 1
    limits:
      memory: "64Gi"
      cpu: "32"
      nvidia.com/gpu: 1
  
  # Node affinity for GPU nodes
  nodeSelector:
    accelerator: nvidia-tesla-v100
  
  # Anti-affinity to spread shards across nodes
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - zig-ai-shard
        topologyKey: kubernetes.io/hostname

storage:
  enabled: true
  storageClass: "managed-premium"
  size: "1Ti"
  
monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
    
autoscaling:
  enabled: true
  minReplicas: 4
  maxReplicas: 16
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

## 📊 Monitoring and Observability

### Azure Monitor Integration

```bash
# Enable Azure Monitor for containers
az aks enable-addons \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --addons monitoring
```

### Prometheus and Grafana

```bash
# Install kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values deploy/aks/monitoring-values.yaml
```

## 🔧 Configuration Management

### Azure Key Vault Integration

```bash
# Create Key Vault
az keyvault create \
  --name "zig-ai-keyvault" \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Store secrets
az keyvault secret set \
  --vault-name "zig-ai-keyvault" \
  --name "model-access-key" \
  --value "your-secret-key"

# Install Azure Key Vault Provider for Secrets Store CSI Driver
helm repo add csi-secrets-store-provider-azure https://azure.github.io/secrets-store-csi-driver-provider-azure/charts
helm install csi csi-secrets-store-provider-azure/csi-secrets-store-provider-azure \
  --namespace kube-system
```

## 🚀 Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
# deploy/aks/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: zig-ai-shard-hpa
  namespace: zig-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: zig-ai-shard
  minReplicas: 4
  maxReplicas: 16
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
  - type: Pods
    pods:
      metric:
        name: inference_requests_per_second
      target:
        type: AverageValue
        averageValue: "10"
```

### Cluster Autoscaler

```bash
# Update cluster with autoscaler settings
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 20
```

## 💾 Storage Configuration

### Azure Files for Model Storage

```yaml
# deploy/aks/storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
  namespace: zig-ai
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: azurefile-premium
  resources:
    requests:
      storage: 1Ti
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shard-storage
  namespace: zig-ai
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: managed-premium
  resources:
    requests:
      storage: 500Gi
```

## 🔒 Security Configuration

### Network Policies

```yaml
# deploy/aks/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: zig-ai-network-policy
  namespace: zig-ai
spec:
  podSelector:
    matchLabels:
      app: zig-ai-shard
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: zig-ai-coordinator
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: zig-ai-coordinator
    ports:
    - protocol: TCP
      port: 8080
```

### Azure Active Directory Integration

```bash
# Enable Azure AD integration
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --enable-aad \
  --aad-admin-group-object-ids "your-admin-group-id"
```

## 📈 Performance Optimization

### GPU Node Pools

```bash
# Create dedicated GPU node pool for shards
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name gpunodes \
  --node-count 4 \
  --node-vm-size Standard_NC24s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 8 \
  --node-taints nvidia.com/gpu=true:NoSchedule
```

### CPU Node Pool for Coordinator

```bash
# Create CPU-optimized node pool for coordinator
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name cpunodes \
  --node-count 2 \
  --node-vm-size Standard_D16s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 4
```

## 🔍 Troubleshooting

### Common AKS Issues

```bash
# Check node status
kubectl get nodes -o wide

# Check pod status
kubectl get pods -n zig-ai

# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Check logs
kubectl logs -n zig-ai deployment/zig-ai-coordinator
kubectl logs -n zig-ai deployment/zig-ai-shard

# Check resource usage
kubectl top nodes
kubectl top pods -n zig-ai
```

### Debug GPU Issues

```bash
# Check GPU operator status
kubectl get pods -n gpu-operator

# Test GPU access
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:11.0-base \
  --limits=nvidia.com/gpu=1 \
  -- nvidia-smi
```

## 🚀 Deployment Commands

### Quick Deploy Script

```bash
#!/bin/bash
# deploy/aks/quick-deploy.sh

set -e

echo "🚀 Deploying Zig AI to AKS..."

# Build and push images
./deploy/aks/build-images.sh

# Deploy infrastructure
kubectl apply -f deploy/aks/namespace.yaml
kubectl apply -f deploy/aks/storage.yaml
kubectl apply -f deploy/aks/secrets.yaml

# Deploy application
helm upgrade --install zig-ai-aks ./deploy/aks/helm/zig-ai-aks \
  --namespace zig-ai \
  --values deploy/aks/values-production.yaml \
  --wait --timeout=10m

# Setup monitoring
kubectl apply -f deploy/aks/monitoring/

# Setup autoscaling
kubectl apply -f deploy/aks/hpa.yaml

echo "✅ Deployment completed!"
echo "🌐 Access the service at:"
kubectl get service -n zig-ai zig-ai-coordinator-service
```

This AKS deployment provides enterprise-grade scalability, monitoring, and security for running GPT-3 scale models in Azure!

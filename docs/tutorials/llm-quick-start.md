# 🚀 Quick Start: Deploy LLM on AKS in 30 Minutes

## 📋 Overview

This quick start guide helps you deploy a large language model on Azure Kubernetes Service (AKS) in approximately 30 minutes. For comprehensive details, see the [Massive LLM Deployment Guide](./MASSIVE_LLM_DEPLOYMENT_GUIDE.md).

## ⚡ Prerequisites

- Azure CLI installed and logged in
- kubectl installed
- Helm installed
- Docker installed
- Azure subscription with GPU quota (request increase if needed)

## 🏃‍♂️ Quick Deployment Steps

### Step 1: Set Environment Variables (2 minutes)

```bash
# Set your configuration
export RESOURCE_GROUP="zig-ai-quickstart-rg"
export LOCATION="eastus"
export ACR_NAME="zigaiquickstart$(date +%s)"
export AKS_CLUSTER_NAME="zig-ai-cluster"
export STORAGE_ACCOUNT="zigaistorage$(date +%s)"

# Login to Azure
az login
```

### Step 2: Create Azure Resources (8 minutes)

```bash
# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create ACR
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Standard --admin-enabled true

# Create storage account
az storage account create --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS

# Create AKS cluster with GPU nodes
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_CLUSTER_NAME \
  --node-count 1 \
  --node-vm-size Standard_D2s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Add GPU node pool
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $AKS_CLUSTER_NAME \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --node-taints nvidia.com/gpu=true:NoSchedule
```

### Step 3: Configure Cluster (5 minutes)

```bash
# Get cluster credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -l agentpool=gpupool
```

### Step 4: Build and Deploy (10 minutes)

```bash
# Clone repository
git clone https://github.com/anachary/zig-ai-platform.git
cd zig-ai-platform

# Login to ACR and build images
az acr login --name $ACR_NAME
docker build -f deploy/aks/Dockerfile.coordinator -t $ACR_NAME.azurecr.io/zig-ai/coordinator:latest .
docker build -f deploy/aks/Dockerfile.shard -t $ACR_NAME.azurecr.io/zig-ai/shard:latest .
docker push $ACR_NAME.azurecr.io/zig-ai/coordinator:latest
docker push $ACR_NAME.azurecr.io/zig-ai/shard:latest

# Create namespace and secrets
kubectl create namespace zig-ai
kubectl create secret docker-registry acr-secret \
  --namespace zig-ai \
  --docker-server=$ACR_NAME.azurecr.io \
  --docker-username=$ACR_NAME \
  --docker-password=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

# Deploy with Helm
helm install zig-ai-aks ./deploy/aks/helm/zig-ai-aks \
  --namespace zig-ai \
  --set global.imageRegistry=$ACR_NAME.azurecr.io \
  --set global.imagePullSecrets[0].name=acr-secret \
  --set coordinator.replicas=1 \
  --set shard.replicas=2 \
  --set shard.resources.requests.nvidia\.com/gpu=1 \
  --set shard.nodeSelector.agentpool=gpupool \
  --wait --timeout=15m
```

### Step 5: Test Deployment (3 minutes)

```bash
# Check pod status
kubectl get pods -n zig-ai

# Port forward to test
kubectl port-forward -n zig-ai service/zig-ai-coordinator-service 8080:8080 &

# Test inference (in another terminal)
curl -X POST http://localhost:8080/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

### Step 6: Monitor (2 minutes)

```bash
# Check logs
kubectl logs -n zig-ai deployment/zig-ai-coordinator
kubectl logs -n zig-ai deployment/zig-ai-shard

# Check GPU utilization
kubectl exec -n zig-ai deployment/zig-ai-shard -- nvidia-smi
```

## 🎯 What You've Deployed

- **Coordinator**: 1 replica managing inference requests
- **Shards**: 2 replicas with GPU acceleration for model inference
- **Storage**: Persistent volumes for model data
- **Networking**: Load balancer service for external access
- **Monitoring**: Basic logging and metrics

## 📈 Next Steps

### Scale Your Deployment

```bash
# Scale shards
kubectl scale deployment zig-ai-shard --replicas=4 -n zig-ai

# Enable autoscaling
kubectl autoscale deployment zig-ai-shard --cpu-percent=70 --min=2 --max=8 -n zig-ai
```

### Add Monitoring

```bash
# Install Prometheus and Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace
```

### Load Your Model

```bash
# Upload your model to Azure Storage
STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT --query '[0].value' --output tsv)
az storage container create --name models --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
az storage blob upload-batch --destination models --source ./your-model-directory --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
```

## 🧹 Cleanup

```bash
# Delete the resource group (removes everything)
az group delete --name $RESOURCE_GROUP --yes --no-wait
```

## 🔗 Additional Resources

- [Massive LLM Deployment Guide](./MASSIVE_LLM_DEPLOYMENT_GUIDE.md) - Comprehensive production deployment
- [Distributed Deployment Guide](./DISTRIBUTED_DEPLOYMENT_GUIDE.md) - Architecture details
- [AKS Deployment Example](../examples/aks_deployment_example.md) - More examples

## ⚠️ Important Notes

- **GPU Quota**: Ensure you have sufficient GPU quota in your Azure subscription
- **Costs**: GPU instances are expensive - monitor your usage
- **Model Size**: This quick start uses smaller models - see the full guide for massive LLMs
- **Security**: This is a basic setup - implement proper security for production

## 🆘 Troubleshooting

### Common Issues

1. **GPU nodes not ready**: Check if NVIDIA device plugin is running
2. **Image pull errors**: Verify ACR credentials and image names
3. **Pod pending**: Check resource requests and node capacity
4. **Connection refused**: Verify service and port-forward configuration

### Debug Commands

```bash
# Check cluster status
kubectl get nodes
kubectl get pods -n zig-ai
kubectl describe pod <pod-name> -n zig-ai

# Check GPU allocation
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu

# Check logs
kubectl logs -n zig-ai deployment/zig-ai-coordinator
kubectl logs -n zig-ai deployment/zig-ai-shard
```

---

🎉 **Congratulations!** You've successfully deployed a distributed LLM inference system on AKS. For production deployments, follow the comprehensive [Massive LLM Deployment Guide](./MASSIVE_LLM_DEPLOYMENT_GUIDE.md).

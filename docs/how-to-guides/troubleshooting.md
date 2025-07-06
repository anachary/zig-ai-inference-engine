# 🔧 LLM Deployment Troubleshooting Guide

## 📋 Overview

This guide helps you diagnose and resolve common issues when deploying large language models on Azure Kubernetes Service (AKS) using the Zig AI platform.

## 🚨 Common Issues and Solutions

### 1. GPU-Related Issues

#### Issue: Pods stuck in "Pending" state with GPU requests

**Symptoms:**
```bash
kubectl get pods -n zig-ai
# Shows: zig-ai-shard-xxx   0/1   Pending   0   5m
```

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n zig-ai
# Look for: "0/3 nodes are available: 3 Insufficient nvidia.com/gpu"
```

**Solutions:**

1. **Check GPU node availability:**
```bash
kubectl get nodes -l accelerator=nvidia-tesla-v100
kubectl describe node <gpu-node> | grep nvidia.com/gpu
```

2. **Verify NVIDIA device plugin:**
```bash
kubectl get pods -n kube-system | grep nvidia
kubectl logs -n kube-system <nvidia-device-plugin-pod>
```

3. **Check node taints and tolerations:**
```bash
kubectl describe node <gpu-node> | grep Taints
# Ensure your pods have matching tolerations
```

4. **Scale GPU node pool:**
```bash
az aks nodepool scale --resource-group $RESOURCE_GROUP --cluster-name $AKS_CLUSTER_NAME --name gpupool --node-count 4
```

#### Issue: GPU memory errors (CUDA out of memory)

**Symptoms:**
```bash
kubectl logs -n zig-ai deployment/zig-ai-shard
# Shows: "CUDA out of memory" or "RuntimeError: CUDA error"
```

**Solutions:**

1. **Reduce model size or batch size:**
```yaml
# In values.yaml
shard:
  config:
    maxMemoryGb: 24  # Reduce from 32
    batchSize: 2     # Reduce from 4
```

2. **Enable model quantization:**
```yaml
shard:
  config:
    enableQuantization: true
    quantizationBits: 8  # Use INT8 instead of FP16
```

3. **Check GPU memory usage:**
```bash
kubectl exec -n zig-ai deployment/zig-ai-shard -- nvidia-smi
```

### 2. Model Loading Issues

#### Issue: Model files not found or corrupted

**Symptoms:**
```bash
kubectl logs -n zig-ai deployment/zig-ai-shard
# Shows: "FileNotFoundError" or "Model loading failed"
```

**Solutions:**

1. **Verify storage mount:**
```bash
kubectl describe pod <shard-pod> -n zig-ai | grep -A 10 Mounts
kubectl exec -n zig-ai <shard-pod> -- ls -la /data/models
```

2. **Check Azure Storage connectivity:**
```bash
kubectl exec -n zig-ai <shard-pod> -- az storage blob list --container-name models --account-name $STORAGE_ACCOUNT
```

3. **Verify model file integrity:**
```bash
kubectl exec -n zig-ai <shard-pod> -- md5sum /data/models/*.onnx
```

#### Issue: Model loading timeout

**Symptoms:**
```bash
kubectl logs -n zig-ai deployment/zig-ai-shard
# Shows: "Model loading timeout" or pod restarts frequently
```

**Solutions:**

1. **Increase timeout values:**
```yaml
shard:
  config:
    modelLoadTimeout: "600s"  # Increase from 300s
    healthCheckInterval: "30s"  # Increase interval
```

2. **Add init containers for pre-loading:**
```yaml
initContainers:
- name: model-preloader
  image: busybox
  command: ['sh', '-c', 'echo "Pre-loading models..." && sleep 60']
```

### 3. Networking Issues

#### Issue: Shards cannot connect to coordinator

**Symptoms:**
```bash
kubectl logs -n zig-ai deployment/zig-ai-shard
# Shows: "Connection refused" or "Coordinator unreachable"
```

**Solutions:**

1. **Check service connectivity:**
```bash
kubectl get svc -n zig-ai
kubectl exec -n zig-ai <shard-pod> -- nslookup zig-ai-coordinator-service
kubectl exec -n zig-ai <shard-pod> -- telnet zig-ai-coordinator-service 8080
```

2. **Verify network policies:**
```bash
kubectl get networkpolicy -n zig-ai
kubectl describe networkpolicy <policy-name> -n zig-ai
```

3. **Check DNS resolution:**
```bash
kubectl exec -n zig-ai <shard-pod> -- cat /etc/resolv.conf
kubectl get svc -n kube-system | grep dns
```

#### Issue: External access not working

**Symptoms:**
- Cannot access inference endpoint from outside cluster
- Load balancer not getting external IP

**Solutions:**

1. **Check service type and status:**
```bash
kubectl get svc -n zig-ai zig-ai-coordinator-service
kubectl describe svc -n zig-ai zig-ai-coordinator-service
```

2. **Verify load balancer configuration:**
```bash
az network lb list --resource-group MC_${RESOURCE_GROUP}_${AKS_CLUSTER_NAME}_${LOCATION}
```

3. **Use port-forward for testing:**
```bash
kubectl port-forward -n zig-ai service/zig-ai-coordinator-service 8080:8080
```

### 4. Performance Issues

#### Issue: Slow inference response times

**Symptoms:**
- High latency (>10 seconds for simple requests)
- Timeouts on inference requests

**Solutions:**

1. **Check resource utilization:**
```bash
kubectl top pods -n zig-ai
kubectl top nodes
```

2. **Monitor GPU utilization:**
```bash
kubectl exec -n zig-ai deployment/zig-ai-shard -- nvidia-smi -l 1
```

3. **Optimize resource allocation:**
```yaml
shard:
  resources:
    requests:
      cpu: "16"      # Increase CPU
      memory: "64Gi" # Increase memory
    limits:
      cpu: "24"
      memory: "96Gi"
```

4. **Enable performance optimizations:**
```yaml
shard:
  config:
    enableOptimizations: true
    workerThreads: 24
    enableKVCache: true
```

#### Issue: High memory usage

**Symptoms:**
```bash
kubectl top pods -n zig-ai
# Shows high memory usage or OOMKilled pods
```

**Solutions:**

1. **Increase memory limits:**
```yaml
shard:
  resources:
    limits:
      memory: "128Gi"  # Increase limit
```

2. **Enable memory optimization:**
```yaml
shard:
  config:
    enableMemoryOptimization: true
    memoryMappedFiles: true
```

3. **Monitor memory usage:**
```bash
kubectl exec -n zig-ai <shard-pod> -- free -h
kubectl exec -n zig-ai <shard-pod> -- cat /proc/meminfo
```

### 5. Scaling Issues

#### Issue: Autoscaling not working

**Symptoms:**
- Pods not scaling up under load
- HPA shows "unknown" metrics

**Solutions:**

1. **Check HPA status:**
```bash
kubectl get hpa -n zig-ai
kubectl describe hpa zig-ai-shard-hpa -n zig-ai
```

2. **Verify metrics server:**
```bash
kubectl get pods -n kube-system | grep metrics-server
kubectl top nodes  # Should show metrics
```

3. **Check custom metrics:**
```bash
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1" | jq .
```

#### Issue: Cluster autoscaler not adding nodes

**Symptoms:**
- Pods pending but no new nodes created
- Cluster autoscaler logs show errors

**Solutions:**

1. **Check cluster autoscaler logs:**
```bash
kubectl logs -n kube-system deployment/cluster-autoscaler
```

2. **Verify node pool configuration:**
```bash
az aks nodepool show --resource-group $RESOURCE_GROUP --cluster-name $AKS_CLUSTER_NAME --name gpupool
```

3. **Check Azure quotas:**
```bash
az vm list-usage --location $LOCATION --query "[?contains(name.value, 'NC')]"
```

## 🔍 Diagnostic Commands

### General Health Check

```bash
#!/bin/bash
# health-check.sh

echo "=== Cluster Status ==="
kubectl get nodes
kubectl get pods -n zig-ai

echo "=== Resource Usage ==="
kubectl top nodes
kubectl top pods -n zig-ai

echo "=== GPU Status ==="
kubectl get nodes -l accelerator=nvidia-tesla-v100
kubectl describe nodes -l accelerator=nvidia-tesla-v100 | grep nvidia.com/gpu

echo "=== Service Status ==="
kubectl get svc -n zig-ai
kubectl get endpoints -n zig-ai

echo "=== Storage Status ==="
kubectl get pv,pvc -n zig-ai

echo "=== Recent Events ==="
kubectl get events -n zig-ai --sort-by='.lastTimestamp' | tail -10
```

### Performance Monitoring

```bash
#!/bin/bash
# performance-monitor.sh

echo "=== GPU Utilization ==="
kubectl exec -n zig-ai deployment/zig-ai-shard -- nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

echo "=== CPU and Memory ==="
kubectl exec -n zig-ai deployment/zig-ai-shard -- top -bn1 | head -20

echo "=== Network Connections ==="
kubectl exec -n zig-ai deployment/zig-ai-coordinator -- netstat -tlnp

echo "=== Disk Usage ==="
kubectl exec -n zig-ai deployment/zig-ai-shard -- df -h
```

### Log Collection

```bash
#!/bin/bash
# collect-logs.sh

mkdir -p logs/$(date +%Y%m%d_%H%M%S)
cd logs/$(date +%Y%m%d_%H%M%S)

echo "Collecting logs..."
kubectl logs -n zig-ai deployment/zig-ai-coordinator > coordinator.log
kubectl logs -n zig-ai deployment/zig-ai-shard > shard.log
kubectl describe pods -n zig-ai > pod-descriptions.txt
kubectl get events -n zig-ai > events.txt
kubectl get all -n zig-ai -o yaml > resources.yaml

echo "Logs collected in $(pwd)"
```

## 📞 Getting Help

### Before Seeking Help

1. **Collect diagnostic information:**
   - Run the health check script
   - Collect logs from affected pods
   - Note the exact error messages
   - Document steps to reproduce

2. **Check known issues:**
   - Review this troubleshooting guide
   - Check the project's GitHub issues
   - Search Azure AKS documentation

3. **Prepare environment details:**
   - AKS version: `kubectl version`
   - Node types and sizes
   - Model size and configuration
   - Resource requests and limits

### Support Channels

- **GitHub Issues**: [Create an issue](https://github.com/anachary/zig-ai-platform/issues)
- **Documentation**: [Project Documentation](../README.md)
- **Azure Support**: For AKS-specific issues

---

## 🎯 Prevention Tips

1. **Monitor resource usage** regularly
2. **Set up proper alerting** for critical metrics
3. **Test deployments** in staging environment first
4. **Keep documentation** updated with your specific configurations
5. **Regular backups** of model data and configurations
6. **Stay updated** with latest versions and security patches

Remember: Most issues are related to resource constraints, networking, or configuration mismatches. Start with the basics and work your way up to more complex diagnostics.

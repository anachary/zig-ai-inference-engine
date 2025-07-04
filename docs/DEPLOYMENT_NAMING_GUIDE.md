# 🏷️ Deployment Naming Guide

## 📋 Overview

This guide explains the improved naming conventions for the Zig AI platform deployments, specifically focusing on the Azure Kubernetes Service (AKS) deployment.

## 🔄 Naming Changes

### Before (Generic Names)
```
Helm Chart: zig-ai
Release Name: zig-ai
Description: A Helm chart for Zig AI Distributed Inference Engine
```

### After (AKS-Specific Names)
```
Helm Chart: zig-ai-aks
Release Name: zig-ai-aks
Description: A Helm chart for Zig AI Distributed LLM Inference on Azure Kubernetes Service
```

## 📁 Updated Directory Structure

```
deploy/
└── aks/
    ├── helm/
    │   └── zig-ai-aks/          # ← Renamed from 'zig-ai'
    │       ├── Chart.yaml       # ← Updated chart name and description
    │       └── values.yaml
    ├── scripts/
    │   └── deploy-to-aks.sh     # ← Updated to use new chart name
    ├── README.md                # ← Updated deployment commands
    └── ...
```

## 🚀 Updated Deployment Commands

### Helm Installation
```bash
# Old command
helm install zig-ai ./deploy/aks/helm/zig-ai

# New command
helm install zig-ai-aks ./deploy/aks/helm/zig-ai-aks
```

### Repository Addition
```bash
# Old command
helm repo add zig-ai ./deploy/aks/helm/zig-ai

# New command
helm repo add zig-ai-aks ./deploy/aks/helm/zig-ai-aks
```

### Release Management
```bash
# List releases
helm list -n zig-ai

# Upgrade release
helm upgrade zig-ai-aks ./deploy/aks/helm/zig-ai-aks --namespace zig-ai

# Uninstall release
helm uninstall zig-ai-aks -n zig-ai
```

## 🎯 Benefits of New Naming

### 1. **Clarity and Specificity**
- **Before**: `zig-ai` (could be any deployment type)
- **After**: `zig-ai-aks` (clearly indicates AKS deployment)

### 2. **Future Extensibility**
The new naming allows for multiple deployment targets:
```
zig-ai-aks          # Azure Kubernetes Service
zig-ai-eks          # Amazon Elastic Kubernetes Service (future)
zig-ai-gke          # Google Kubernetes Engine (future)
zig-ai-openshift    # Red Hat OpenShift (future)
zig-ai-local        # Local development (future)
```

### 3. **Better Documentation**
- Chart description now mentions "LLM Inference on Azure Kubernetes Service"
- Keywords include "aks", "azure", "kubernetes", "llm"
- Repository URLs updated to reflect the platform nature

### 4. **Operational Benefits**
- Easier to identify deployments in multi-cloud environments
- Clearer Helm release names in production
- Better alignment with Azure naming conventions

## 📚 Updated Documentation

All documentation has been updated to reflect the new naming:

### Deployment Guides
- [Massive LLM Deployment Guide](./MASSIVE_LLM_DEPLOYMENT_GUIDE.md)
- [Quick Start Guide](./QUICK_START_LLM_DEPLOYMENT.md)
- [AKS Deployment Example](../examples/aks_deployment_example.md)

### Configuration Files
- `deploy/aks/helm/zig-ai-aks/Chart.yaml`
- `deploy/aks/scripts/deploy-to-aks.sh`
- `deploy/aks/README.md`

## 🔧 Migration Guide

If you have existing deployments with the old naming, here's how to migrate:

### Option 1: Clean Migration (Recommended)
```bash
# 1. Backup current configuration
helm get values zig-ai -n zig-ai > backup-values.yaml

# 2. Uninstall old release
helm uninstall zig-ai -n zig-ai

# 3. Install with new name
helm install zig-ai-aks ./deploy/aks/helm/zig-ai-aks \
  --namespace zig-ai \
  --values backup-values.yaml
```

### Option 2: In-Place Upgrade
```bash
# Upgrade existing release to use new chart
helm upgrade zig-ai ./deploy/aks/helm/zig-ai-aks --namespace zig-ai

# Optionally rename the release (requires Helm 3.7+)
helm upgrade zig-ai-aks ./deploy/aks/helm/zig-ai-aks --namespace zig-ai
```

## 🏷️ Naming Conventions

### Chart Names
- **Pattern**: `zig-ai-{platform}`
- **Examples**: `zig-ai-aks`, `zig-ai-eks`, `zig-ai-gke`

### Release Names
- **Pattern**: `zig-ai-{platform}[-{environment}]`
- **Examples**: 
  - `zig-ai-aks` (production)
  - `zig-ai-aks-staging` (staging environment)
  - `zig-ai-aks-dev` (development environment)

### Namespace Strategy
- **Production**: `zig-ai`
- **Staging**: `zig-ai-staging`
- **Development**: `zig-ai-dev`

### Resource Labels
All Kubernetes resources now include consistent labels:
```yaml
labels:
  app.kubernetes.io/name: zig-ai-aks
  app.kubernetes.io/instance: zig-ai-aks
  app.kubernetes.io/component: coordinator|shard
  app.kubernetes.io/part-of: zig-ai-platform
  app.kubernetes.io/managed-by: Helm
```

## 🔍 Verification

After migration, verify the new naming:

```bash
# Check Helm releases
helm list -n zig-ai

# Check pod labels
kubectl get pods -n zig-ai --show-labels

# Check chart information
helm show chart ./deploy/aks/helm/zig-ai-aks
```

## 📞 Support

If you encounter issues during migration:

1. **Check the logs**: `kubectl logs -n zig-ai deployment/zig-ai-coordinator`
2. **Verify chart**: `helm template ./deploy/aks/helm/zig-ai-aks`
3. **Review documentation**: [Troubleshooting Guide](./LLM_TROUBLESHOOTING_GUIDE.md)
4. **Create an issue**: [GitHub Issues](https://github.com/anachary/zig-ai-platform/issues)

---

## 🎉 Summary

The new naming convention provides:
- ✅ **Clear identification** of deployment platform (AKS)
- ✅ **Future-proof structure** for multi-cloud deployments
- ✅ **Better operational clarity** in production environments
- ✅ **Consistent documentation** across all guides
- ✅ **Improved discoverability** through better keywords

This change positions the Zig AI platform for scalable, multi-cloud deployment strategies while maintaining clarity and operational excellence.

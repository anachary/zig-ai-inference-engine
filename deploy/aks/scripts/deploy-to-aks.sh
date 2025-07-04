#!/bin/bash

# Zig AI AKS Deployment Script
# This script deploys the Zig AI Distributed Inference Engine to Azure Kubernetes Service

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="${RESOURCE_GROUP:-zig-ai-rg}"
CLUSTER_NAME="${CLUSTER_NAME:-zig-ai-aks}"
LOCATION="${LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-zigairegistry}"
NAMESPACE="${NAMESPACE:-zig-ai}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-zig-ai-aks}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check Azure login
    if ! az account show &> /dev/null; then
        log_error "Not logged into Azure. Please run 'az login' first."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

create_resource_group() {
    log_info "Creating resource group: $RESOURCE_GROUP"
    
    if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        log_warning "Resource group $RESOURCE_GROUP already exists"
    else
        az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
        log_success "Resource group created"
    fi
}

create_acr() {
    log_info "Creating Azure Container Registry: $ACR_NAME"
    
    if az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        log_warning "ACR $ACR_NAME already exists"
    else
        az acr create \
            --resource-group "$RESOURCE_GROUP" \
            --name "$ACR_NAME" \
            --sku Premium \
            --admin-enabled true
        log_success "ACR created"
    fi
}

create_aks_cluster() {
    log_info "Creating AKS cluster: $CLUSTER_NAME"
    
    if az aks show --name "$CLUSTER_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        log_warning "AKS cluster $CLUSTER_NAME already exists"
    else
        az aks create \
            --resource-group "$RESOURCE_GROUP" \
            --name "$CLUSTER_NAME" \
            --location "$LOCATION" \
            --node-count 3 \
            --node-vm-size Standard_D4s_v3 \
            --enable-addons monitoring \
            --attach-acr "$ACR_NAME" \
            --enable-cluster-autoscaler \
            --min-count 1 \
            --max-count 10 \
            --kubernetes-version 1.28.0 \
            --enable-managed-identity \
            --generate-ssh-keys
        log_success "AKS cluster created"
    fi
}

add_gpu_nodepool() {
    log_info "Adding GPU node pool to AKS cluster"
    
    if az aks nodepool show --cluster-name "$CLUSTER_NAME" --resource-group "$RESOURCE_GROUP" --name gpunodes &> /dev/null; then
        log_warning "GPU node pool already exists"
    else
        az aks nodepool add \
            --resource-group "$RESOURCE_GROUP" \
            --cluster-name "$CLUSTER_NAME" \
            --name gpunodes \
            --node-count 2 \
            --node-vm-size Standard_NC6s_v3 \
            --enable-cluster-autoscaler \
            --min-count 1 \
            --max-count 8 \
            --node-taints nvidia.com/gpu=true:NoSchedule \
            --labels accelerator=nvidia-tesla-v100
        log_success "GPU node pool added"
    fi
}

configure_kubectl() {
    log_info "Configuring kubectl"
    
    az aks get-credentials \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CLUSTER_NAME" \
        --overwrite-existing
    
    log_success "kubectl configured"
}

install_gpu_operator() {
    log_info "Installing NVIDIA GPU Operator"
    
    # Add NVIDIA Helm repository
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    # Check if GPU operator is already installed
    if helm list -n gpu-operator | grep -q gpu-operator; then
        log_warning "GPU operator already installed"
    else
        # Install GPU operator
        helm install --wait --generate-name \
            -n gpu-operator --create-namespace \
            nvidia/gpu-operator \
            --set driver.enabled=true
        log_success "GPU operator installed"
    fi
}

build_and_push_images() {
    log_info "Building and pushing Docker images"
    
    # Login to ACR
    az acr login --name "$ACR_NAME"
    
    # Get ACR login server
    ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer --output tsv)
    
    # Build coordinator image
    log_info "Building coordinator image"
    docker build -f deploy/aks/Dockerfile.coordinator -t "$ACR_LOGIN_SERVER/zig-ai/coordinator:latest" .
    docker push "$ACR_LOGIN_SERVER/zig-ai/coordinator:latest"
    
    # Build shard image
    log_info "Building shard image"
    docker build -f deploy/aks/Dockerfile.shard -t "$ACR_LOGIN_SERVER/zig-ai/shard:latest" .
    docker push "$ACR_LOGIN_SERVER/zig-ai/shard:latest"
    
    log_success "Images built and pushed"
}

create_namespace() {
    log_info "Creating Kubernetes namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace created"
    fi
}

deploy_azure_integrations() {
    log_info "Deploying Azure integrations"
    
    # Apply Azure-specific configurations
    kubectl apply -f deploy/aks/azure-integration.yaml
    
    log_success "Azure integrations deployed"
}

deploy_with_helm() {
    log_info "Deploying Zig AI with Helm"
    
    # Get ACR login server
    ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer --output tsv)
    
    # Deploy with Helm
    helm upgrade --install "$HELM_RELEASE_NAME" ./deploy/aks/helm/zig-ai-aks \
        --namespace "$NAMESPACE" \
        --create-namespace \
        --values deploy/aks/values-production.yaml \
        --set global.imageRegistry="$ACR_LOGIN_SERVER" \
        --set coordinator.replicas=1 \
        --set shard.replicas=4 \
        --set shard.resources.requests.nvidia\.com/gpu=1 \
        --set shard.resources.limits.nvidia\.com/gpu=1 \
        --wait --timeout=15m
    
    log_success "Zig AI deployed with Helm"
}

setup_monitoring() {
    log_info "Setting up monitoring"
    
    # Install Prometheus and Grafana
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    if helm list -n monitoring | grep -q prometheus; then
        log_warning "Prometheus already installed"
    else
        helm install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --create-namespace \
            --values deploy/aks/monitoring-values.yaml
        log_success "Monitoring stack installed"
    fi
}

setup_autoscaling() {
    log_info "Setting up autoscaling"
    
    # Apply HPA configuration
    kubectl apply -f deploy/aks/hpa.yaml
    
    log_success "Autoscaling configured"
}

verify_deployment() {
    log_info "Verifying deployment"
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=zig-ai-coordinator -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=zig-ai-shard -n "$NAMESPACE" --timeout=300s
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE"
    
    # Get service information
    kubectl get services -n "$NAMESPACE"
    
    # Get external IP
    EXTERNAL_IP=$(kubectl get service zig-ai-coordinator-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -n "$EXTERNAL_IP" ]; then
        log_success "Deployment successful! Access the service at: http://$EXTERNAL_IP"
        
        # Test health endpoint
        log_info "Testing health endpoint..."
        if curl -f "http://$EXTERNAL_IP/api/v1/health" &> /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed - service may still be starting"
        fi
    else
        log_warning "External IP not yet assigned. Check service status with: kubectl get service -n $NAMESPACE"
    fi
}

cleanup() {
    log_info "Cleaning up temporary files"
    # Add any cleanup tasks here
}

main() {
    log_info "Starting Zig AI AKS deployment"
    
    check_prerequisites
    create_resource_group
    create_acr
    create_aks_cluster
    add_gpu_nodepool
    configure_kubectl
    install_gpu_operator
    build_and_push_images
    create_namespace
    deploy_azure_integrations
    deploy_with_helm
    setup_monitoring
    setup_autoscaling
    verify_deployment
    cleanup
    
    log_success "Zig AI deployment to AKS completed successfully!"
    
    echo ""
    echo "Next steps:"
    echo "1. Access the service using the external IP shown above"
    echo "2. Monitor the deployment: kubectl get pods -n $NAMESPACE -w"
    echo "3. View logs: kubectl logs -f deployment/zig-ai-coordinator -n $NAMESPACE"
    echo "4. Scale shards: kubectl scale deployment zig-ai-shard --replicas=8 -n $NAMESPACE"
    echo "5. Access Grafana dashboard for monitoring"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        log_info "Cleaning up AKS deployment"
        helm uninstall "$HELM_RELEASE_NAME" -n "$NAMESPACE" || true
        kubectl delete namespace "$NAMESPACE" || true
        az aks delete --name "$CLUSTER_NAME" --resource-group "$RESOURCE_GROUP" --yes --no-wait
        log_success "Cleanup initiated"
        ;;
    "status")
        kubectl get all -n "$NAMESPACE"
        ;;
    *)
        echo "Usage: $0 [deploy|cleanup|status]"
        exit 1
        ;;
esac

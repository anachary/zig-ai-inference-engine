# IoT Demo Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Zig AI Platform IoT demonstration on various environments, from local development to production edge deployments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Real Raspberry Pi Deployment](#real-raspberry-pi-deployment)
5. [Cloud Edge Deployment](#cloud-edge-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Network**: Broadband internet connection

#### Software Dependencies
- **Docker Desktop**: Latest version
- **PowerShell**: 5.1+ (Windows) or PowerShell Core 7+ (cross-platform)
- **Git**: Latest version
- **Python**: 3.8+ (for real Pi deployment)

#### Hardware for Real Deployment
- **Raspberry Pi 4**: 4GB or 8GB RAM recommended
- **MicroSD Card**: 32GB Class 10 or better
- **Power Supply**: Official Raspberry Pi 4 power supply
- **Network**: Ethernet or WiFi connectivity

## Local Development Setup

### Quick Start

1. **Clone the Repository**
   ```powershell
   git clone https://github.com/anachary/zig-ai-platform.git
   cd zig-ai-platform/iot-demo
   ```

2. **Run Setup Script**
   ```powershell
   .\setup.ps1
   ```

3. **Start the Demo**
   ```powershell
   .\start-iot-demo.ps1
   ```

4. **Run Test Scenarios**
   ```powershell
   .\run-scenarios.ps1
   ```

### Detailed Setup Process

#### Step 1: Environment Preparation

```powershell
# Check prerequisites
.\setup.ps1 -Verbose

# Quick setup (skips model downloads)
.\setup.ps1 -QuickSetup

# Setup without Docker (for testing)
.\setup.ps1 -SkipDocker
```

#### Step 2: Configuration Customization

Edit configuration files in the `config/` directory:

**Device Configuration** (`config/pi-devices.yaml`):
```yaml
devices:
  - name: "custom-pi"
    type: "raspberry-pi-4"
    memory_mb: 8192
    cpu_cores: 4
    architecture: "arm64"
    network:
      type: "ethernet"
      bandwidth_mbps: 100
      latency_ms: 5
    location: "custom-location"
    use_case: "custom-application"
```

**Model Configuration** (`config/models.yaml`):
```yaml
models:
  - name: "custom-model"
    parameters: 1500000000
    quantization: "int8"
    memory_requirement_mb: 1500
    use_cases: ["custom-application"]
    inference_time_ms: 2500
```

#### Step 3: Advanced Startup Options

```powershell
# Start with specific scenario
.\start-iot-demo.ps1 -Scenario "smart-home"

# Start with verbose logging
.\start-iot-demo.ps1 -Verbose

# Rebuild containers
.\start-iot-demo.ps1 -Rebuild

# Extended timeout for slow systems
.\start-iot-demo.ps1 -Timeout 600
```

## Production Deployment

### Docker Compose Production Setup

1. **Create Production Configuration**
   ```yaml
   # docker/docker-compose.prod.yml
   version: '3.8'
   
   services:
     edge-coordinator:
       build:
         context: .
         dockerfile: docker/Dockerfile.edge-coordinator
       ports:
         - "8080:8080"
       environment:
         - LOG_LEVEL=WARN
         - COORDINATOR_PORT=8080
         - PRODUCTION=true
       volumes:
         - ./config:/app/config:ro
         - ./logs:/app/logs
       restart: unless-stopped
       deploy:
         resources:
           limits:
             memory: 1G
             cpus: '0.5'
   
     smart-home-pi:
       build:
         context: .
         dockerfile: docker/Dockerfile.pi-simulator
       environment:
         - DEVICE_NAME=smart-home-pi
         - DEVICE_PORT=8081
         - COORDINATOR_URL=http://edge-coordinator:8080
         - PRODUCTION=true
       volumes:
         - ./config:/app/config:ro
         - ./models:/app/models:ro
       restart: unless-stopped
       deploy:
         resources:
           limits:
             memory: 2G
             cpus: '1.0'
   ```

2. **Deploy to Production**
   ```bash
   # Deploy with production configuration
   docker-compose -f docker/docker-compose.prod.yml up -d
   
   # Scale specific services
   docker-compose -f docker/docker-compose.prod.yml up -d --scale smart-home-pi=3
   ```

### Kubernetes Deployment

1. **Create Kubernetes Manifests**
   ```yaml
   # k8s/edge-coordinator-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: edge-coordinator
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: edge-coordinator
     template:
       metadata:
         labels:
           app: edge-coordinator
       spec:
         containers:
         - name: edge-coordinator
           image: zig-ai-platform/edge-coordinator:latest
           ports:
           - containerPort: 8080
           env:
           - name: LOG_LEVEL
             value: "INFO"
           resources:
             requests:
               memory: "512Mi"
               cpu: "250m"
             limits:
               memory: "1Gi"
               cpu: "500m"
   ```

2. **Deploy to Kubernetes**
   ```bash
   # Apply manifests
   kubectl apply -f k8s/
   
   # Check deployment status
   kubectl get pods -l app=edge-coordinator
   
   # Scale deployment
   kubectl scale deployment edge-coordinator --replicas=3
   ```

## Real Raspberry Pi Deployment

### Raspberry Pi OS Setup

1. **Prepare Raspberry Pi OS**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install dependencies
   sudo apt install -y python3 python3-pip git docker.io
   
   # Add user to docker group
   sudo usermod -aG docker $USER
   ```

2. **Install Zig on Raspberry Pi**
   ```bash
   # Download Zig for ARM64
   wget https://ziglang.org/download/0.11.0/zig-linux-aarch64-0.11.0.tar.xz
   tar -xf zig-linux-aarch64-0.11.0.tar.xz
   sudo mv zig-linux-aarch64-0.11.0 /opt/zig
   sudo ln -s /opt/zig/zig /usr/local/bin/zig
   ```

3. **Deploy Application**
   ```bash
   # Clone repository
   git clone https://github.com/anachary/zig-ai-platform.git
   cd zig-ai-platform/iot-demo
   
   # Install Python dependencies
   pip3 install -r requirements.txt
   
   # Configure for real hardware
   cp config/pi-devices.yaml.example config/pi-devices.yaml
   # Edit configuration for your specific Pi
   
   # Start the Pi simulator (now running on real hardware)
   python3 src/pi-simulator/main.py
   ```

### Hardware-Specific Optimizations

1. **Memory Optimization**
   ```bash
   # Increase GPU memory split for AI workloads
   echo "gpu_mem=128" | sudo tee -a /boot/config.txt
   
   # Enable memory cgroup
   echo "cgroup_enable=memory cgroup_memory=1" | sudo tee -a /boot/cmdline.txt
   
   # Reboot to apply changes
   sudo reboot
   ```

2. **Performance Tuning**
   ```bash
   # Set CPU governor to performance
   echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   
   # Increase swap for large models
   sudo dphys-swapfile swapoff
   sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

### Multi-Pi Cluster Setup

1. **Configure Master Pi (Edge Coordinator)**
   ```bash
   # Set up as coordinator
   export DEVICE_ROLE=coordinator
   export COORDINATOR_PORT=8080
   
   # Start coordinator
   zig run src/edge-coordinator/main.zig
   ```

2. **Configure Worker Pis**
   ```bash
   # Set up as worker
   export DEVICE_ROLE=worker
   export DEVICE_NAME=pi-worker-01
   export COORDINATOR_URL=http://192.168.1.100:8080
   
   # Start worker
   python3 src/pi-simulator/main.py
   ```

## Cloud Edge Deployment

### AWS IoT Greengrass

1. **Setup Greengrass Core**
   ```bash
   # Install Greengrass Core
   curl -s https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip > greengrass-nucleus-latest.zip
   unzip greengrass-nucleus-latest.zip -d GreengrassCore
   
   # Configure and start
   sudo java -Droot="/greengrass/v2" -Dlog.store=FILE \
     -jar ./GreengrassCore/lib/Greengrass.jar \
     --aws-region us-west-2 \
     --thing-name MyGreengrassCore \
     --thing-group-name MyGreengrassCoreGroup \
     --component-default-user ggc_user:ggc_group \
     --provision true \
     --setup-system-service true
   ```

2. **Deploy Zig AI Component**
   ```json
   {
     "ComponentName": "com.zigai.iot.inference",
     "ComponentVersion": "1.0.0",
     "ComponentDescription": "Zig AI Platform IoT Inference Component",
     "ComponentPublisher": "ZigAI",
     "ComponentConfiguration": {
       "DefaultConfiguration": {
         "device_name": "greengrass-pi",
         "coordinator_url": "http://localhost:8080",
         "model_path": "/opt/zigai/models"
       }
     },
     "Manifests": [
       {
         "Platform": {
           "os": "linux",
           "architecture": "aarch64"
         },
         "Lifecycle": {
           "Install": "pip3 install -r requirements.txt",
           "Run": "python3 src/pi-simulator/main.py"
         }
       }
     ]
   }
   ```

### Azure IoT Edge

1. **Install IoT Edge Runtime**
   ```bash
   # Install Microsoft package repository
   curl https://packages.microsoft.com/config/ubuntu/18.04/multiarch/packages-microsoft-prod.deb > ./packages-microsoft-prod.deb
   sudo dpkg -i ./packages-microsoft-prod.deb
   
   # Install IoT Edge
   sudo apt-get update
   sudo apt-get install aziot-edge
   ```

2. **Configure IoT Edge Module**
   ```json
   {
     "modulesContent": {
       "$edgeAgent": {
         "properties.desired": {
           "modules": {
             "zigai-inference": {
               "version": "1.0",
               "type": "docker",
               "status": "running",
               "restartPolicy": "always",
               "settings": {
                 "image": "zigai/iot-inference:latest",
                 "createOptions": {
                   "ExposedPorts": {
                     "8081/tcp": {}
                   },
                   "HostConfig": {
                     "PortBindings": {
                       "8081/tcp": [{"HostPort": "8081"}]
                     }
                   }
                 }
               },
               "env": {
                 "DEVICE_NAME": {"value": "azure-edge-pi"},
                 "COORDINATOR_URL": {"value": "http://edgeHub:8080"}
               }
             }
           }
         }
       }
     }
   }
   ```

## Monitoring and Maintenance

### Health Monitoring

1. **Automated Health Checks**
   ```powershell
   # Create monitoring script
   # monitor-health.ps1
   while ($true) {
       $health = Invoke-RestMethod "http://localhost:8080/health"
       if ($health.status -ne "healthy") {
           Send-MailMessage -To "admin@company.com" -Subject "IoT Demo Health Alert"
       }
       Start-Sleep 300  # Check every 5 minutes
   }
   ```

2. **Prometheus Metrics**
   ```yaml
   # prometheus.yml
   global:
     scrape_interval: 15s
   
   scrape_configs:
     - job_name: 'iot-demo'
       static_configs:
         - targets: ['localhost:8080', 'localhost:8081', 'localhost:8082', 'localhost:8083']
   ```

### Log Management

1. **Centralized Logging**
   ```yaml
   # docker-compose.logging.yml
   version: '3.8'
   
   services:
     fluentd:
       image: fluent/fluentd:v1.14-1
       volumes:
         - ./fluentd/conf:/fluentd/etc
         - ./logs:/var/log
       ports:
         - "24224:24224"
   
     elasticsearch:
       image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
       environment:
         - discovery.type=single-node
   
     kibana:
       image: docker.elastic.co/kibana/kibana:7.15.0
       ports:
         - "5601:5601"
   ```

### Backup and Recovery

1. **Data Backup**
   ```powershell
   # backup-data.ps1
   $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
   $backupPath = "backups/iot-demo-$timestamp"
   
   New-Item -ItemType Directory -Path $backupPath -Force
   Copy-Item -Path "config/*" -Destination "$backupPath/config/" -Recurse
   Copy-Item -Path "data/*" -Destination "$backupPath/data/" -Recurse
   Copy-Item -Path "logs/*" -Destination "$backupPath/logs/" -Recurse
   
   Compress-Archive -Path $backupPath -DestinationPath "$backupPath.zip"
   ```

2. **Disaster Recovery**
   ```powershell
   # restore-backup.ps1
   param([string]$BackupFile)
   
   Expand-Archive -Path $BackupFile -DestinationPath "temp-restore"
   Copy-Item -Path "temp-restore/config/*" -Destination "config/" -Recurse -Force
   Copy-Item -Path "temp-restore/data/*" -Destination "data/" -Recurse -Force
   
   .\start-iot-demo.ps1
   ```

## Troubleshooting

### Common Issues

1. **Docker Container Won't Start**
   ```powershell
   # Check Docker status
   docker ps -a
   
   # View container logs
   docker logs iot-demo_smart-home-pi_1
   
   # Restart specific container
   docker-compose restart smart-home-pi
   ```

2. **Port Conflicts**
   ```powershell
   # Check what's using the port
   netstat -ano | findstr :8080
   
   # Kill process using port
   taskkill /PID <process_id> /F
   ```

3. **Memory Issues**
   ```powershell
   # Check system memory
   Get-CimInstance -ClassName Win32_ComputerSystem | Select-Object TotalPhysicalMemory
   
   # Check Docker memory usage
   docker stats
   ```

### Performance Optimization

1. **Resource Allocation**
   ```yaml
   # Optimize Docker Compose resources
   services:
     smart-home-pi:
       deploy:
         resources:
           limits:
             cpus: '1.0'
             memory: 2G
           reservations:
             cpus: '0.5'
             memory: 1G
   ```

2. **Network Optimization**
   ```bash
   # Optimize network settings
   echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
   echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
   sysctl -p
   ```

For additional support, see the [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on GitHub.

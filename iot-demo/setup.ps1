# Zig AI Platform IoT Demo Setup Script
# This script sets up the complete IoT demonstration environment

param(
    [switch]$SkipDocker,
    [switch]$QuickSetup,
    [string]$LogLevel = "INFO"
)

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"

# Colors for output
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Header = "Magenta"
}

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput "=" * 60 -Color "Header"
    Write-ColorOutput "  $Title" -Color "Header"
    Write-ColorOutput "=" * 60 -Color "Header"
    Write-Host ""
}

function Test-Prerequisites {
    Write-Header "Checking Prerequisites"
    
    $prerequisites = @()
    
    # Check PowerShell version
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        $prerequisites += "PowerShell 5.1 or higher required"
    } else {
        Write-ColorOutput "✓ PowerShell $($PSVersionTable.PSVersion) detected" -Color "Success"
    }
    
    # Check Docker
    if (-not $SkipDocker) {
        try {
            $dockerVersion = docker --version 2>$null
            if ($dockerVersion) {
                Write-ColorOutput "✓ Docker detected: $dockerVersion" -Color "Success"
            } else {
                $prerequisites += "Docker Desktop not found or not running"
            }
        } catch {
            $prerequisites += "Docker Desktop not found or not running"
        }
    }
    
    # Check Git
    try {
        $gitVersion = git --version 2>$null
        if ($gitVersion) {
            Write-ColorOutput "✓ Git detected: $gitVersion" -Color "Success"
        } else {
            $prerequisites += "Git not found in PATH"
        }
    } catch {
        $prerequisites += "Git not found in PATH"
    }
    
    # Check available memory
    $memory = Get-CimInstance -ClassName Win32_ComputerSystem
    $totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 1)
    if ($totalMemoryGB -lt 8) {
        $prerequisites += "At least 8GB RAM recommended (detected: $totalMemoryGB GB)"
    } else {
        Write-ColorOutput "✓ Memory: $totalMemoryGB GB available" -Color "Success"
    }
    
    # Check disk space
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
    $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 1)
    if ($freeSpaceGB -lt 10) {
        $prerequisites += "At least 10GB free disk space required (available: $freeSpaceGB GB)"
    } else {
        Write-ColorOutput "✓ Disk space: $freeSpaceGB GB available" -Color "Success"
    }
    
    if ($prerequisites.Count -gt 0) {
        Write-ColorOutput "❌ Prerequisites not met:" -Color "Error"
        foreach ($req in $prerequisites) {
            Write-ColorOutput "   - $req" -Color "Error"
        }
        exit 1
    }
    
    Write-ColorOutput "✓ All prerequisites met!" -Color "Success"
}

function Initialize-Directories {
    Write-Header "Creating Directory Structure"
    
    $directories = @(
        "config",
        "src/pi-simulator",
        "src/edge-coordinator", 
        "src/iot-client",
        "src/monitoring",
        "docker",
        "models/lightweight-llm-1b",
        "models/industrial-ai-3b", 
        "models/retail-ai-2b",
        "scripts",
        "docs",
        "logs",
        "data/sensor-data",
        "data/inference-results",
        "web/dashboard"
    )
    
    foreach ($dir in $directories) {
        $fullPath = Join-Path $PSScriptRoot $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
            Write-ColorOutput "✓ Created directory: $dir" -Color "Success"
        } else {
            Write-ColorOutput "✓ Directory exists: $dir" -Color "Info"
        }
    }
}

function Create-ConfigurationFiles {
    Write-Header "Creating Configuration Files"
    
    # Pi devices configuration
    $piDevicesConfig = @"
devices:
  - name: "smart-home-pi"
    type: "raspberry-pi-4"
    memory_mb: 4096
    cpu_cores: 4
    architecture: "arm64"
    network:
      type: "wifi"
      bandwidth_mbps: 50
      latency_ms: 20
    location: "living-room"
    use_case: "smart-home"
    
  - name: "industrial-pi"
    type: "raspberry-pi-4"
    memory_mb: 8192
    cpu_cores: 4
    architecture: "arm64"
    network:
      type: "ethernet"
      bandwidth_mbps: 100
      latency_ms: 5
    location: "factory-floor"
    use_case: "industrial-iot"
    
  - name: "retail-pi"
    type: "raspberry-pi-4"
    memory_mb: 4096
    cpu_cores: 4
    architecture: "arm64"
    network:
      type: "4g"
      bandwidth_mbps: 25
      latency_ms: 50
    location: "store-front"
    use_case: "retail-edge"

coordinator:
  port: 8080
  max_devices: 10
  health_check_interval: 30
  load_balancing_strategy: "least_loaded"
"@
    
    $piDevicesConfig | Out-File -FilePath "config/pi-devices.yaml" -Encoding UTF8
    Write-ColorOutput "✓ Created pi-devices.yaml" -Color "Success"
    
    # Models configuration
    $modelsConfig = @"
models:
  - name: "lightweight-llm-1b"
    parameters: 1000000000
    quantization: "int8"
    memory_requirement_mb: 1200
    use_cases: ["smart-home", "general-chat"]
    inference_time_ms: 3200
    
  - name: "industrial-ai-3b"
    parameters: 3000000000
    quantization: "int4"
    memory_requirement_mb: 1800
    use_cases: ["industrial-iot", "predictive-maintenance"]
    inference_time_ms: 2800
    
  - name: "retail-ai-2b"
    parameters: 2000000000
    quantization: "int8"
    memory_requirement_mb: 2100
    use_cases: ["retail-edge", "customer-service"]
    inference_time_ms: 4100

optimization:
  enable_model_caching: true
  cache_size_mb: 1024
  enable_dynamic_loading: true
  prefetch_popular_models: true
"@
    
    $modelsConfig | Out-File -FilePath "config/models.yaml" -Encoding UTF8
    Write-ColorOutput "✓ Created models.yaml" -Color "Success"
    
    # Scenarios configuration
    $scenariosConfig = @"
scenarios:
  smart_home:
    name: "Smart Home Assistant"
    description: "Voice commands and home automation"
    device: "smart-home-pi"
    model: "lightweight-llm-1b"
    test_queries:
      - "Turn on the living room lights"
      - "What's the weather like today?"
      - "Set a timer for 10 minutes"
      - "Play some relaxing music"
    expected_response_time_ms: 3000
    
  industrial_monitoring:
    name: "Industrial Predictive Maintenance"
    description: "Sensor data analysis and anomaly detection"
    device: "industrial-pi"
    model: "industrial-ai-3b"
    test_queries:
      - "Analyze vibration data: 12.5Hz, 0.8mm amplitude"
      - "Temperature reading: 85°C, normal range?"
      - "Pressure sensor shows 45 PSI, evaluate"
      - "Motor current: 15.2A, predict maintenance needs"
    expected_response_time_ms: 2500
    
  retail_assistant:
    name: "Retail Customer Service"
    description: "Customer queries and inventory assistance"
    device: "retail-pi"
    model: "retail-ai-2b"
    test_queries:
      - "Where can I find organic vegetables?"
      - "Any discounts on electronics today?"
      - "Do you have this item in stock?"
      - "What are your store hours?"
    expected_response_time_ms: 4000

load_testing:
  concurrent_requests: [1, 5, 10, 20]
  duration_minutes: 5
  ramp_up_seconds: 30
"@
    
    $scenariosConfig | Out-File -FilePath "config/scenarios.yaml" -Encoding UTF8
    Write-ColorOutput "✓ Created scenarios.yaml" -Color "Success"
}

function Setup-DockerEnvironment {
    if ($SkipDocker) {
        Write-ColorOutput "⚠ Skipping Docker setup as requested" -Color "Warning"
        return
    }
    
    Write-Header "Setting Up Docker Environment"
    
    # Create Dockerfile for Pi Simulator
    $piSimulatorDockerfile = @"
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    python3 \
    python3-pip \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Zig
RUN wget https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz \
    && tar -xf zig-linux-x86_64-0.11.0.tar.xz \
    && mv zig-linux-x86_64-0.11.0 /opt/zig \
    && ln -s /opt/zig/zig /usr/local/bin/zig

# Set up working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Install Python dependencies for simulation
RUN pip3 install fastapi uvicorn pyyaml psutil

# Expose ports
EXPOSE 8081 8082 8083

# Start script
CMD ["python3", "src/pi-simulator/main.py"]
"@

    $piSimulatorDockerfile | Out-File -FilePath "docker/Dockerfile.pi-simulator" -Encoding UTF8
    Write-ColorOutput "✓ Created Pi Simulator Dockerfile" -Color "Success"

    # Create Dockerfile for Edge Coordinator
    $edgeCoordinatorDockerfile = @"
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Zig
RUN wget https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz \
    && tar -xf zig-linux-x86_64-0.11.0.tar.xz \
    && mv zig-linux-x86_64-0.11.0 /opt/zig \
    && ln -s /opt/zig/zig /usr/local/bin/zig

# Set up working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Build the edge coordinator
RUN cd src/edge-coordinator && zig build-exe main.zig

# Expose port
EXPOSE 8080

# Start script
CMD ["./src/edge-coordinator/main"]
"@

    $edgeCoordinatorDockerfile | Out-File -FilePath "docker/Dockerfile.edge-coordinator" -Encoding UTF8
    Write-ColorOutput "✓ Created Edge Coordinator Dockerfile" -Color "Success"
    
    # Create Docker Compose file
    $dockerCompose = @"
version: '3.8'

services:
  edge-coordinator:
    build:
      context: .
      dockerfile: docker/Dockerfile.edge-coordinator
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=INFO
      - COORDINATOR_PORT=8080
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - iot-network

  smart-home-pi:
    build:
      context: .
      dockerfile: docker/Dockerfile.pi-simulator
    ports:
      - "8081:8081"
    environment:
      - DEVICE_NAME=smart-home-pi
      - DEVICE_PORT=8081
      - COORDINATOR_URL=http://edge-coordinator:8080
    volumes:
      - ./config:/app/config
      - ./models:/app/models
    networks:
      - iot-network
    depends_on:
      - edge-coordinator

  industrial-pi:
    build:
      context: .
      dockerfile: docker/Dockerfile.pi-simulator
    ports:
      - "8082:8082"
    environment:
      - DEVICE_NAME=industrial-pi
      - DEVICE_PORT=8082
      - COORDINATOR_URL=http://edge-coordinator:8080
    volumes:
      - ./config:/app/config
      - ./models:/app/models
    networks:
      - iot-network
    depends_on:
      - edge-coordinator

  retail-pi:
    build:
      context: .
      dockerfile: docker/Dockerfile.pi-simulator
    ports:
      - "8083:8083"
    environment:
      - DEVICE_NAME=retail-pi
      - DEVICE_PORT=8083
      - COORDINATOR_URL=http://edge-coordinator:8080
    volumes:
      - ./config:/app/config
      - ./models:/app/models
    networks:
      - iot-network
    depends_on:
      - edge-coordinator

  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    networks:
      - iot-network

networks:
  iot-network:
    driver: bridge

volumes:
  grafana-storage:
"@
    
    $dockerCompose | Out-File -FilePath "docker/docker-compose.yml" -Encoding UTF8
    Write-ColorOutput "✓ Created Docker Compose configuration" -Color "Success"
}

function Download-Models {
    if ($QuickSetup) {
        Write-ColorOutput "⚠ Skipping model download in quick setup mode" -Color "Warning"
        return
    }
    
    Write-Header "Downloading AI Models"
    
    # Create placeholder model files for demo
    $modelDirs = @("lightweight-llm-1b", "industrial-ai-3b", "retail-ai-2b")
    
    foreach ($modelDir in $modelDirs) {
        $modelPath = "models/$modelDir"
        
        # Create model metadata
        $metadata = @"
{
    "name": "$modelDir",
    "version": "1.0.0",
    "architecture": "transformer",
    "quantization": "int8",
    "created": "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')",
    "size_mb": 1200,
    "description": "Optimized model for IoT edge deployment"
}
"@
        $metadata | Out-File -FilePath "$modelPath/metadata.json" -Encoding UTF8
        
        # Create dummy model file
        $dummyModel = "# Placeholder model file for $modelDir`n# In production, this would contain the actual model weights"
        $dummyModel | Out-File -FilePath "$modelPath/model.zig" -Encoding UTF8
        
        Write-ColorOutput "✓ Created placeholder model: $modelDir" -Color "Success"
    }
}

function Create-Scripts {
    Write-Header "Creating Utility Scripts"
    
    # Create start script
    $startScript = @"
# Start IoT Demo Script
Write-Host "Starting Zig AI Platform IoT Demo..." -ForegroundColor Green

# Check if Docker is running
try {
    docker ps | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Start the demo environment
Write-Host "Starting containers..." -ForegroundColor Cyan
Set-Location docker
docker-compose up -d

Write-Host ""
Write-Host "🚀 IoT Demo is starting up!" -ForegroundColor Green
Write-Host "📊 Dashboard: http://localhost:8080" -ForegroundColor Cyan
Write-Host "📈 Monitoring: http://localhost:3000 (admin/admin)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run './run-scenarios.ps1' to start test scenarios" -ForegroundColor Yellow
"@
    
    $startScript | Out-File -FilePath "start-iot-demo.ps1" -Encoding UTF8
    Write-ColorOutput "✓ Created start-iot-demo.ps1" -Color "Success"
    
    # Create monitoring script
    $monitorScript = @"
# IoT Demo Monitoring Script
Write-Host "IoT Demo Monitoring Dashboard" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

while (`$true) {
    Clear-Host
    Write-Host "Zig AI Platform IoT Demo - Live Monitoring" -ForegroundColor Green
    Write-Host "Time: $(Get-Date)" -ForegroundColor Cyan
    Write-Host ""
    
    # Check container status
    Write-Host "Container Status:" -ForegroundColor Yellow
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    Write-Host ""
    Write-Host "Press Ctrl+C to exit monitoring" -ForegroundColor Gray
    Start-Sleep -Seconds 5
}
"@
    
    $monitorScript | Out-File -FilePath "monitor-iot.ps1" -Encoding UTF8
    Write-ColorOutput "✓ Created monitor-iot.ps1" -Color "Success"
    
    # Create cleanup script
    $cleanupScript = @"
# Cleanup IoT Demo Script
Write-Host "Cleaning up IoT Demo environment..." -ForegroundColor Yellow

Set-Location docker
docker-compose down -v
docker system prune -f

Write-Host "✓ Cleanup completed" -ForegroundColor Green
"@
    
    $cleanupScript | Out-File -FilePath "cleanup.ps1" -Encoding UTF8
    Write-ColorOutput "✓ Created cleanup.ps1" -Color "Success"
}

function Show-Summary {
    Write-Header "Setup Complete!"
    
    Write-ColorOutput "🎉 IoT Demo environment has been set up successfully!" -Color "Success"
    Write-Host ""
    Write-ColorOutput "Next steps:" -Color "Info"
    Write-ColorOutput "1. Start the demo: .\start-iot-demo.ps1" -Color "Info"
    Write-ColorOutput "2. Run test scenarios: .\run-scenarios.ps1" -Color "Info"
    Write-ColorOutput "3. Monitor performance: .\monitor-iot.ps1" -Color "Info"
    Write-Host ""
    Write-ColorOutput "Access points:" -Color "Info"
    Write-ColorOutput "• Main Dashboard: http://localhost:8080" -Color "Info"
    Write-ColorOutput "• Monitoring: http://localhost:3000 (admin/admin)" -Color "Info"
    Write-ColorOutput "• Smart Home Pi: http://localhost:8081" -Color "Info"
    Write-ColorOutput "• Industrial Pi: http://localhost:8082" -Color "Info"
    Write-ColorOutput "• Retail Pi: http://localhost:8083" -Color "Info"
    Write-Host ""
    Write-ColorOutput "For help: Get-Help .\setup.ps1 -Full" -Color "Info"
}

# Main execution
try {
    Write-Header "Zig AI Platform IoT Demo Setup"
    
    Test-Prerequisites
    Initialize-Directories
    Create-ConfigurationFiles
    Setup-DockerEnvironment
    Download-Models
    Create-Scripts
    Show-Summary
    
} catch {
    Write-ColorOutput "❌ Setup failed: $($_.Exception.Message)" -Color "Error"
    Write-ColorOutput "Stack trace: $($_.ScriptStackTrace)" -Color "Error"
    exit 1
}

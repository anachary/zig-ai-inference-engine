# Zig AI Platform IoT Demo Launcher
# This script starts the complete IoT demonstration environment

param(
    [switch]$Rebuild,
    [switch]$Verbose,
    [string]$Scenario = "all",
    [int]$Timeout = 300
)

$ErrorActionPreference = "Stop"

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

function Test-DockerStatus {
    Write-Header "Checking Docker Status"
    
    try {
        $dockerInfo = docker info 2>$null
        if ($dockerInfo) {
            Write-ColorOutput "✓ Docker is running" -Color "Success"
        } else {
            throw "Docker not responding"
        }
    } catch {
        Write-ColorOutput "❌ Docker is not running. Please start Docker Desktop." -Color "Error"
        Write-ColorOutput "   After starting Docker, wait a few minutes and try again." -Color "Info"
        exit 1
    }
    
    # Check available resources
    $dockerStats = docker system df 2>$null
    if ($dockerStats) {
        Write-ColorOutput "✓ Docker system resources available" -Color "Success"
    }
}

function Start-IoTEnvironment {
    Write-Header "Starting IoT Demo Environment"
    
    # Change to docker directory
    $dockerPath = Join-Path $PSScriptRoot "docker"
    if (-not (Test-Path $dockerPath)) {
        Write-ColorOutput "❌ Docker directory not found. Please run setup.ps1 first." -Color "Error"
        exit 1
    }
    
    Set-Location $dockerPath
    
    # Stop any existing containers
    Write-ColorOutput "Stopping any existing containers..." -Color "Info"
    docker-compose down 2>$null
    
    # Rebuild if requested
    if ($Rebuild) {
        Write-ColorOutput "Rebuilding containers..." -Color "Info"
        docker-compose build --no-cache
    }
    
    # Start the environment
    Write-ColorOutput "Starting IoT demo containers..." -Color "Info"
    
    if ($Verbose) {
        docker-compose up -d
    } else {
        docker-compose up -d 2>$null
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "❌ Failed to start containers" -Color "Error"
        exit 1
    }
    
    Write-ColorOutput "✓ Containers started successfully" -Color "Success"
}

function Wait-ForServices {
    Write-Header "Waiting for Services to Initialize"
    
    $services = @(
        @{ Name = "Edge Coordinator"; Url = "http://localhost:8080/health"; Port = 8080 },
        @{ Name = "Smart Home Pi"; Url = "http://localhost:8081/health"; Port = 8081 },
        @{ Name = "Industrial Pi"; Url = "http://localhost:8082/health"; Port = 8082 },
        @{ Name = "Retail Pi"; Url = "http://localhost:8083/health"; Port = 8083 }
    )
    
    $maxWaitTime = $Timeout
    $waitInterval = 5
    $elapsed = 0
    
    while ($elapsed -lt $maxWaitTime) {
        $allReady = $true
        
        foreach ($service in $services) {
            try {
                $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 3 -UseBasicParsing 2>$null
                if ($response.StatusCode -eq 200) {
                    Write-ColorOutput "✓ $($service.Name) is ready" -Color "Success"
                } else {
                    $allReady = $false
                }
            } catch {
                Write-ColorOutput "⏳ Waiting for $($service.Name)..." -Color "Info"
                $allReady = $false
            }
        }
        
        if ($allReady) {
            Write-ColorOutput "🎉 All services are ready!" -Color "Success"
            return
        }
        
        Start-Sleep -Seconds $waitInterval
        $elapsed += $waitInterval
        
        if ($elapsed % 30 -eq 0) {
            Write-ColorOutput "Still waiting... ($elapsed/$maxWaitTime seconds)" -Color "Warning"
        }
    }
    
    Write-ColorOutput "⚠ Timeout waiting for services. Some may still be starting up." -Color "Warning"
    Write-ColorOutput "You can check status manually or wait a bit longer." -Color "Info"
}

function Show-ServiceStatus {
    Write-Header "Service Status"
    
    # Show container status
    Write-ColorOutput "Container Status:" -Color "Info"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | ForEach-Object {
        if ($_ -match "iot-demo") {
            Write-ColorOutput $_ -Color "Success"
        } else {
            Write-Host $_
        }
    }
    
    Write-Host ""
    
    # Show service endpoints
    Write-ColorOutput "Service Endpoints:" -Color "Info"
    $endpoints = @(
        "📊 Main Dashboard: http://localhost:8080",
        "📈 Monitoring (Grafana): http://localhost:3000 (admin/admin)",
        "🏠 Smart Home Pi: http://localhost:8081",
        "🏭 Industrial Pi: http://localhost:8082", 
        "🛒 Retail Pi: http://localhost:8083"
    )
    
    foreach ($endpoint in $endpoints) {
        Write-ColorOutput "   $endpoint" -Color "Cyan"
    }
}

function Test-BasicFunctionality {
    Write-Header "Testing Basic Functionality"
    
    # Test edge coordinator
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8080/api/status" -TimeoutSec 10
        Write-ColorOutput "✓ Edge Coordinator API responding" -Color "Success"
    } catch {
        Write-ColorOutput "⚠ Edge Coordinator API not responding" -Color "Warning"
    }
    
    # Test Pi devices
    $piDevices = @(
        @{ Name = "Smart Home Pi"; Port = 8081 },
        @{ Name = "Industrial Pi"; Port = 8082 },
        @{ Name = "Retail Pi"; Port = 8083 }
    )
    
    foreach ($device in $piDevices) {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:$($device.Port)/api/device-info" -TimeoutSec 5
            Write-ColorOutput "✓ $($device.Name) responding" -Color "Success"
        } catch {
            Write-ColorOutput "⚠ $($device.Name) not responding" -Color "Warning"
        }
    }
}

function Show-NextSteps {
    Write-Header "Demo Ready!"
    
    Write-ColorOutput "🚀 Your IoT Demo is now running!" -Color "Success"
    Write-Host ""
    
    Write-ColorOutput "What you can do now:" -Color "Info"
    Write-ColorOutput "1. 🧪 Run test scenarios:" -Color "Info"
    Write-ColorOutput "   .\run-scenarios.ps1" -Color "Cyan"
    Write-Host ""
    
    Write-ColorOutput "2. 📊 Monitor performance:" -Color "Info"
    Write-ColorOutput "   .\monitor-iot.ps1" -Color "Cyan"
    Write-Host ""
    
    Write-ColorOutput "3. 🌐 Access web interfaces:" -Color "Info"
    Write-ColorOutput "   • Dashboard: http://localhost:8080" -Color "Cyan"
    Write-ColorOutput "   • Monitoring: http://localhost:3000" -Color "Cyan"
    Write-Host ""
    
    Write-ColorOutput "4. 🔧 Test individual devices:" -Color "Info"
    Write-ColorOutput "   curl http://localhost:8081/api/inference -d '{\"query\":\"Hello\"}'" -Color "Cyan"
    Write-Host ""
    
    Write-ColorOutput "5. 🛑 Stop the demo:" -Color "Info"
    Write-ColorOutput "   .\cleanup.ps1" -Color "Cyan"
    Write-Host ""
    
    Write-ColorOutput "📚 For more information, see README.md" -Color "Info"
}

function Start-SpecificScenario {
    param([string]$ScenarioName)
    
    Write-Header "Starting Specific Scenario: $ScenarioName"
    
    switch ($ScenarioName.ToLower()) {
        "smart-home" {
            Write-ColorOutput "🏠 Starting Smart Home scenario..." -Color "Info"
            # Add specific smart home initialization
        }
        "industrial" {
            Write-ColorOutput "🏭 Starting Industrial IoT scenario..." -Color "Info"
            # Add specific industrial initialization
        }
        "retail" {
            Write-ColorOutput "🛒 Starting Retail Edge scenario..." -Color "Info"
            # Add specific retail initialization
        }
        default {
            Write-ColorOutput "Starting all scenarios..." -Color "Info"
        }
    }
}

# Main execution
try {
    Write-Header "Zig AI Platform IoT Demo Launcher"
    
    # Validate environment
    Test-DockerStatus
    
    # Start the environment
    Start-IoTEnvironment
    
    # Wait for services to be ready
    Wait-ForServices
    
    # Show current status
    Show-ServiceStatus
    
    # Test basic functionality
    Test-BasicFunctionality
    
    # Start specific scenario if requested
    if ($Scenario -ne "all") {
        Start-SpecificScenario -ScenarioName $Scenario
    }
    
    # Show next steps
    Show-NextSteps
    
} catch {
    Write-ColorOutput "❌ Demo startup failed: $($_.Exception.Message)" -Color "Error"
    Write-ColorOutput "💡 Try running: .\cleanup.ps1 && .\setup.ps1 && .\start-iot-demo.ps1" -Color "Info"
    exit 1
}

# Keep script running to show logs if verbose
if ($Verbose) {
    Write-ColorOutput "Showing live logs (Ctrl+C to exit)..." -Color "Info"
    Set-Location (Join-Path $PSScriptRoot "docker")
    docker-compose logs -f
}

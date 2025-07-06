# Zig AI Platform IoT Demo Cleanup Script
# Safely stops and removes all demo components

param(
    [switch]$Force,
    [switch]$KeepImages,
    [switch]$KeepData,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

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

function Confirm-Action {
    param([string]$Message)
    
    if ($Force) {
        return $true
    }
    
    Write-ColorOutput "$Message (y/N): " -Color "Warning"
    $response = Read-Host
    return $response.ToLower() -eq 'y' -or $response.ToLower() -eq 'yes'
}

function Stop-DockerContainers {
    Write-Header "Stopping Docker Containers"
    
    $dockerPath = Join-Path $PSScriptRoot "docker"
    if (-not (Test-Path $dockerPath)) {
        Write-ColorOutput "⚠ Docker directory not found. Skipping container cleanup." -Color "Warning"
        return
    }
    
    try {
        Set-Location $dockerPath
        
        # Check if containers are running
        $runningContainers = docker-compose ps -q 2>$null
        
        if ($runningContainers) {
            Write-ColorOutput "Stopping IoT demo containers..." -Color "Info"
            
            if ($Verbose) {
                docker-compose down
            } else {
                docker-compose down 2>$null
            }
            
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "✓ Containers stopped successfully" -Color "Success"
            } else {
                Write-ColorOutput "⚠ Some containers may not have stopped cleanly" -Color "Warning"
            }
        } else {
            Write-ColorOutput "ℹ No running containers found" -Color "Info"
        }
        
        # Remove volumes if requested
        if (-not $KeepData) {
            if (Confirm-Action "Remove Docker volumes (this will delete all data)?") {
                Write-ColorOutput "Removing Docker volumes..." -Color "Info"
                docker-compose down -v 2>$null
                Write-ColorOutput "✓ Volumes removed" -Color "Success"
            }
        }
        
    } catch {
        Write-ColorOutput "❌ Error stopping containers: $($_.Exception.Message)" -Color "Error"
    } finally {
        Set-Location $PSScriptRoot
    }
}

function Remove-DockerImages {
    if ($KeepImages) {
        Write-ColorOutput "⚠ Keeping Docker images as requested" -Color "Warning"
        return
    }
    
    Write-Header "Removing Docker Images"
    
    if (-not (Confirm-Action "Remove IoT demo Docker images?")) {
        Write-ColorOutput "Skipping image removal" -Color "Info"
        return
    }
    
    try {
        # Find IoT demo related images
        $images = docker images --format "{{.Repository}}:{{.Tag}}" | Where-Object { $_ -like "*iot-demo*" }
        
        if ($images) {
            Write-ColorOutput "Removing IoT demo images..." -Color "Info"
            foreach ($image in $images) {
                Write-ColorOutput "  Removing: $image" -Color "Info"
                docker rmi $image 2>$null
            }
            Write-ColorOutput "✓ Images removed" -Color "Success"
        } else {
            Write-ColorOutput "ℹ No IoT demo images found" -Color "Info"
        }
        
        # Clean up dangling images
        Write-ColorOutput "Cleaning up dangling images..." -Color "Info"
        docker image prune -f 2>$null
        Write-ColorOutput "✓ Dangling images cleaned" -Color "Success"
        
    } catch {
        Write-ColorOutput "❌ Error removing images: $($_.Exception.Message)" -Color "Error"
    }
}

function Clean-GeneratedFiles {
    Write-Header "Cleaning Generated Files"
    
    $filesToClean = @(
        "logs/*.log",
        "data/sensor-data/*",
        "data/inference-results/*",
        "iot-metrics-*.json",
        "scenario-results.json",
        "iot-monitoring.log"
    )
    
    foreach ($pattern in $filesToClean) {
        $files = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
        
        if ($files) {
            if ($KeepData) {
                Write-ColorOutput "⚠ Keeping data files: $pattern" -Color "Warning"
                continue
            }
            
            if (Confirm-Action "Remove files matching '$pattern'?") {
                foreach ($file in $files) {
                    try {
                        Remove-Item $file.FullName -Force
                        Write-ColorOutput "✓ Removed: $($file.Name)" -Color "Success"
                    } catch {
                        Write-ColorOutput "❌ Failed to remove: $($file.Name)" -Color "Error"
                    }
                }
            }
        } else {
            Write-ColorOutput "ℹ No files found matching: $pattern" -Color "Info"
        }
    }
}

function Clean-TempDirectories {
    Write-Header "Cleaning Temporary Directories"
    
    $tempDirs = @(
        "logs",
        "data/sensor-data", 
        "data/inference-results"
    )
    
    foreach ($dir in $tempDirs) {
        $fullPath = Join-Path $PSScriptRoot $dir
        
        if (Test-Path $fullPath) {
            $itemCount = (Get-ChildItem $fullPath -ErrorAction SilentlyContinue).Count
            
            if ($itemCount -gt 0) {
                if ($KeepData) {
                    Write-ColorOutput "⚠ Keeping directory contents: $dir" -Color "Warning"
                    continue
                }
                
                if (Confirm-Action "Clean directory '$dir' ($itemCount items)?") {
                    try {
                        Get-ChildItem $fullPath | Remove-Item -Recurse -Force
                        Write-ColorOutput "✓ Cleaned directory: $dir" -Color "Success"
                    } catch {
                        Write-ColorOutput "❌ Failed to clean directory: $dir" -Color "Error"
                    }
                }
            } else {
                Write-ColorOutput "ℹ Directory already empty: $dir" -Color "Info"
            }
        }
    }
}

function Reset-Configuration {
    Write-Header "Resetting Configuration"
    
    if (-not (Confirm-Action "Reset configuration files to defaults?")) {
        Write-ColorOutput "Skipping configuration reset" -Color "Info"
        return
    }
    
    $configFiles = @(
        "config/pi-devices.yaml",
        "config/models.yaml",
        "config/scenarios.yaml"
    )
    
    foreach ($configFile in $configFiles) {
        $fullPath = Join-Path $PSScriptRoot $configFile
        
        if (Test-Path $fullPath) {
            try {
                # Create backup
                $backupPath = "$fullPath.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
                Copy-Item $fullPath $backupPath
                Write-ColorOutput "✓ Backed up: $configFile" -Color "Success"
                
                # Note: In a real implementation, you'd restore default configs here
                Write-ColorOutput "ℹ Configuration reset would happen here" -Color "Info"
                
            } catch {
                Write-ColorOutput "❌ Failed to backup: $configFile" -Color "Error"
            }
        }
    }
}

function Show-CleanupSummary {
    Write-Header "Cleanup Summary"
    
    # Check what's left
    $dockerPath = Join-Path $PSScriptRoot "docker"
    $containersRunning = $false
    
    if (Test-Path $dockerPath) {
        try {
            Set-Location $dockerPath
            $runningContainers = docker-compose ps -q 2>$null
            $containersRunning = $runningContainers.Count -gt 0
        } catch {
            # Docker might not be available
        } finally {
            Set-Location $PSScriptRoot
        }
    }
    
    # Check for remaining files
    $logFiles = Get-ChildItem -Path "*.log" -ErrorAction SilentlyContinue
    $dataFiles = Get-ChildItem -Path "data" -Recurse -ErrorAction SilentlyContinue
    $metricFiles = Get-ChildItem -Path "iot-metrics-*.json" -ErrorAction SilentlyContinue
    
    Write-ColorOutput "🧹 Cleanup Results:" -Color "Info"
    
    if ($containersRunning) {
        Write-ColorOutput "  ⚠ Some containers may still be running" -Color "Warning"
        Write-ColorOutput "    Run 'docker-compose down' in the docker/ directory" -Color "Info"
    } else {
        Write-ColorOutput "  ✓ No containers running" -Color "Success"
    }
    
    if ($logFiles -or $dataFiles -or $metricFiles) {
        Write-ColorOutput "  ℹ Some files remain (use -Force to remove all)" -Color "Info"
        if ($logFiles) { Write-ColorOutput "    - Log files: $($logFiles.Count)" -Color "Info" }
        if ($dataFiles) { Write-ColorOutput "    - Data files: $($dataFiles.Count)" -Color "Info" }
        if ($metricFiles) { Write-ColorOutput "    - Metric files: $($metricFiles.Count)" -Color "Info" }
    } else {
        Write-ColorOutput "  ✓ All generated files cleaned" -Color "Success"
    }
    
    Write-Host ""
    Write-ColorOutput "To restart the demo:" -Color "Info"
    Write-ColorOutput "  .\start-iot-demo.ps1" -Color "Cyan"
    Write-Host ""
    Write-ColorOutput "To completely reinstall:" -Color "Info"
    Write-ColorOutput "  .\setup.ps1" -Color "Cyan"
}

function Perform-SystemCleanup {
    Write-Header "System Cleanup"
    
    if (Confirm-Action "Perform Docker system cleanup (removes unused containers, networks, images)?") {
        try {
            Write-ColorOutput "Running Docker system cleanup..." -Color "Info"
            docker system prune -f 2>$null
            Write-ColorOutput "✓ Docker system cleanup completed" -Color "Success"
        } catch {
            Write-ColorOutput "❌ Docker system cleanup failed: $($_.Exception.Message)" -Color "Error"
        }
    }
}

# Main cleanup execution
try {
    Write-Header "Zig AI Platform IoT Demo Cleanup"
    
    if (-not $Force) {
        Write-ColorOutput "This will clean up the IoT demo environment." -Color "Warning"
        Write-ColorOutput "Use -Force to skip confirmations, -KeepImages to preserve Docker images, -KeepData to preserve data files." -Color "Info"
        Write-Host ""
        
        if (-not (Confirm-Action "Continue with cleanup?")) {
            Write-ColorOutput "Cleanup cancelled." -Color "Info"
            exit 0
        }
    }
    
    # Perform cleanup steps
    Stop-DockerContainers
    Remove-DockerImages
    Clean-GeneratedFiles
    Clean-TempDirectories
    
    if ($Force) {
        Perform-SystemCleanup
    }
    
    # Show summary
    Show-CleanupSummary
    
    Write-ColorOutput "🎉 Cleanup completed!" -Color "Success"
    
} catch {
    Write-ColorOutput "❌ Cleanup failed: $($_.Exception.Message)" -Color "Error"
    Write-ColorOutput "You may need to manually clean up some resources." -Color "Warning"
    exit 1
}

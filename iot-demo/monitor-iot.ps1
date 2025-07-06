# Zig AI Platform IoT Demo Monitoring Script
# Real-time monitoring dashboard for IoT devices and performance metrics

param(
    [int]$RefreshInterval = 5,
    [switch]$Detailed,
    [switch]$SaveLogs,
    [string]$LogFile = "iot-monitoring.log"
)

$ErrorActionPreference = "Continue"

# Colors for output
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Header = "Magenta"
    Metric = "White"
}

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput "=" * 80 -Color "Header"
    Write-ColorOutput "  $Title" -Color "Header"
    Write-ColorOutput "=" * 80 -Color "Header"
}

function Get-ContainerStatus {
    try {
        $containers = docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Select-Object -Skip 1
        return $containers
    } catch {
        return @("Docker not available")
    }
}

function Get-DeviceMetrics {
    param([string]$DeviceName, [int]$Port)
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$Port/api/metrics" -TimeoutSec 3
        return $response
    } catch {
        return @{
            device_name = $DeviceName
            status = "offline"
            error = $_.Exception.Message
        }
    }
}

function Get-CoordinatorStatus {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8080/api/status" -TimeoutSec 3
        return $response
    } catch {
        return @{
            coordinator_status = "offline"
            error = $_.Exception.Message
        }
    }
}

function Format-Uptime {
    param([float]$Seconds)
    
    if ($Seconds -lt 60) {
        return "$([math]::Round($Seconds, 1))s"
    } elseif ($Seconds -lt 3600) {
        return "$([math]::Round($Seconds / 60, 1))m"
    } else {
        return "$([math]::Round($Seconds / 3600, 1))h"
    }
}

function Format-MemoryUsage {
    param([float]$Percentage)
    
    $color = if ($Percentage -lt 70) { "Success" } elseif ($Percentage -lt 85) { "Warning" } else { "Error" }
    return @{ Text = "$([math]::Round($Percentage, 1))%"; Color = $color }
}

function Format-Temperature {
    param([float]$Celsius)
    
    $color = if ($Celsius -lt 60) { "Success" } elseif ($Celsius -lt 70) { "Warning" } else { "Error" }
    return @{ Text = "$([math]::Round($Celsius, 1))°C"; Color = $color }
}

function Show-DeviceStatus {
    param([hashtable]$Metrics)
    
    if ($Metrics.ContainsKey("error")) {
        Write-ColorOutput "    Status: OFFLINE - $($Metrics.error)" -Color "Error"
        return
    }
    
    $uptime = Format-Uptime -Seconds $Metrics.uptime_seconds
    $memory = Format-MemoryUsage -Percentage $Metrics.memory_usage_percent
    $temp = Format-Temperature -Celsius $Metrics.temperature_celsius
    
    Write-ColorOutput "    Status: ONLINE" -Color "Success"
    Write-ColorOutput "    Uptime: $uptime" -Color "Info"
    Write-ColorOutput "    CPU: $([math]::Round($Metrics.cpu_usage_percent, 1))%" -Color "Metric"
    Write-ColorOutput "    Memory: $($memory.Text)" -Color $memory.Color
    Write-ColorOutput "    Temperature: $($temp.Text)" -Color $temp.Color
    Write-ColorOutput "    Requests: $($Metrics.requests_processed)" -Color "Metric"
    Write-ColorOutput "    Avg Response: $([math]::Round($Metrics.avg_response_time_ms, 1))ms" -Color "Metric"
    Write-ColorOutput "    Model: $($Metrics.model_loaded)" -Color "Info"
    
    if ($Detailed) {
        Write-ColorOutput "    Requests/min: $($Metrics.requests_per_minute)" -Color "Metric"
    }
}

function Show-CoordinatorStatus {
    param([hashtable]$Status)
    
    if ($Status.ContainsKey("error")) {
        Write-ColorOutput "  Coordinator: OFFLINE - $($Status.error)" -Color "Error"
        return
    }
    
    Write-ColorOutput "  Coordinator: ONLINE" -Color "Success"
    
    if ($Status.ContainsKey("devices")) {
        Write-ColorOutput "  Managed Devices: $($Status.devices.Count)" -Color "Info"
        
        if ($Detailed) {
            foreach ($device in $Status.devices) {
                $loadColor = if ($device.load -lt 0.5) { "Success" } elseif ($device.load -lt 0.8) { "Warning" } else { "Error" }
                Write-ColorOutput "    - $($device.name): $($device.status), Load: $([math]::Round($device.load, 2))" -Color $loadColor
            }
        }
    }
}

function Show-SystemOverview {
    Write-Header "System Overview - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    
    # Container status
    Write-ColorOutput "Docker Containers:" -Color "Info"
    $containers = Get-ContainerStatus
    foreach ($container in $containers) {
        if ($container -like "*iot-demo*" -or $container -like "*Up*") {
            Write-ColorOutput "  ✓ $container" -Color "Success"
        } else {
            Write-ColorOutput "  ❌ $container" -Color "Error"
        }
    }
    
    Write-Host ""
    
    # Coordinator status
    Write-ColorOutput "Edge Coordinator:" -Color "Info"
    $coordinatorStatus = Get-CoordinatorStatus
    Show-CoordinatorStatus -Status $coordinatorStatus
    
    Write-Host ""
}

function Show-DeviceMetrics {
    Write-Header "Device Metrics"
    
    $devices = @(
        @{ Name = "Smart Home Pi"; Port = 8081; Icon = "🏠" },
        @{ Name = "Industrial Pi"; Port = 8082; Icon = "🏭" },
        @{ Name = "Retail Pi"; Port = 8083; Icon = "🛒" }
    )
    
    foreach ($device in $devices) {
        Write-ColorOutput "$($device.Icon) $($device.Name):" -Color "Info"
        
        $metrics = Get-DeviceMetrics -DeviceName $device.Name -Port $device.Port
        Show-DeviceStatus -Metrics $metrics
        
        Write-Host ""
    }
}

function Show-PerformanceSummary {
    Write-Header "Performance Summary"
    
    $devices = @(8081, 8082, 8083)
    $totalRequests = 0
    $totalResponseTime = 0
    $onlineDevices = 0
    $avgMemoryUsage = 0
    $avgCpuUsage = 0
    
    foreach ($port in $devices) {
        $metrics = Get-DeviceMetrics -DeviceName "device-$port" -Port $port
        
        if (-not $metrics.ContainsKey("error")) {
            $onlineDevices++
            $totalRequests += $metrics.requests_processed
            $totalResponseTime += $metrics.avg_response_time_ms
            $avgMemoryUsage += $metrics.memory_usage_percent
            $avgCpuUsage += $metrics.cpu_usage_percent
        }
    }
    
    if ($onlineDevices -gt 0) {
        $avgResponseTime = $totalResponseTime / $onlineDevices
        $avgMemoryUsage = $avgMemoryUsage / $onlineDevices
        $avgCpuUsage = $avgCpuUsage / $onlineDevices
        
        Write-ColorOutput "📊 Cluster Statistics:" -Color "Info"
        Write-ColorOutput "  Online Devices: $onlineDevices/3" -Color "Success"
        Write-ColorOutput "  Total Requests: $totalRequests" -Color "Metric"
        Write-ColorOutput "  Avg Response Time: $([math]::Round($avgResponseTime, 1))ms" -Color "Metric"
        Write-ColorOutput "  Avg CPU Usage: $([math]::Round($avgCpuUsage, 1))%" -Color "Metric"
        Write-ColorOutput "  Avg Memory Usage: $([math]::Round($avgMemoryUsage, 1))%" -Color "Metric"
    } else {
        Write-ColorOutput "❌ No devices online" -Color "Error"
    }
    
    Write-Host ""
}

function Show-QuickActions {
    Write-Header "Quick Actions"
    
    Write-ColorOutput "Available Commands:" -Color "Info"
    Write-ColorOutput "  [R] Refresh now" -Color "Cyan"
    Write-ColorOutput "  [D] Toggle detailed view" -Color "Cyan"
    Write-ColorOutput "  [T] Test inference" -Color "Cyan"
    Write-ColorOutput "  [L] View logs" -Color "Cyan"
    Write-ColorOutput "  [S] Save current metrics" -Color "Cyan"
    Write-ColorOutput "  [Q] Quit monitoring" -Color "Cyan"
    Write-Host ""
    Write-ColorOutput "Press any key or wait $RefreshInterval seconds for auto-refresh..." -Color "Warning"
}

function Test-InferenceEndpoint {
    Write-Header "Testing Inference Endpoints"
    
    $testQuery = "Hello, this is a test query from the monitoring system."
    $devices = @(
        @{ Name = "Smart Home Pi"; Port = 8081 },
        @{ Name = "Industrial Pi"; Port = 8082 },
        @{ Name = "Retail Pi"; Port = 8083 }
    )
    
    foreach ($device in $devices) {
        Write-ColorOutput "Testing $($device.Name)..." -Color "Info"
        
        try {
            $requestBody = @{
                query = $testQuery
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            } | ConvertTo-Json
            
            $startTime = Get-Date
            $response = Invoke-RestMethod -Uri "http://localhost:$($device.Port)/api/inference" -Method POST -Body $requestBody -ContentType "application/json" -TimeoutSec 10
            $endTime = Get-Date
            
            $responseTime = ($endTime - $startTime).TotalMilliseconds
            
            Write-ColorOutput "  ✓ Response: $($response.result.Substring(0, [math]::Min(50, $response.result.Length)))..." -Color "Success"
            Write-ColorOutput "  ✓ Time: $([math]::Round($responseTime, 1))ms" -Color "Success"
        } catch {
            Write-ColorOutput "  ❌ Failed: $($_.Exception.Message)" -Color "Error"
        }
        
        Write-Host ""
    }
}

function Save-CurrentMetrics {
    $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $metricsFile = "iot-metrics-$timestamp.json"
    
    Write-ColorOutput "Saving metrics to $metricsFile..." -Color "Info"
    
    $allMetrics = @{
        timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fffZ"
        coordinator = Get-CoordinatorStatus
        devices = @{}
    }
    
    $devices = @(
        @{ Name = "smart-home-pi"; Port = 8081 },
        @{ Name = "industrial-pi"; Port = 8082 },
        @{ Name = "retail-pi"; Port = 8083 }
    )
    
    foreach ($device in $devices) {
        $metrics = Get-DeviceMetrics -DeviceName $device.Name -Port $device.Port
        $allMetrics.devices[$device.Name] = $metrics
    }
    
    $allMetrics | ConvertTo-Json -Depth 10 | Out-File -FilePath $metricsFile -Encoding UTF8
    Write-ColorOutput "✓ Metrics saved to $metricsFile" -Color "Success"
}

function Show-LogTail {
    Write-Header "Recent Docker Logs"
    
    try {
        $dockerPath = Join-Path $PSScriptRoot "docker"
        if (Test-Path $dockerPath) {
            Set-Location $dockerPath
            docker-compose logs --tail=20
        } else {
            Write-ColorOutput "Docker directory not found" -Color "Error"
        }
    } catch {
        Write-ColorOutput "Failed to retrieve logs: $($_.Exception.Message)" -Color "Error"
    }
}

# Main monitoring loop
function Start-Monitoring {
    Write-Header "Zig AI Platform IoT Demo - Live Monitoring"
    Write-ColorOutput "Monitoring started at $(Get-Date)" -Color "Success"
    Write-ColorOutput "Refresh interval: $RefreshInterval seconds" -Color "Info"
    
    if ($SaveLogs) {
        Write-ColorOutput "Logging to: $LogFile" -Color "Info"
    }
    
    while ($true) {
        Clear-Host
        
        # Show all monitoring sections
        Show-SystemOverview
        Show-DeviceMetrics
        Show-PerformanceSummary
        Show-QuickActions
        
        # Save logs if requested
        if ($SaveLogs) {
            $logEntry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Monitoring refresh completed"
            Add-Content -Path $LogFile -Value $logEntry
        }
        
        # Wait for user input or timeout
        $timeout = $RefreshInterval * 1000  # Convert to milliseconds
        $key = $null
        
        $startTime = Get-Date
        while (((Get-Date) - $startTime).TotalMilliseconds -lt $timeout) {
            if ([Console]::KeyAvailable) {
                $key = [Console]::ReadKey($true)
                break
            }
            Start-Sleep -Milliseconds 100
        }
        
        # Handle user input
        if ($key) {
            switch ($key.KeyChar.ToString().ToUpper()) {
                'R' { 
                    Write-ColorOutput "Refreshing..." -Color "Info"
                    continue 
                }
                'D' { 
                    $Detailed = -not $Detailed
                    Write-ColorOutput "Detailed view: $Detailed" -Color "Info"
                    Start-Sleep -Seconds 1
                }
                'T' { 
                    Test-InferenceEndpoint
                    Write-ColorOutput "Press any key to continue..." -Color "Warning"
                    [Console]::ReadKey($true) | Out-Null
                }
                'L' { 
                    Show-LogTail
                    Write-ColorOutput "Press any key to continue..." -Color "Warning"
                    [Console]::ReadKey($true) | Out-Null
                }
                'S' { 
                    Save-CurrentMetrics
                    Start-Sleep -Seconds 2
                }
                'Q' { 
                    Write-ColorOutput "Monitoring stopped." -Color "Info"
                    return 
                }
                default { 
                    Write-ColorOutput "Unknown command: $($key.KeyChar)" -Color "Warning"
                    Start-Sleep -Seconds 1
                }
            }
        }
    }
}

# Start monitoring
try {
    Start-Monitoring
} catch {
    Write-ColorOutput "Monitoring failed: $($_.Exception.Message)" -Color "Error"
    exit 1
}

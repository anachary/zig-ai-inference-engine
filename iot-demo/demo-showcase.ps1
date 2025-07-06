# Zig AI Platform IoT Demo Showcase
# Comprehensive demonstration of all IoT capabilities and features

param(
    [switch]$Interactive,
    [switch]$FullDemo,
    [int]$PauseSeconds = 3,
    [switch]$SaveResults
)

$ErrorActionPreference = "Continue"

# Colors for output
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Header = "Magenta"
    Highlight = "Yellow"
    Demo = "White"
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
    Write-Host ""
}

function Write-DemoStep {
    param([string]$Step, [string]$Description)
    Write-ColorOutput "🎯 $Step" -Color "Highlight"
    Write-ColorOutput "   $Description" -Color "Demo"
    Write-Host ""
}

function Wait-ForUser {
    param([string]$Message = "Press any key to continue...")
    
    if ($Interactive) {
        Write-ColorOutput $Message -Color "Warning"
        [Console]::ReadKey($true) | Out-Null
    } else {
        Start-Sleep -Seconds $PauseSeconds
    }
}

function Show-WelcomeMessage {
    Clear-Host
    Write-Header "Zig AI Platform IoT Demo Showcase"
    
    Write-ColorOutput "🚀 Welcome to the Zig AI Platform IoT Demonstration!" -Color "Success"
    Write-Host ""
    Write-ColorOutput "This showcase will demonstrate:" -Color "Info"
    Write-ColorOutput "  • Edge AI inference on simulated Raspberry Pi devices" -Color "Demo"
    Write-ColorOutput "  • Distributed load balancing and coordination" -Color "Demo"
    Write-ColorOutput "  • Real-world IoT scenarios (Smart Home, Industrial, Retail)" -Color "Demo"
    Write-ColorOutput "  • Performance monitoring and analytics" -Color "Demo"
    Write-ColorOutput "  • Resource optimization and efficiency" -Color "Demo"
    Write-Host ""
    
    if ($Interactive) {
        Write-ColorOutput "Running in INTERACTIVE mode - you control the pace" -Color "Highlight"
    } else {
        Write-ColorOutput "Running in AUTOMATIC mode - $PauseSeconds second pauses" -Color "Highlight"
    }
    
    Write-Host ""
    Wait-ForUser "Ready to begin the demonstration?"
}

function Demo-SystemStatus {
    Write-Header "Step 1: System Status Overview"
    
    Write-DemoStep "Checking System Health" "Verifying all components are running properly"
    
    # Check Docker containers
    Write-ColorOutput "📦 Docker Container Status:" -Color "Info"
    try {
        $containers = docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        $containers | ForEach-Object {
            if ($_ -match "iot-demo" -and $_ -match "Up") {
                Write-ColorOutput "  ✅ $_" -Color "Success"
            } elseif ($_ -match "iot-demo") {
                Write-ColorOutput "  ❌ $_" -Color "Error"
            }
        }
    } catch {
        Write-ColorOutput "  ❌ Docker not available or containers not running" -Color "Error"
    }
    
    Write-Host ""
    
    # Check service endpoints
    Write-ColorOutput "🌐 Service Endpoint Status:" -Color "Info"
    $endpoints = @(
        @{ Name = "Edge Coordinator"; Url = "http://localhost:8080/health" },
        @{ Name = "Smart Home Pi"; Url = "http://localhost:8081/health" },
        @{ Name = "Industrial Pi"; Url = "http://localhost:8082/health" },
        @{ Name = "Retail Pi"; Url = "http://localhost:8083/health" }
    )
    
    foreach ($endpoint in $endpoints) {
        try {
            $response = Invoke-WebRequest -Uri $endpoint.Url -TimeoutSec 3 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-ColorOutput "  ✅ $($endpoint.Name): Online" -Color "Success"
            } else {
                Write-ColorOutput "  ⚠️ $($endpoint.Name): Status $($response.StatusCode)" -Color "Warning"
            }
        } catch {
            Write-ColorOutput "  ❌ $($endpoint.Name): Offline" -Color "Error"
        }
    }
    
    Wait-ForUser
}

function Demo-SmartHomeScenario {
    Write-Header "Step 2: Smart Home AI Assistant"
    
    Write-DemoStep "Smart Home Scenario" "Demonstrating voice commands and home automation AI"
    
    $smartHomeQueries = @(
        @{ Query = "Turn on the living room lights"; Expected = "lighting control" },
        @{ Query = "What's the weather like today?"; Expected = "weather information" },
        @{ Query = "Set a timer for 10 minutes"; Expected = "timer functionality" },
        @{ Query = "Play some relaxing music"; Expected = "music control" }
    )
    
    Write-ColorOutput "🏠 Smart Home Pi (Raspberry Pi 4 - 4GB RAM)" -Color "Info"
    Write-ColorOutput "   Use Case: Voice assistant and home automation" -Color "Demo"
    Write-ColorOutput "   Model: Lightweight conversational AI (1B parameters)" -Color "Demo"
    Write-Host ""
    
    foreach ($query in $smartHomeQueries) {
        Write-ColorOutput "👤 User: ""$($query.Query)""" -Color "Highlight"
        
        try {
            $requestBody = @{
                query = $query.Query
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                device_id = "smart-home-pi"
            } | ConvertTo-Json
            
            $startTime = Get-Date
            $response = Invoke-RestMethod -Uri "http://localhost:8081/api/inference" -Method POST -Body $requestBody -ContentType "application/json" -TimeoutSec 15
            $endTime = Get-Date
            
            $responseTime = ($endTime - $startTime).TotalMilliseconds
            
            Write-ColorOutput "🤖 Assistant: ""$($response.result)""" -Color "Success"
            Write-ColorOutput "   ⏱️ Response time: $([math]::Round($responseTime, 1))ms" -Color "Info"
            Write-ColorOutput "   🧠 Model: $($response.model_used)" -Color "Info"
            Write-ColorOutput "   💾 Memory usage: $([math]::Round($response.resource_usage.memory_usage_percent, 1))%" -Color "Info"
            
        } catch {
            Write-ColorOutput "❌ Error: $($_.Exception.Message)" -Color "Error"
        }
        
        Write-Host ""
        Start-Sleep -Seconds 2
    }
    
    Wait-ForUser
}

function Demo-IndustrialScenario {
    Write-Header "Step 3: Industrial IoT Monitoring"
    
    Write-DemoStep "Industrial Scenario" "Demonstrating predictive maintenance and sensor analysis"
    
    $industrialQueries = @(
        @{ Query = "Analyze vibration data: 12.5Hz, 0.8mm amplitude"; Type = "Vibration Analysis" },
        @{ Query = "Temperature reading: 85°C, normal range?"; Type = "Temperature Monitoring" },
        @{ Query = "Pressure sensor shows 45 PSI, evaluate"; Type = "Pressure Analysis" },
        @{ Query = "Motor current: 15.2A, predict maintenance needs"; Type = "Predictive Maintenance" }
    )
    
    Write-ColorOutput "🏭 Industrial Pi (Raspberry Pi 4 - 8GB RAM)" -Color "Info"
    Write-ColorOutput "   Use Case: Predictive maintenance and anomaly detection" -Color "Demo"
    Write-ColorOutput "   Model: Specialized industrial AI (3B parameters)" -Color "Demo"
    Write-Host ""
    
    foreach ($query in $industrialQueries) {
        Write-ColorOutput "📊 Sensor Data: $($query.Type)" -Color "Highlight"
        Write-ColorOutput "   Input: ""$($query.Query)""" -Color "Demo"
        
        try {
            $requestBody = @{
                query = $query.Query
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                device_id = "industrial-pi"
                priority = "high"
            } | ConvertTo-Json
            
            $startTime = Get-Date
            $response = Invoke-RestMethod -Uri "http://localhost:8082/api/inference" -Method POST -Body $requestBody -ContentType "application/json" -TimeoutSec 15
            $endTime = Get-Date
            
            $responseTime = ($endTime - $startTime).TotalMilliseconds
            
            Write-ColorOutput "🔍 Analysis: ""$($response.result)""" -Color "Success"
            Write-ColorOutput "   ⏱️ Processing time: $([math]::Round($responseTime, 1))ms" -Color "Info"
            Write-ColorOutput "   🌡️ Device temp: $([math]::Round($response.resource_usage.temperature_celsius, 1))°C" -Color "Info"
            Write-ColorOutput "   ⚡ CPU usage: $([math]::Round($response.resource_usage.cpu_usage_percent, 1))%" -Color "Info"
            
        } catch {
            Write-ColorOutput "❌ Analysis failed: $($_.Exception.Message)" -Color "Error"
        }
        
        Write-Host ""
        Start-Sleep -Seconds 2
    }
    
    Wait-ForUser
}

function Demo-RetailScenario {
    Write-Header "Step 4: Retail Customer Service"
    
    Write-DemoStep "Retail Scenario" "Demonstrating customer service chatbot and inventory assistance"
    
    $retailQueries = @(
        @{ Query = "Where can I find organic vegetables?"; Type = "Product Location" },
        @{ Query = "Any discounts on electronics today?"; Type = "Promotions Inquiry" },
        @{ Query = "Do you have this item in stock?"; Type = "Inventory Check" },
        @{ Query = "What are your store hours?"; Type = "Store Information" }
    )
    
    Write-ColorOutput "🛒 Retail Pi (Raspberry Pi 4 - 4GB RAM)" -Color "Info"
    Write-ColorOutput "   Use Case: Customer service chatbot and inventory analysis" -Color "Demo"
    Write-ColorOutput "   Model: Retail-optimized AI (2B parameters)" -Color "Demo"
    Write-Host ""
    
    foreach ($query in $retailQueries) {
        Write-ColorOutput "🛍️ Customer: $($query.Type)" -Color "Highlight"
        Write-ColorOutput "   Question: ""$($query.Query)""" -Color "Demo"
        
        try {
            $requestBody = @{
                query = $query.Query
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                device_id = "retail-pi"
            } | ConvertTo-Json
            
            $startTime = Get-Date
            $response = Invoke-RestMethod -Uri "http://localhost:8083/api/inference" -Method POST -Body $requestBody -ContentType "application/json" -TimeoutSec 15
            $endTime = Get-Date
            
            $responseTime = ($endTime - $startTime).TotalMilliseconds
            
            Write-ColorOutput "🤝 Assistant: ""$($response.result)""" -Color "Success"
            Write-ColorOutput "   ⏱️ Response time: $([math]::Round($responseTime, 1))ms" -Color "Info"
            Write-ColorOutput "   📱 Network: 4G connection simulated" -Color "Info"
            
        } catch {
            Write-ColorOutput "❌ Service unavailable: $($_.Exception.Message)" -Color "Error"
        }
        
        Write-Host ""
        Start-Sleep -Seconds 2
    }
    
    Wait-ForUser
}

function Demo-LoadBalancing {
    Write-Header "Step 5: Load Balancing and Coordination"
    
    Write-DemoStep "Edge Coordination" "Demonstrating intelligent load balancing across devices"
    
    Write-ColorOutput "🎛️ Edge Coordinator Status:" -Color "Info"
    
    try {
        $coordinatorStatus = Invoke-RestMethod -Uri "http://localhost:8080/api/status" -TimeoutSec 5
        
        Write-ColorOutput "   Status: $($coordinatorStatus.coordinator_status)" -Color "Success"
        Write-ColorOutput "   Managed devices: $($coordinatorStatus.devices.Count)" -Color "Info"
        
        Write-Host ""
        Write-ColorOutput "📊 Device Load Distribution:" -Color "Info"
        
        foreach ($device in $coordinatorStatus.devices) {
            $loadColor = if ($device.load -lt 0.5) { "Success" } elseif ($device.load -lt 0.8) { "Warning" } else { "Error" }
            $statusIcon = if ($device.status -eq "online") { "🟢" } else { "🔴" }
            
            Write-ColorOutput "   $statusIcon $($device.name): Load $([math]::Round($device.load, 2)) | Requests: $($device.requests)" -Color $loadColor
        }
        
    } catch {
        Write-ColorOutput "❌ Coordinator not responding: $($_.Exception.Message)" -Color "Error"
    }
    
    Write-Host ""
    Write-ColorOutput "🔄 Testing Load Balancing:" -Color "Info"
    
    # Send multiple requests to test load balancing
    $testQueries = @(
        "Test query 1 - routing test",
        "Test query 2 - load balancing",
        "Test query 3 - distribution test"
    )
    
    foreach ($query in $testQueries) {
        try {
            $requestBody = @{
                query = $query
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            } | ConvertTo-Json
            
            $response = Invoke-RestMethod -Uri "http://localhost:8080/api/inference" -Method POST -Body $requestBody -ContentType "application/json" -TimeoutSec 10
            
            Write-ColorOutput "   ✅ Routed to: $($response.device_name) ($([math]::Round($response.processing_time_ms, 1))ms)" -Color "Success"
            
        } catch {
            Write-ColorOutput "   ❌ Routing failed: $($_.Exception.Message)" -Color "Error"
        }
        
        Start-Sleep -Seconds 1
    }
    
    Wait-ForUser
}

function Demo-PerformanceMetrics {
    Write-Header "Step 6: Performance Analytics"
    
    Write-DemoStep "Performance Monitoring" "Showing real-time metrics and resource utilization"
    
    $devices = @(
        @{ Name = "Smart Home Pi"; Port = 8081; Icon = "🏠" },
        @{ Name = "Industrial Pi"; Port = 8082; Icon = "🏭" },
        @{ Name = "Retail Pi"; Port = 8083; Icon = "🛒" }
    )
    
    $allMetrics = @()
    
    foreach ($device in $devices) {
        Write-ColorOutput "$($device.Icon) $($device.Name) Metrics:" -Color "Info"
        
        try {
            $metrics = Invoke-RestMethod -Uri "http://localhost:$($device.Port)/api/metrics" -TimeoutSec 5
            $allMetrics += $metrics
            
            Write-ColorOutput "   📈 Requests processed: $($metrics.requests_processed)" -Color "Demo"
            Write-ColorOutput "   ⚡ Avg response time: $([math]::Round($metrics.avg_response_time_ms, 1))ms" -Color "Demo"
            Write-ColorOutput "   💾 Memory usage: $([math]::Round($metrics.memory_usage_percent, 1))%" -Color "Demo"
            Write-ColorOutput "   🔥 CPU usage: $([math]::Round($metrics.cpu_usage_percent, 1))%" -Color "Demo"
            Write-ColorOutput "   🌡️ Temperature: $([math]::Round($metrics.temperature_celsius, 1))°C" -Color "Demo"
            Write-ColorOutput "   ⏰ Uptime: $([math]::Round($metrics.uptime_seconds / 60, 1)) minutes" -Color "Demo"
            
        } catch {
            Write-ColorOutput "   ❌ Metrics unavailable: $($_.Exception.Message)" -Color "Error"
        }
        
        Write-Host ""
    }
    
    # Calculate cluster statistics
    if ($allMetrics.Count -gt 0) {
        $totalRequests = ($allMetrics | Measure-Object -Property requests_processed -Sum).Sum
        $avgResponseTime = ($allMetrics | Measure-Object -Property avg_response_time_ms -Average).Average
        $avgMemoryUsage = ($allMetrics | Measure-Object -Property memory_usage_percent -Average).Average
        $avgCpuUsage = ($allMetrics | Measure-Object -Property cpu_usage_percent -Average).Average
        
        Write-ColorOutput "🎯 Cluster Performance Summary:" -Color "Highlight"
        Write-ColorOutput "   Total requests processed: $totalRequests" -Color "Success"
        Write-ColorOutput "   Average response time: $([math]::Round($avgResponseTime, 1))ms" -Color "Success"
        Write-ColorOutput "   Average memory usage: $([math]::Round($avgMemoryUsage, 1))%" -Color "Success"
        Write-ColorOutput "   Average CPU usage: $([math]::Round($avgCpuUsage, 1))%" -Color "Success"
    }
    
    Wait-ForUser
}

function Demo-Conclusion {
    Write-Header "Demo Complete - Key Achievements"
    
    Write-ColorOutput "🎉 Zig AI Platform IoT Demo Successfully Completed!" -Color "Success"
    Write-Host ""
    
    Write-ColorOutput "✅ Demonstrated Capabilities:" -Color "Info"
    Write-ColorOutput "   • Edge AI inference on resource-constrained devices" -Color "Demo"
    Write-ColorOutput "   • Intelligent load balancing and request routing" -Color "Demo"
    Write-ColorOutput "   • Real-world IoT scenarios across multiple domains" -Color "Demo"
    Write-ColorOutput "   • Performance monitoring and resource optimization" -Color "Demo"
    Write-ColorOutput "   • Distributed coordination and fault tolerance" -Color "Demo"
    Write-Host ""
    
    Write-ColorOutput "🚀 Next Steps:" -Color "Info"
    Write-ColorOutput "   • Deploy on real Raspberry Pi hardware" -Color "Demo"
    Write-ColorOutput "   • Scale to larger device clusters" -Color "Demo"
    Write-ColorOutput "   • Integrate with cloud services" -Color "Demo"
    Write-ColorOutput "   • Customize for your specific use cases" -Color "Demo"
    Write-Host ""
    
    Write-ColorOutput "📚 Resources:" -Color "Info"
    Write-ColorOutput "   • Documentation: docs/DEPLOYMENT_GUIDE.md" -Color "Demo"
    Write-ColorOutput "   • Source code: https://github.com/anachary/zig-ai-platform" -Color "Demo"
    Write-ColorOutput "   • Issues & support: GitHub Issues" -Color "Demo"
    Write-Host ""
    
    if ($SaveResults) {
        $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
        $resultsFile = "demo-results-$timestamp.txt"
        
        "Zig AI Platform IoT Demo Results - $timestamp" | Out-File $resultsFile
        "Demo completed successfully at $(Get-Date)" | Add-Content $resultsFile
        
        Write-ColorOutput "💾 Demo results saved to: $resultsFile" -Color "Success"
    }
    
    Write-ColorOutput "Thank you for exploring the Zig AI Platform IoT capabilities!" -Color "Highlight"
}

# Main demo execution
try {
    Show-WelcomeMessage
    Demo-SystemStatus
    Demo-SmartHomeScenario
    Demo-IndustrialScenario
    Demo-RetailScenario
    Demo-LoadBalancing
    Demo-PerformanceMetrics
    Demo-Conclusion
    
} catch {
    Write-ColorOutput "❌ Demo failed: $($_.Exception.Message)" -Color "Error"
    Write-ColorOutput "Please ensure the IoT demo is running: .\start-iot-demo.ps1" -Color "Warning"
    exit 1
}

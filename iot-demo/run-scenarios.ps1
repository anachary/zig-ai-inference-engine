# Zig AI Platform IoT Demo Scenarios Runner
# This script runs comprehensive test scenarios to demonstrate IoT capabilities

param(
    [string]$Scenario = "all",
    [int]$Iterations = 1,
    [switch]$Detailed,
    [switch]$SaveResults,
    [string]$OutputFile = "scenario-results.json"
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

# Global results storage
$Global:ScenarioResults = @{
    StartTime = Get-Date
    Scenarios = @()
    Summary = @{}
}

function Test-ServiceAvailability {
    Write-Header "Checking Service Availability"
    
    $services = @(
        @{ Name = "Edge Coordinator"; Url = "http://localhost:8080/health" },
        @{ Name = "Smart Home Pi"; Url = "http://localhost:8081/health" },
        @{ Name = "Industrial Pi"; Url = "http://localhost:8082/health" },
        @{ Name = "Retail Pi"; Url = "http://localhost:8083/health" }
    )
    
    $allAvailable = $true
    
    foreach ($service in $services) {
        try {
            $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-ColorOutput "✓ $($service.Name) is available" -Color "Success"
            } else {
                Write-ColorOutput "❌ $($service.Name) returned status $($response.StatusCode)" -Color "Error"
                $allAvailable = $false
            }
        } catch {
            Write-ColorOutput "❌ $($service.Name) is not responding" -Color "Error"
            $allAvailable = $false
        }
    }
    
    if (-not $allAvailable) {
        Write-ColorOutput "⚠ Some services are not available. Please run .\start-iot-demo.ps1 first." -Color "Warning"
        exit 1
    }
    
    Write-ColorOutput "🎉 All services are available!" -Color "Success"
}

function Invoke-InferenceRequest {
    param(
        [string]$DeviceUrl,
        [string]$Query,
        [string]$DeviceName
    )
    
    $startTime = Get-Date
    
    try {
        $requestBody = @{
            query = $Query
            timestamp = $startTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            device_id = $DeviceName
        } | ConvertTo-Json
        
        $response = Invoke-RestMethod -Uri "$DeviceUrl/api/inference" -Method POST -Body $requestBody -ContentType "application/json" -TimeoutSec 30
        
        $endTime = Get-Date
        $responseTime = ($endTime - $startTime).TotalMilliseconds
        
        return @{
            Success = $true
            Query = $Query
            Response = $response.result
            ResponseTimeMs = [math]::Round($responseTime, 2)
            Timestamp = $startTime
            DeviceName = $DeviceName
        }
    } catch {
        $endTime = Get-Date
        $responseTime = ($endTime - $startTime).TotalMilliseconds
        
        return @{
            Success = $false
            Query = $Query
            Error = $_.Exception.Message
            ResponseTimeMs = [math]::Round($responseTime, 2)
            Timestamp = $startTime
            DeviceName = $DeviceName
        }
    }
}

function Run-SmartHomeScenario {
    Write-Header "Smart Home Assistant Scenario"
    
    $deviceUrl = "http://localhost:8081"
    $queries = @(
        "Turn on the living room lights",
        "What's the weather like today?",
        "Set a timer for 10 minutes",
        "Play some relaxing music",
        "What's the temperature in the bedroom?",
        "Lock the front door",
        "Start the coffee maker",
        "Dim the lights to 50%"
    )
    
    $scenarioResults = @{
        Name = "Smart Home Assistant"
        DeviceName = "smart-home-pi"
        StartTime = Get-Date
        Requests = @()
        Statistics = @{}
    }
    
    Write-ColorOutput "🏠 Testing Smart Home Pi with $($queries.Count) queries..." -Color "Info"
    
    foreach ($query in $queries) {
        Write-ColorOutput "   Query: $query" -Color "Cyan"
        
        $result = Invoke-InferenceRequest -DeviceUrl $deviceUrl -Query $query -DeviceName "smart-home-pi"
        $scenarioResults.Requests += $result
        
        if ($result.Success) {
            Write-ColorOutput "   ✓ Response: $($result.Response) ($($result.ResponseTimeMs)ms)" -Color "Success"
        } else {
            Write-ColorOutput "   ❌ Error: $($result.Error) ($($result.ResponseTimeMs)ms)" -Color "Error"
        }
        
        if ($Detailed) {
            Start-Sleep -Seconds 1  # Prevent overwhelming the device
        }
    }
    
    # Calculate statistics
    $successfulRequests = $scenarioResults.Requests | Where-Object { $_.Success }
    $failedRequests = $scenarioResults.Requests | Where-Object { -not $_.Success }
    
    $scenarioResults.Statistics = @{
        TotalRequests = $scenarioResults.Requests.Count
        SuccessfulRequests = $successfulRequests.Count
        FailedRequests = $failedRequests.Count
        SuccessRate = [math]::Round(($successfulRequests.Count / $scenarioResults.Requests.Count) * 100, 2)
        AverageResponseTime = if ($successfulRequests.Count -gt 0) { [math]::Round(($successfulRequests | Measure-Object -Property ResponseTimeMs -Average).Average, 2) } else { 0 }
        MinResponseTime = if ($successfulRequests.Count -gt 0) { ($successfulRequests | Measure-Object -Property ResponseTimeMs -Minimum).Minimum } else { 0 }
        MaxResponseTime = if ($successfulRequests.Count -gt 0) { ($successfulRequests | Measure-Object -Property ResponseTimeMs -Maximum).Maximum } else { 0 }
    }
    
    $scenarioResults.EndTime = Get-Date
    $Global:ScenarioResults.Scenarios += $scenarioResults
    
    Write-ColorOutput "📊 Smart Home Results:" -Color "Info"
    Write-ColorOutput "   Success Rate: $($scenarioResults.Statistics.SuccessRate)%" -Color "Success"
    Write-ColorOutput "   Average Response Time: $($scenarioResults.Statistics.AverageResponseTime)ms" -Color "Info"
    Write-ColorOutput "   Range: $($scenarioResults.Statistics.MinResponseTime)ms - $($scenarioResults.Statistics.MaxResponseTime)ms" -Color "Info"
}

function Run-IndustrialScenario {
    Write-Header "Industrial IoT Monitoring Scenario"
    
    $deviceUrl = "http://localhost:8082"
    $queries = @(
        "Analyze vibration data: 12.5Hz, 0.8mm amplitude",
        "Temperature reading: 85°C, normal range?",
        "Pressure sensor shows 45 PSI, evaluate",
        "Motor current: 15.2A, predict maintenance needs",
        "Oil pressure: 32 PSI, status check",
        "Belt tension: 150N, within specifications?",
        "Bearing temperature: 78°C, assessment needed",
        "Flow rate: 125 L/min, performance analysis"
    )
    
    $scenarioResults = @{
        Name = "Industrial IoT Monitoring"
        DeviceName = "industrial-pi"
        StartTime = Get-Date
        Requests = @()
        Statistics = @{}
    }
    
    Write-ColorOutput "🏭 Testing Industrial Pi with $($queries.Count) sensor queries..." -Color "Info"
    
    foreach ($query in $queries) {
        Write-ColorOutput "   Sensor Query: $query" -Color "Cyan"
        
        $result = Invoke-InferenceRequest -DeviceUrl $deviceUrl -Query $query -DeviceName "industrial-pi"
        $scenarioResults.Requests += $result
        
        if ($result.Success) {
            Write-ColorOutput "   ✓ Analysis: $($result.Response) ($($result.ResponseTimeMs)ms)" -Color "Success"
        } else {
            Write-ColorOutput "   ❌ Error: $($result.Error) ($($result.ResponseTimeMs)ms)" -Color "Error"
        }
        
        if ($Detailed) {
            Start-Sleep -Seconds 1
        }
    }
    
    # Calculate statistics
    $successfulRequests = $scenarioResults.Requests | Where-Object { $_.Success }
    $failedRequests = $scenarioResults.Requests | Where-Object { -not $_.Success }
    
    $scenarioResults.Statistics = @{
        TotalRequests = $scenarioResults.Requests.Count
        SuccessfulRequests = $successfulRequests.Count
        FailedRequests = $failedRequests.Count
        SuccessRate = [math]::Round(($successfulRequests.Count / $scenarioResults.Requests.Count) * 100, 2)
        AverageResponseTime = if ($successfulRequests.Count -gt 0) { [math]::Round(($successfulRequests | Measure-Object -Property ResponseTimeMs -Average).Average, 2) } else { 0 }
        MinResponseTime = if ($successfulRequests.Count -gt 0) { ($successfulRequests | Measure-Object -Property ResponseTimeMs -Minimum).Minimum } else { 0 }
        MaxResponseTime = if ($successfulRequests.Count -gt 0) { ($successfulRequests | Measure-Object -Property ResponseTimeMs -Maximum).Maximum } else { 0 }
    }
    
    $scenarioResults.EndTime = Get-Date
    $Global:ScenarioResults.Scenarios += $scenarioResults
    
    Write-ColorOutput "📊 Industrial Results:" -Color "Info"
    Write-ColorOutput "   Success Rate: $($scenarioResults.Statistics.SuccessRate)%" -Color "Success"
    Write-ColorOutput "   Average Response Time: $($scenarioResults.Statistics.AverageResponseTime)ms" -Color "Info"
    Write-ColorOutput "   Range: $($scenarioResults.Statistics.MinResponseTime)ms - $($scenarioResults.Statistics.MaxResponseTime)ms" -Color "Info"
}

function Run-RetailScenario {
    Write-Header "Retail Customer Service Scenario"
    
    $deviceUrl = "http://localhost:8083"
    $queries = @(
        "Where can I find organic vegetables?",
        "Any discounts on electronics today?",
        "Do you have this item in stock?",
        "What are your store hours?",
        "Where is the customer service desk?",
        "Can you help me find size medium shirts?",
        "What's the return policy?",
        "Are there any sales this weekend?"
    )
    
    $scenarioResults = @{
        Name = "Retail Customer Service"
        DeviceName = "retail-pi"
        StartTime = Get-Date
        Requests = @()
        Statistics = @{}
    }
    
    Write-ColorOutput "🛒 Testing Retail Pi with $($queries.Count) customer queries..." -Color "Info"
    
    foreach ($query in $queries) {
        Write-ColorOutput "   Customer: $query" -Color "Cyan"
        
        $result = Invoke-InferenceRequest -DeviceUrl $deviceUrl -Query $query -DeviceName "retail-pi"
        $scenarioResults.Requests += $result
        
        if ($result.Success) {
            Write-ColorOutput "   ✓ Assistant: $($result.Response) ($($result.ResponseTimeMs)ms)" -Color "Success"
        } else {
            Write-ColorOutput "   ❌ Error: $($result.Error) ($($result.ResponseTimeMs)ms)" -Color "Error"
        }
        
        if ($Detailed) {
            Start-Sleep -Seconds 1
        }
    }
    
    # Calculate statistics
    $successfulRequests = $scenarioResults.Requests | Where-Object { $_.Success }
    $failedRequests = $scenarioResults.Requests | Where-Object { -not $_.Success }
    
    $scenarioResults.Statistics = @{
        TotalRequests = $scenarioResults.Requests.Count
        SuccessfulRequests = $successfulRequests.Count
        FailedRequests = $failedRequests.Count
        SuccessRate = [math]::Round(($successfulRequests.Count / $scenarioResults.Requests.Count) * 100, 2)
        AverageResponseTime = if ($successfulRequests.Count -gt 0) { [math]::Round(($successfulRequests | Measure-Object -Property ResponseTimeMs -Average).Average, 2) } else { 0 }
        MinResponseTime = if ($successfulRequests.Count -gt 0) { ($successfulRequests | Measure-Object -Property ResponseTimeMs -Minimum).Minimum } else { 0 }
        MaxResponseTime = if ($successfulRequests.Count -gt 0) { ($successfulRequests | Measure-Object -Property ResponseTimeMs -Maximum).Maximum } else { 0 }
    }
    
    $scenarioResults.EndTime = Get-Date
    $Global:ScenarioResults.Scenarios += $scenarioResults
    
    Write-ColorOutput "📊 Retail Results:" -Color "Info"
    Write-ColorOutput "   Success Rate: $($scenarioResults.Statistics.SuccessRate)%" -Color "Success"
    Write-ColorOutput "   Average Response Time: $($scenarioResults.Statistics.AverageResponseTime)ms" -Color "Info"
    Write-ColorOutput "   Range: $($scenarioResults.Statistics.MinResponseTime)ms - $($scenarioResults.Statistics.MaxResponseTime)ms" -Color "Info"
}

function Run-LoadTestScenario {
    Write-Header "Load Testing Scenario"
    
    Write-ColorOutput "🔥 Running concurrent load test..." -Color "Info"
    
    $concurrentRequests = @(1, 3, 5, 8)
    $testQuery = "What is the current system status?"
    
    foreach ($concurrency in $concurrentRequests) {
        Write-ColorOutput "   Testing with $concurrency concurrent requests..." -Color "Cyan"
        
        $jobs = @()
        $startTime = Get-Date
        
        for ($i = 1; $i -le $concurrency; $i++) {
            $job = Start-Job -ScriptBlock {
                param($DeviceUrl, $Query, $RequestId)
                
                try {
                    $requestBody = @{
                        query = $Query
                        request_id = $RequestId
                        timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                    } | ConvertTo-Json
                    
                    $response = Invoke-RestMethod -Uri "$DeviceUrl/api/inference" -Method POST -Body $requestBody -ContentType "application/json" -TimeoutSec 30
                    return @{ Success = $true; Response = $response; RequestId = $RequestId }
                } catch {
                    return @{ Success = $false; Error = $_.Exception.Message; RequestId = $RequestId }
                }
            } -ArgumentList "http://localhost:8081", $testQuery, $i
            
            $jobs += $job
        }
        
        # Wait for all jobs to complete
        $results = $jobs | Wait-Job | Receive-Job
        $jobs | Remove-Job
        
        $endTime = Get-Date
        $totalTime = ($endTime - $startTime).TotalMilliseconds
        
        $successCount = ($results | Where-Object { $_.Success }).Count
        $successRate = [math]::Round(($successCount / $concurrency) * 100, 2)
        
        Write-ColorOutput "   ✓ $concurrency requests: $successCount/$concurrency successful ($successRate%) in $([math]::Round($totalTime, 2))ms" -Color "Success"
    }
}

function Show-FinalSummary {
    Write-Header "Scenario Testing Complete"
    
    $Global:ScenarioResults.EndTime = Get-Date
    $totalDuration = ($Global:ScenarioResults.EndTime - $Global:ScenarioResults.StartTime).TotalSeconds
    
    # Calculate overall statistics
    $allRequests = $Global:ScenarioResults.Scenarios | ForEach-Object { $_.Requests } | Where-Object { $_ -ne $null }
    $totalRequests = $allRequests.Count
    $successfulRequests = ($allRequests | Where-Object { $_.Success }).Count
    $overallSuccessRate = if ($totalRequests -gt 0) { [math]::Round(($successfulRequests / $totalRequests) * 100, 2) } else { 0 }
    
    $Global:ScenarioResults.Summary = @{
        TotalDuration = [math]::Round($totalDuration, 2)
        TotalRequests = $totalRequests
        SuccessfulRequests = $successfulRequests
        OverallSuccessRate = $overallSuccessRate
        ScenariosRun = $Global:ScenarioResults.Scenarios.Count
    }
    
    Write-ColorOutput "🎉 All scenarios completed!" -Color "Success"
    Write-ColorOutput "📊 Overall Summary:" -Color "Info"
    Write-ColorOutput "   Total Duration: $($Global:ScenarioResults.Summary.TotalDuration) seconds" -Color "Info"
    Write-ColorOutput "   Scenarios Run: $($Global:ScenarioResults.Summary.ScenariosRun)" -Color "Info"
    Write-ColorOutput "   Total Requests: $($Global:ScenarioResults.Summary.TotalRequests)" -Color "Info"
    Write-ColorOutput "   Overall Success Rate: $($Global:ScenarioResults.Summary.OverallSuccessRate)%" -Color "Success"
    
    Write-Host ""
    Write-ColorOutput "📋 Individual Scenario Results:" -Color "Info"
    foreach ($scenario in $Global:ScenarioResults.Scenarios) {
        Write-ColorOutput "   $($scenario.Name): $($scenario.Statistics.SuccessRate)% success, $($scenario.Statistics.AverageResponseTime)ms avg" -Color "Cyan"
    }
    
    if ($SaveResults) {
        $resultsPath = Join-Path $PSScriptRoot $OutputFile
        $Global:ScenarioResults | ConvertTo-Json -Depth 10 | Out-File -FilePath $resultsPath -Encoding UTF8
        Write-ColorOutput "💾 Results saved to: $resultsPath" -Color "Success"
    }
}

# Main execution
try {
    Write-Header "Zig AI Platform IoT Demo Scenarios"
    
    # Check service availability
    Test-ServiceAvailability
    
    # Run scenarios based on parameter
    for ($iteration = 1; $iteration -le $Iterations; $iteration++) {
        if ($Iterations -gt 1) {
            Write-ColorOutput "🔄 Running iteration $iteration of $Iterations" -Color "Info"
        }
        
        switch ($Scenario.ToLower()) {
            "smart-home" { Run-SmartHomeScenario }
            "industrial" { Run-IndustrialScenario }
            "retail" { Run-RetailScenario }
            "load-test" { Run-LoadTestScenario }
            "all" {
                Run-SmartHomeScenario
                Run-IndustrialScenario
                Run-RetailScenario
                Run-LoadTestScenario
            }
            default {
                Write-ColorOutput "❌ Unknown scenario: $Scenario" -Color "Error"
                Write-ColorOutput "Available scenarios: smart-home, industrial, retail, load-test, all" -Color "Info"
                exit 1
            }
        }
        
        if ($iteration -lt $Iterations) {
            Write-ColorOutput "⏳ Waiting before next iteration..." -Color "Info"
            Start-Sleep -Seconds 5
        }
    }
    
    # Show final summary
    Show-FinalSummary
    
} catch {
    Write-ColorOutput "❌ Scenario testing failed: $($_.Exception.Message)" -Color "Error"
    exit 1
}

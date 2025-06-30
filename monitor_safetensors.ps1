# Monitor Phi-2 Safetensors Download
Write-Host "🧠 Monitoring Phi-2 Safetensors download..." -ForegroundColor Cyan
Write-Host "Expected total size: ~5.6 GB (5GB + 564MB + configs)" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
Write-Host ""

$targetFiles = @(
    @{name="model-00001-of-00002.safetensors"; expectedSize=5000000000},  # ~5GB
    @{name="model-00002-of-00002.safetensors"; expectedSize=564000000},   # ~564MB
    @{name="model.safetensors.index.json"; expectedSize=35000}            # ~35KB
)

$startTime = Get-Date

while ($true) {
    $totalDownloaded = 0
    $totalExpected = 0
    $filesCompleted = 0
    
    Write-Host "$(Get-Date -Format 'HH:mm:ss') | Checking download progress..." -ForegroundColor Cyan
    
    foreach ($file in $targetFiles) {
        $filePath = "models/$($file.name)"
        $totalExpected += $file.expectedSize
        
        if (Test-Path $filePath) {
            $currentSize = (Get-Item $filePath).Length
            $totalDownloaded += $currentSize
            
            $sizeMB = [math]::Round($currentSize / 1MB, 2)
            $expectedMB = [math]::Round($file.expectedSize / 1MB, 2)
            $percent = [math]::Round(($currentSize / $file.expectedSize) * 100, 1)
            
            if ($currentSize -ge ($file.expectedSize * 0.95)) {
                Write-Host "  ✅ $($file.name): $sizeMB MB (Complete)" -ForegroundColor Green
                $filesCompleted++
            } else {
                Write-Host "  📥 $($file.name): $sizeMB MB / $expectedMB MB ($percent%)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ⏳ $($file.name): Waiting..." -ForegroundColor Gray
        }
    }
    
    # Calculate overall progress
    $overallPercent = if ($totalExpected -gt 0) { [math]::Round(($totalDownloaded / $totalExpected) * 100, 1) } else { 0 }
    $totalDownloadedGB = [math]::Round($totalDownloaded / 1GB, 2)
    $totalExpectedGB = [math]::Round($totalExpected / 1GB, 2)
    
    # Calculate speed
    $elapsed = (Get-Date) - $startTime
    if ($elapsed.TotalSeconds -gt 0 -and $totalDownloaded -gt 0) {
        $speedMBps = [math]::Round(($totalDownloaded / 1MB) / $elapsed.TotalSeconds, 2)
        $speedStr = "$speedMBps MB/s"
        
        # ETA calculation
        if ($speedMBps -gt 0 -and $overallPercent -lt 100) {
            $remainingBytes = $totalExpected - $totalDownloaded
            $etaSeconds = ($remainingBytes / 1MB) / $speedMBps
            $eta = [TimeSpan]::FromSeconds($etaSeconds)
            $etaStr = "{0:hh\:mm\:ss}" -f $eta
        } else {
            $etaStr = "Complete"
        }
    } else {
        $speedStr = "Calculating..."
        $etaStr = "Unknown"
    }
    
    $elapsedStr = "{0:hh\:mm\:ss}" -f $elapsed
    
    Write-Host ""
    Write-Host "📊 Overall: $totalDownloadedGB GB / $totalExpectedGB GB ($overallPercent%)" -ForegroundColor Cyan
    Write-Host "⚡ Speed: $speedStr | ETA: $etaStr | Elapsed: $elapsedStr" -ForegroundColor White
    Write-Host "📁 Files completed: $filesCompleted / $($targetFiles.Count)" -ForegroundColor White
    
    # Check if all main files are complete
    if ($filesCompleted -eq $targetFiles.Count) {
        Write-Host ""
        Write-Host "🎉 All main files downloaded successfully!" -ForegroundColor Green
        Write-Host "📋 Checking additional files..." -ForegroundColor Yellow
        
        # Check for additional files
        $additionalFiles = @("config.json", "tokenizer.json", "vocab.json")
        $additionalComplete = 0
        
        foreach ($file in $additionalFiles) {
            if (Test-Path "models/$file") {
                $additionalComplete++
                Write-Host "  ✅ $file" -ForegroundColor Green
            } else {
                Write-Host "  ⏳ $file" -ForegroundColor Yellow
            }
        }
        
        if ($additionalComplete -eq $additionalFiles.Count) {
            Write-Host ""
            Write-Host "🚀 Phi-2 download complete! Ready to test!" -ForegroundColor Green
            Write-Host ""
            Write-Host "🧪 Test commands:" -ForegroundColor Cyan
            Write-Host "zig build cli -- interactive --model ./models --max-tokens 400" -ForegroundColor White
            break
        }
    }
    
    Write-Host ""
    Write-Host "----------------------------------------" -ForegroundColor DarkGray
    Start-Sleep -Seconds 15
}

Write-Host ""
Write-Host "Monitor stopped." -ForegroundColor Gray

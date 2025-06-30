# Phi-2 Model Download Monitor
Write-Host "🚀 Monitoring Phi-2 download progress..." -ForegroundColor Cyan
Write-Host "Expected size: ~5.4 GB" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
Write-Host ""

$targetSize = 5400000000  # 5.4 GB in bytes
$startTime = Get-Date

while ($true) {
    if (Test-Path "models/phi2.bin") {
        $file = Get-Item "models/phi2.bin"
        $size = $file.Length
        $sizeMB = [math]::Round($size / 1MB, 2)
        $sizeGB = [math]::Round($size / 1GB, 2)
        $percent = [math]::Round(($size / $targetSize) * 100, 1)
        
        $elapsed = (Get-Date) - $startTime
        $elapsedStr = "{0:hh\:mm\:ss}" -f $elapsed
        
        # Calculate download speed
        if ($elapsed.TotalSeconds -gt 0) {
            $speedMBps = [math]::Round($sizeMB / $elapsed.TotalSeconds, 2)
            $speedStr = "$speedMBps MB/s"
        } else {
            $speedStr = "calculating..."
        }
        
        # Estimate time remaining
        if ($speedMBps -gt 0 -and $percent -lt 100) {
            $remainingMB = ($targetSize - $size) / 1MB
            $etaSeconds = $remainingMB / $speedMBps
            $eta = [TimeSpan]::FromSeconds($etaSeconds)
            $etaStr = "{0:hh\:mm\:ss}" -f $eta
        } else {
            $etaStr = "unknown"
        }
        
        Write-Host "$(Get-Date -Format 'HH:mm:ss') | $sizeMB MB ($sizeGB GB) | $percent% | Speed: $speedStr | ETA: $etaStr | Elapsed: $elapsedStr" -ForegroundColor Green
        
        # Check if download is complete
        if ($size -gt ($targetSize * 0.95)) {  # 95% of expected size
            Write-Host ""
            Write-Host "✅ Download appears complete!" -ForegroundColor Yellow
            Write-Host "Final size: $sizeGB GB" -ForegroundColor Green
            Write-Host "Total time: $elapsedStr" -ForegroundColor Green
            Write-Host ""
            Write-Host "🧪 Ready to test! Run:" -ForegroundColor Cyan
            Write-Host "zig build cli -- interactive --model ./models/phi2.bin --max-tokens 400 --verbose" -ForegroundColor White
            break
        }
    } else {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') | Waiting for download to start..." -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds 10
}

Write-Host ""
Write-Host "Monitor stopped." -ForegroundColor Gray

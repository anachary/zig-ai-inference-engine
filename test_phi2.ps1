# Phi-2 Model Testing Script
Write-Host "🧠 Phi-2 Model Testing Suite" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

# Check if model file exists
if (-not (Test-Path "models/phi2.bin")) {
    Write-Host "❌ Error: Phi-2 model not found at models/phi2.bin" -ForegroundColor Red
    Write-Host "Please ensure the download completed successfully." -ForegroundColor Yellow
    exit 1
}

# Verify file size
$file = Get-Item "models/phi2.bin"
$sizeGB = [math]::Round($file.Length / 1GB, 2)
Write-Host "📁 Model file: $($file.Name)" -ForegroundColor Green
Write-Host "📊 Size: $sizeGB GB" -ForegroundColor Green
Write-Host "📅 Downloaded: $($file.LastWriteTime)" -ForegroundColor Green
Write-Host ""

if ($file.Length -lt 4000000000) {  # Less than 4GB
    Write-Host "⚠️  Warning: File size seems small. Download may be incomplete." -ForegroundColor Yellow
    Write-Host "Expected size: ~5.4 GB" -ForegroundColor Yellow
    Write-Host ""
}

# Test 1: Basic inference
Write-Host "🧪 Test 1: Basic Inference" -ForegroundColor Cyan
Write-Host "Running: zig build cli -- inference --model ./models/phi2.bin --prompt 'What is artificial intelligence?'" -ForegroundColor Gray
Write-Host ""

# Test 2: Interactive mode
Write-Host "🧪 Test 2: Interactive Chat Mode" -ForegroundColor Cyan
Write-Host "To test interactive mode, run:" -ForegroundColor Gray
Write-Host "zig build cli -- interactive --model ./models/phi2.bin --max-tokens 400 --verbose" -ForegroundColor White
Write-Host ""

# Test 3: HTTP Server
Write-Host "🧪 Test 3: HTTP Server Mode" -ForegroundColor Cyan
Write-Host "To start HTTP server, run:" -ForegroundColor Gray
Write-Host "zig build cli -- server --model ./models/phi2.bin --port 8080" -ForegroundColor White
Write-Host ""

# Performance expectations
Write-Host "📈 Expected Performance:" -ForegroundColor Cyan
Write-Host "• Model: Phi-2 (2.7B parameters)" -ForegroundColor White
Write-Host "• Memory: ~6GB RAM required" -ForegroundColor White
Write-Host "• Speed: Varies by hardware" -ForegroundColor White
Write-Host "• Quality: High reasoning capabilities" -ForegroundColor White
Write-Host ""

# Sample prompts
Write-Host "💡 Sample Prompts to Try:" -ForegroundColor Cyan
Write-Host "• 'Explain quantum computing in simple terms'" -ForegroundColor White
Write-Host "• 'Write a Python function to sort a list'" -ForegroundColor White
Write-Host "• 'What are the benefits of renewable energy?'" -ForegroundColor White
Write-Host "• 'Solve this math problem: 2x + 5 = 15'" -ForegroundColor White
Write-Host ""

Write-Host "🚀 Ready to test Phi-2! Choose a test above to run." -ForegroundColor Green

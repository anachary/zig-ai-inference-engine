# PowerShell script to download GPT-2 ONNX model
Write-Host "🤖 Downloading GPT-2 ONNX Model for zig-ai-platform" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Create models directory
if (!(Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
    Write-Host "📁 Created models directory" -ForegroundColor Yellow
}

# GPT-2 ONNX model URLs to try
$urls = @(
    @{
        Name = "GPT-2 from ONNX Model Zoo"
        Url = "https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-10.onnx"
        Output = "models\gpt2.onnx"
    },
    @{
        Name = "GPT-2 from Microsoft ONNX Runtime"
        Url = "https://github.com/microsoft/onnxruntime/raw/main/onnxruntime/test/testdata/gpt2_past.onnx"
        Output = "models\gpt2_past.onnx"
    }
)

$downloaded = $false

foreach ($model in $urls) {
    Write-Host ""
    Write-Host "🔄 Trying: $($model.Name)" -ForegroundColor Cyan
    Write-Host "   URL: $($model.Url)" -ForegroundColor Gray
    
    try {
        # Download with progress
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($model.Url, $model.Output)
        
        # Check if file exists and has reasonable size
        if (Test-Path $model.Output) {
            $fileSize = (Get-Item $model.Output).Length
            $fileSizeMB = [math]::Round($fileSize / 1MB, 2)
            
            if ($fileSize -gt 1MB) {
                Write-Host "✅ Successfully downloaded: $($model.Output)" -ForegroundColor Green
                Write-Host "📊 File size: $fileSizeMB MB" -ForegroundColor Green
                $downloaded = $true
                break
            } else {
                Write-Host "⚠️  File too small ($fileSizeMB MB), trying next source..." -ForegroundColor Yellow
                Remove-Item $model.Output -ErrorAction SilentlyContinue
            }
        }
    }
    catch {
        Write-Host "❌ Failed to download: $($_.Exception.Message)" -ForegroundColor Red
    }
}

if ($downloaded) {
    Write-Host ""
    Write-Host "🎉 GPT-2 ONNX model ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 Next steps:" -ForegroundColor Cyan
    Write-Host "   1. Build the project: zig build" -ForegroundColor White
    Write-Host "   2. Test GPT-2: .\zig-out\bin\zig-ai.exe chat --model models\gpt2.onnx" -ForegroundColor White
    Write-Host ""
    Write-Host "🎯 Expected output with GPT-2:" -ForegroundColor Cyan
    Write-Host "   🔍 Detected architecture: gpt, processor: text_generation" -ForegroundColor Gray
    Write-Host "   🎯 Detected vocabulary logits tensor - using text generation" -ForegroundColor Gray
    Write-Host "   🤖 Neural network generated response: [Actual GPT-2 text]" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "❌ Could not download GPT-2 ONNX model automatically." -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Manual alternatives:" -ForegroundColor Yellow
    Write-Host "   1. Install Python: pip install torch transformers onnx" -ForegroundColor White
    Write-Host "   2. Run converter: python scripts\download_gpt2.py" -ForegroundColor White
    Write-Host "   3. Download manually from:" -ForegroundColor White
    Write-Host "      https://github.com/onnx/models/tree/main/text/machine_comprehension/gpt-2" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

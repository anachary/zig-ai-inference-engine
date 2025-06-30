# Phi-2 Model Download Script (Safetensors Format)
Write-Host "🧠 Downloading Phi-2 Model (Safetensors Format)" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Create models directory
Write-Host "📁 Creating models directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "models" | Out-Null

# Download progress function
function Show-DownloadProgress {
    param($Activity, $Status, $PercentComplete)
    Write-Progress -Activity $Activity -Status $Status -PercentComplete $PercentComplete
}

# Set progress preference
$ProgressPreference = 'Continue'

Write-Host "🌐 Downloading Phi-2 model files from Hugging Face..." -ForegroundColor Green
Write-Host ""

# Download main model files (Safetensors format)
Write-Host "📥 Downloading model-00001-of-00002.safetensors (5GB)..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri "https://huggingface.co/microsoft/phi-2/resolve/main/model-00001-of-00002.safetensors" -OutFile "models/model-00001-of-00002.safetensors" -UseBasicParsing
    Write-Host "✅ Downloaded model-00001-of-00002.safetensors" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to download model-00001-of-00002.safetensors: $_" -ForegroundColor Red
}

Write-Host "📥 Downloading model-00002-of-00002.safetensors (564MB)..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri "https://huggingface.co/microsoft/phi-2/resolve/main/model-00002-of-00002.safetensors" -OutFile "models/model-00002-of-00002.safetensors" -UseBasicParsing
    Write-Host "✅ Downloaded model-00002-of-00002.safetensors" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to download model-00002-of-00002.safetensors: $_" -ForegroundColor Red
}

# Download model index
Write-Host "📥 Downloading model index..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri "https://huggingface.co/microsoft/phi-2/resolve/main/model.safetensors.index.json" -OutFile "models/model.safetensors.index.json" -UseBasicParsing
    Write-Host "✅ Downloaded model.safetensors.index.json" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to download model.safetensors.index.json: $_" -ForegroundColor Red
}

# Download configuration files
Write-Host "📥 Downloading configuration files..." -ForegroundColor Yellow
$configFiles = @(
    @{url="https://huggingface.co/microsoft/phi-2/resolve/main/config.json"; file="config.json"},
    @{url="https://huggingface.co/microsoft/phi-2/resolve/main/generation_config.json"; file="generation_config.json"}
)

foreach ($config in $configFiles) {
    try {
        Invoke-WebRequest -Uri $config.url -OutFile "models/$($config.file)" -UseBasicParsing
        Write-Host "✅ Downloaded $($config.file)" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to download $($config.file): $_" -ForegroundColor Red
    }
}

# Download tokenizer files
Write-Host "📥 Downloading tokenizer files..." -ForegroundColor Yellow
$tokenizerFiles = @(
    @{url="https://huggingface.co/microsoft/phi-2/resolve/main/tokenizer.json"; file="tokenizer.json"},
    @{url="https://huggingface.co/microsoft/phi-2/resolve/main/tokenizer_config.json"; file="tokenizer_config.json"},
    @{url="https://huggingface.co/microsoft/phi-2/resolve/main/vocab.json"; file="vocab.json"},
    @{url="https://huggingface.co/microsoft/phi-2/resolve/main/merges.txt"; file="merges.txt"},
    @{url="https://huggingface.co/microsoft/phi-2/resolve/main/added_tokens.json"; file="added_tokens.json"},
    @{url="https://huggingface.co/microsoft/phi-2/resolve/main/special_tokens_map.json"; file="special_tokens_map.json"}
)

foreach ($tokenizer in $tokenizerFiles) {
    try {
        Invoke-WebRequest -Uri $tokenizer.url -OutFile "models/$($tokenizer.file)" -UseBasicParsing
        Write-Host "✅ Downloaded $($tokenizer.file)" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to download $($tokenizer.file): $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📊 Verifying downloads..." -ForegroundColor Cyan

# Check downloaded files
$downloadedFiles = Get-ChildItem "models" -File
$totalSize = ($downloadedFiles | Measure-Object -Property Length -Sum).Sum
$totalSizeGB = [math]::Round($totalSize / 1GB, 2)

Write-Host "📁 Downloaded files:" -ForegroundColor Green
foreach ($file in $downloadedFiles) {
    $sizeStr = if ($file.Length -gt 1GB) {
        "$([math]::Round($file.Length / 1GB, 2)) GB"
    } elseif ($file.Length -gt 1MB) {
        "$([math]::Round($file.Length / 1MB, 2)) MB"
    } else {
        "$([math]::Round($file.Length / 1KB, 2)) KB"
    }
    Write-Host "  • $($file.Name) - $sizeStr" -ForegroundColor White
}

Write-Host ""
Write-Host "📊 Total downloaded: $totalSizeGB GB" -ForegroundColor Green

# Check if main model files exist
$mainFiles = @("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json")
$allMainFilesExist = $true

foreach ($file in $mainFiles) {
    if (-not (Test-Path "models/$file")) {
        Write-Host "❌ Missing critical file: $file" -ForegroundColor Red
        $allMainFilesExist = $false
    }
}

if ($allMainFilesExist) {
    Write-Host ""
    Write-Host "🎉 Phi-2 model download complete!" -ForegroundColor Green
    Write-Host "🧪 Ready to test with your Zig AI Inference Engine!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "💡 Test commands:" -ForegroundColor Yellow
    Write-Host "zig build cli -- interactive --model ./models --max-tokens 400 --verbose" -ForegroundColor White
    Write-Host "zig build cli -- inference --model ./models --prompt 'What is AI?'" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "⚠️ Download incomplete. Some files are missing." -ForegroundColor Yellow
    Write-Host "Please re-run the script or download missing files manually." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📚 Model info:" -ForegroundColor Cyan
Write-Host "• Model: Microsoft Phi-2" -ForegroundColor White
Write-Host "• Parameters: 2.7B" -ForegroundColor White
Write-Host "• Format: Safetensors" -ForegroundColor White
Write-Host "• License: MIT" -ForegroundColor White

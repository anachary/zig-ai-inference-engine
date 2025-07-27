@echo off
echo 🤖 Downloading GPT-2 ONNX Model for zig-ai-platform
echo ================================================

if not exist models mkdir models

echo.
echo 📥 Downloading GPT-2 ONNX model...
echo This may take a few minutes depending on your internet connection.
echo.

REM Try to download from ONNX Model Zoo
echo 🔄 Attempting download from ONNX Model Zoo...
curl -L -o models\gpt2.onnx "https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-10.onnx"

if exist models\gpt2.onnx (
    echo ✅ Successfully downloaded GPT-2 ONNX model!
    echo.
    echo 📁 Location: models\gpt2.onnx
    for %%A in (models\gpt2.onnx) do echo 📊 Size: %%~zA bytes
    echo.
    echo 🚀 Ready to test with zig-ai-platform:
    echo    zig build
    echo    .\zig-out\bin\zig-ai.exe chat --model models\gpt2.onnx
    echo.
) else (
    echo ❌ Download failed. Trying alternative method...
    echo.
    echo 💡 Alternative options:
    echo    1. Install Python dependencies: pip install torch transformers onnx
    echo    2. Run full conversion: python scripts\download_gpt2.py
    echo    3. Download manually from: https://github.com/onnx/models/tree/main/text/machine_comprehension/gpt-2
    echo.
)

pause

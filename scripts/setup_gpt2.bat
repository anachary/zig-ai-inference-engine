@echo off
echo 🤖 Setting up GPT-2 for zig-ai-platform
echo ========================================

echo.
echo 📦 Installing Python dependencies...
pip install -r scripts/requirements.txt

echo.
echo 🚀 Downloading and converting GPT-2...
python scripts/download_gpt2.py

echo.
echo ✅ Setup complete! 
echo.
echo 🎯 To test GPT-2 with zig-ai-platform:
echo    zig build
echo    .\zig-out\bin\zig-ai.exe chat --model models\gpt2.onnx
echo.
pause

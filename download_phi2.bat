@echo off
echo 🧠 Downloading Phi-2 Model (Safetensors Format)
echo =================================================
echo.

REM Create models directory
if not exist "models" mkdir models

echo 📥 Downloading main model file (5GB)...
echo This may take 10-60 minutes depending on your internet speed.
echo.

REM Download main model file using PowerShell
powershell -Command "& {$ProgressPreference = 'Continue'; Invoke-WebRequest -Uri 'https://huggingface.co/microsoft/phi-2/resolve/main/model-00001-of-00002.safetensors' -OutFile 'models/model-00001-of-00002.safetensors' -UseBasicParsing}"

if %ERRORLEVEL% EQU 0 (
    echo ✅ Main model file downloaded successfully!
) else (
    echo ❌ Failed to download main model file
    pause
    exit /b 1
)

echo.
echo 📥 Downloading second model file (564MB)...
powershell -Command "& {$ProgressPreference = 'Continue'; Invoke-WebRequest -Uri 'https://huggingface.co/microsoft/phi-2/resolve/main/model-00002-of-00002.safetensors' -OutFile 'models/model-00002-of-00002.safetensors' -UseBasicParsing}"

if %ERRORLEVEL% EQU 0 (
    echo ✅ Second model file downloaded successfully!
) else (
    echo ❌ Failed to download second model file
)

echo.
echo 📥 Downloading model index...
powershell -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/microsoft/phi-2/resolve/main/model.safetensors.index.json' -OutFile 'models/model.safetensors.index.json' -UseBasicParsing}"

echo.
echo 📥 Downloading configuration files...
powershell -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/microsoft/phi-2/resolve/main/config.json' -OutFile 'models/config.json' -UseBasicParsing}"
powershell -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/microsoft/phi-2/resolve/main/tokenizer.json' -OutFile 'models/tokenizer.json' -UseBasicParsing}"
powershell -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/microsoft/phi-2/resolve/main/vocab.json' -OutFile 'models/vocab.json' -UseBasicParsing}"

echo.
echo 📊 Checking downloaded files...
dir models /b

echo.
echo 🎉 Phi-2 download complete!
echo.
echo 🧪 Test with your Zig AI Inference Engine:
echo zig build cli -- interactive --model ./models --max-tokens 400
echo.
pause

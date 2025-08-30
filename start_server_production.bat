@echo off
echo Starting Stable Diffusion Server (Production)...
cd /d D:\code\stable-diffusion-server

:: Set environment variables
if exist "secrets\google-credentials.json" (
    set GOOGLE_APPLICATION_CREDENTIALS=secrets\google-credentials.json
    echo Google Cloud credentials set.
)

:: Optional: Set model paths
:: set DF11_MODEL_PATH=DFloat11/FLUX.1-schnell-DF11
:: set CONTROLNET_LORA=black-forest-labs/flux-controlnet-line-lora
:: set LOAD_LCM_LORA=1

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting production server with gunicorn...
gunicorn -k uvicorn.workers.UvicornWorker -b :8000 main:app --timeout 600 -w 1

echo Server stopped.
pause
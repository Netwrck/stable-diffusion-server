@echo off
cd /d "D:\code\stable-diffusion-server"
REM set GOOGLE_APPLICATION_CREDENTIALS=secrets\google-credentials.json
set PYTHONPATH=.
set HF_HUB_DISABLE_XET=1

REM Load R2 credentials from .secretbashrc if it exists
if exist "%USERPROFILE%\.secretbashrc" (
    echo Loading R2 credentials from ~/.secretbashrc...
    for /f "tokens=1,* delims==" %%a in ('type "%USERPROFILE%\.secretbashrc" ^| findstr /B /C:"export R2_" /C:"export CLOUDFLARE_R2_" /C:"export AWS_"') do (
        for /f "tokens=2" %%i in ("%%a") do set "%%i=%%~b"
    )
)

REM Normalize the names used by the shared shell secrets to boto3's names.
if defined R2_ENDPOINT set "R2_ENDPOINT_URL=%R2_ENDPOINT%"
if defined CLOUDFLARE_R2_ACCESS_KEY_ID set "AWS_ACCESS_KEY_ID=%CLOUDFLARE_R2_ACCESS_KEY_ID%"
if defined CLOUDFLARE_R2_SECRET_ACCESS_KEY set "AWS_SECRET_ACCESS_KEY=%CLOUDFLARE_R2_SECRET_ACCESS_KEY%"

REM Or set them manually here (uncomment and fill in):
REM set R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
REM set AWS_ACCESS_KEY_ID=your_access_key_id
REM set AWS_SECRET_ACCESS_KEY=your_secret_access_key

:restart
echo Starting Stable Diffusion Server at %date% %time%
"%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" run uvicorn --port 8000 --timeout-keep-alive 600 --workers 1 --backlog 1 --limit-concurrency 2 low_ram_main:app
echo Server stopped at %date% %time%
echo Restarting in 5 seconds...
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -Command "Start-Sleep -Seconds 5"
goto restart

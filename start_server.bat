@echo off
echo Starting Stable Diffusion Server...
cd /d D:\code\stable-diffusion-server

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting server with uvicorn...
uvicorn main:app --port 8000 --timeout-keep-alive 600 --workers 1 --backlog 1 --limit-concurrency 4

echo Server stopped.
pause
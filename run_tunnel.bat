@echo off
cd /d "D:\code\stable-diffusion-server"
:restart
echo Starting Cloudflare Tunnel at %date% %time%
"C:\Users\lee_p\Downloads\cloudflared.exe" tunnel --url 127.0.0.1:8000 --protocol http2 --name images3
echo Tunnel stopped at %date% %time%
echo Restarting tunnel in 5 seconds...
timeout /t 5 /nobreak
goto restart

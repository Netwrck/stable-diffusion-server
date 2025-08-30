@echo off
echo Creating Windows Task Scheduler task for Stable Diffusion Server...

:: Create the task to run at system startup
schtasks /create /tn "StableDiffusionServer" ^
    /tr "D:\code\stable-diffusion-server\start_server.bat" ^
    /sc onstart ^
    /ru SYSTEM ^
    /rl highest ^
    /f

:: Alternative: Run as current user at logon
:: schtasks /create /tn "StableDiffusionServer" ^
::     /tr "D:\code\stable-diffusion-server\start_server.bat" ^
::     /sc onlogon ^
::     /rl highest ^
::     /f

echo Task created successfully!
echo.
echo To manage the task:
echo   Start:  schtasks /run /tn "StableDiffusionServer"
echo   Stop:   schtasks /end /tn "StableDiffusionServer"
echo   Delete: schtasks /delete /tn "StableDiffusionServer" /f
echo   Query:  schtasks /query /tn "StableDiffusionServer"
echo.
pause
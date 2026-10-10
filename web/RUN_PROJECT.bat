@echo off
cd /d "%~dp0"

if not exist "backend\.venv\Scripts\python.exe" (
    echo Backend environment is missing.
    echo Run SETUP_WINDOWS.bat first.
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo Frontend packages are missing.
    echo Run SETUP_WINDOWS.bat first.
    pause
    exit /b 1
)

start "Smart Traffic Backend" cmd /k call "%~dp0backend\run_backend.bat"
start "Smart Traffic Frontend" cmd /k call "%~dp0frontend\run_frontend.bat"

timeout /t 4 /nobreak >nul
start "" "http://localhost:3000"

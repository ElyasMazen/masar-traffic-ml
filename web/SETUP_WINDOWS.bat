@echo off
setlocal
cd /d "%~dp0"

echo ======================================================
echo Smart City Traffic Intelligence - One-Time Setup
echo ======================================================
echo.

echo [1/4] Preparing Python environment...
if not exist "backend\.venv\Scripts\python.exe" (
    python -m venv "backend\.venv"
    if errorlevel 1 goto :error
)

"backend\.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo [2/4] Installing backend packages...
"backend\.venv\Scripts\python.exe" -m pip install -r "backend\requirements.txt"
if errorlevel 1 goto :error

echo.
echo [3/4] Training local traffic models (NYC + Jeddah)...
"backend\.venv\Scripts\python.exe" "backend\scripts\train_model.py"
if errorlevel 1 goto :error
rem Jeddah training runs from the repo root (one level up) in its own environment.
if not exist "..\.venv\Scripts\python.exe" (
    python -m venv "..\.venv"
    if errorlevel 1 goto :error
)
"..\.venv\Scripts\python.exe" -m pip install -r "..\requirements.txt"
if errorlevel 1 goto :error
"backend\.venv\Scripts\python.exe" "backend\scripts\build_jeddah.py"
if errorlevel 1 goto :error

echo.
echo [4/4] Installing frontend packages...
cd /d "%~dp0frontend"
call npm install
if errorlevel 1 goto :error
cd /d "%~dp0"

echo.
echo ======================================================
echo Setup completed successfully.
echo Now run RUN_PROJECT.bat
echo ======================================================
pause
exit /b 0

:error
echo.
echo SETUP FAILED. Read the error above.
pause
exit /b 1

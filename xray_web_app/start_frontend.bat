@echo off
echo ==========================================
echo   X-Ray Analyzer Frontend (Windows)
echo ==========================================
echo.

cd /d "%~dp0frontend"

echo [1/3] Checking Node.js installation...
where node >nul 2>nul
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH.
    echo Please install Node.js LTS from https://nodejs.org/
    pause
    exit /b 1
)

echo [2/3] Installing npm dependencies...
echo       This may take a few minutes on first run...
call npm install

echo [3/3] Starting React development server...
echo.
echo ==========================================
echo   Frontend running at http://localhost:3000
echo   Press Ctrl+C to stop
echo ==========================================
echo.

call npm start

pause

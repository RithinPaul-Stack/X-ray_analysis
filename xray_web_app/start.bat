@echo off
echo ==========================================
echo   X-Ray Analyzer - Full Application
echo   (Windows CPU Edition)
echo ==========================================
echo.
echo This will start both Backend and Frontend servers.
echo.

:: Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10-3.12 from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check Node.js
where node >nul 2>nul
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH.
    echo Please install Node.js LTS from https://nodejs.org/
    pause
    exit /b 1
)

echo Starting Backend server...
start "X-Ray Backend" cmd /k "%~dp0start_backend.bat"

:: Wait for backend to initialize
echo Waiting for backend to start (10 seconds)...
timeout /t 10 /nobreak > nul

echo Starting Frontend server...
start "X-Ray Frontend" cmd /k "%~dp0start_frontend.bat"

echo.
echo ==========================================
echo   Both servers are starting!
echo.
echo   Backend API:  http://localhost:8080
echo   Frontend UI:  http://localhost:3000
echo.
echo   The frontend will open in your browser
echo   automatically when ready.
echo.
echo   To stop: Close the Backend and Frontend
echo   terminal windows.
echo ==========================================
echo.
pause

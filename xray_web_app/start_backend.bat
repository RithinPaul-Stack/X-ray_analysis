@echo off
echo ==========================================
echo   X-Ray Analyzer Backend (Windows CPU)
echo ==========================================
echo.

cd /d "%~dp0backend"

echo [1/4] Checking Python virtual environment...
if not exist "venv" (
    echo       Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Is Python installed?
        echo Please install Python 3.10-3.12 from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies (CPU-only PyTorch)...
echo       This may take a few minutes on first run...
if exist "requirements-windows.txt" (
    pip install -r requirements-windows.txt --quiet
) else (
    echo       Installing PyTorch CPU version...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    pip install -r requirements.txt --quiet
)

echo [4/4] Starting Flask server...
echo.
echo ==========================================
echo   Server running at http://localhost:8080
echo   Press Ctrl+C to stop
echo ==========================================
echo.

python app.py

pause

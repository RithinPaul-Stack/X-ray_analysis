Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  X-Ray Analyzer Backend (Windows CPU)" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "$PSScriptRoot\backend"

Write-Host "[1/4] Checking Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    Write-Host "      Creating virtual environment..." -ForegroundColor Gray
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        Write-Host "Please install Python 3.10-3.12 from https://www.python.org/downloads/" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host "[3/4] Installing dependencies (CPU-only PyTorch)..." -ForegroundColor Yellow
Write-Host "      This may take a few minutes on first run..." -ForegroundColor Gray
if (Test-Path "requirements-windows.txt") {
    pip install -r requirements-windows.txt --quiet
} else {
    Write-Host "      Installing PyTorch CPU version..." -ForegroundColor Gray
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    pip install -r requirements.txt --quiet
}

Write-Host "[4/4] Starting Flask server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Server running at http://localhost:8080" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

python app.py

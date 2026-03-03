Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  X-Ray Analyzer Frontend (Windows)" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "$PSScriptRoot\frontend"

Write-Host "[1/3] Checking Node.js installation..." -ForegroundColor Yellow
$nodeCheck = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCheck) {
    Write-Host "ERROR: Node.js is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Node.js LTS from https://nodejs.org/" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[2/3] Installing npm dependencies..." -ForegroundColor Yellow
Write-Host "      This may take a few minutes on first run..." -ForegroundColor Gray
npm install

Write-Host "[3/3] Starting React development server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Frontend running at http://localhost:3000" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

npm start

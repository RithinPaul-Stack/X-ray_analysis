# X-Ray Web App - Windows Porting Guide (CPU-Only)

This document provides a detailed plan for porting the X-Ray Analysis Web Application from macOS to **Windows 10/11 (CPU-only, no NVIDIA GPU)**.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start Summary](#quick-start-summary)
4. [Detailed Changes Required](#detailed-changes-required)
5. [Windows Startup Scripts](#windows-startup-scripts)
6. [Step-by-Step Installation](#step-by-step-installation)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)
9. [Verification Checklist](#verification-checklist)

---

## Overview

### Application Components
| Component | Technology | Notes |
|-----------|------------|-------|
| Backend | Flask (Python) + PyTorch | ML inference |
| Frontend | React.js | User interface |
| Database | SQLite | Analysis history |
| ML Models | TorchXRayVision DenseNet-121 | 3-way classifier (TB/Pneumonia/Normal) |

### Platform Migration
| Aspect | macOS (Current) | Windows (Target) |
|--------|-----------------|------------------|
| OS | macOS | Windows 10/11 |
| GPU Acceleration | Apple MPS | **CPU only** |
| Shell | Bash (.sh) | CMD/PowerShell (.bat/.ps1) |
| Python venv activation | `source venv/bin/activate` | `venv\Scripts\activate.bat` |

---

## Prerequisites

### Required Software (Windows)

| Software | Version | Download | Notes |
|----------|---------|----------|-------|
| **Python** | 3.10 - 3.12 (64-bit) | https://www.python.org/downloads/ | **Must check "Add to PATH"** |
| **Node.js** | 18.x or 20.x LTS | https://nodejs.org/ | LTS version recommended |
| **Git** | Latest | https://git-scm.com/download/win | Optional, for cloning |

### NOT Required
- NVIDIA GPU
- CUDA Toolkit
- cuDNN

---

## Quick Start Summary

### Files to Create (New)
```
xray_web_app/
├── start.bat                    # Launch both servers
├── start_backend.bat            # Backend startup (CMD)
├── start_frontend.bat           # Frontend startup (CMD)
├── start_backend.ps1            # Backend startup (PowerShell)
├── start_frontend.ps1           # Frontend startup (PowerShell)
└── backend/
    └── requirements-windows.txt # CPU-only PyTorch dependencies
```

### Files to Modify (Existing)
| File | Change |
|------|--------|
| `backend/xray_analyzer.py` | Simplify device detection (CPU-only) |
| `backend/gradcam_analyzer.py` | Simplify device detection (CPU-only) |
| `backend/tb_fine_tuning/config.py` | Fix hardcoded paths + device detection |

---

## Detailed Changes Required

### 1. Device Detection Changes

Since there's no GPU, we can simplify the device detection to always use CPU.

#### File: `backend/xray_analyzer.py` (lines 77-82)

**Current (macOS with MPS):**
```python
if torch.backends.mps.is_available():
    self.device = torch.device("mps")
    print("  Using Apple Silicon GPU (MPS)")
else:
    self.device = torch.device("cpu")
    print("  Using CPU")
```

**Windows CPU-only version:**
```python
# CPU-only for Windows (no GPU)
self.device = torch.device("cpu")
print("  Using CPU")
```

**OR cross-platform version (recommended):**
```python
# Cross-platform device detection
if torch.cuda.is_available():
    self.device = torch.device("cuda")
    print("  Using NVIDIA GPU (CUDA)")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    self.device = torch.device("mps")
    print("  Using Apple Silicon GPU (MPS)")
else:
    self.device = torch.device("cpu")
    print("  Using CPU")
```

#### File: `backend/gradcam_analyzer.py` (lines 37-40)

**Current:**
```python
if torch.backends.mps.is_available():
    self.device = torch.device("mps")
else:
    self.device = torch.device("cpu")
```

**Windows CPU-only version:**
```python
self.device = torch.device("cpu")
```

#### File: `backend/tb_fine_tuning/config.py` (lines 85-91)

**Current:**
```python
if torch.backends.mps.is_available():
    self.device = "mps"
elif torch.cuda.is_available():
    self.device = "cuda"
else:
    self.device = "cpu"
```

**Windows CPU-only version:**
```python
self.device = "cpu"
```

---

### 2. Fix Hardcoded macOS Paths

#### File: `backend/tb_fine_tuning/config.py` (lines 17-24)

**Current (hardcoded macOS paths):**
```python
data_root: str = "/Users/rithinreddy/Documents/TB_Chest_Radiography_Database"
pneumonia_data_root: str = "/Users/rithinreddy/medic_ai_experiments/chest_xray/chest_xray/train"
```

**Windows-compatible version:**
```python
# Dataset paths - uses environment variables or defaults to user's Documents folder
data_root: str = os.environ.get(
    "TB_DATA_ROOT",
    os.path.join(os.path.expanduser("~"), "Documents", "TB_Chest_Radiography_Database")
)

pneumonia_data_root: str = os.environ.get(
    "PNEUMONIA_DATA_ROOT",
    os.path.join(os.path.expanduser("~"), "Documents", "chest_xray", "train")
)
```

> **Note:** These paths are only needed if you want to **retrain** the model. For inference (running the app), only the pre-trained checkpoint files are needed.

---

### 3. Create Windows Requirements File

#### File: `backend/requirements-windows.txt` (NEW)

```
# PyTorch CPU-only for Windows (no NVIDIA GPU)
--index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
torchvision>=0.15.0

# Core dependencies (cross-platform compatible)
flask>=2.3.0
flask-cors>=4.0.0
torchxrayvision>=0.0.37
scikit-image>=0.21.0
numpy>=1.24.0
pillow>=10.0.0
pydicom>=2.4.0
ultralytics>=8.0.0
opencv-python>=4.8.0
reportlab>=4.0.0
werkzeug>=2.3.0
tensorboard>=2.14.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## Windows Startup Scripts

### `start_backend.bat`

```batch
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
        pause
        exit /b 1
    )
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies (CPU-only PyTorch)...
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
echo   Server starting at http://localhost:8080
echo   Press Ctrl+C to stop
echo ==========================================
echo.

python app.py

pause
```

### `start_frontend.bat`

```batch
@echo off
echo ==========================================
echo   X-Ray Analyzer Frontend (Windows)
echo ==========================================
echo.

cd /d "%~dp0frontend"

echo [1/2] Installing npm dependencies...
call npm install

echo [2/2] Starting React development server...
echo.
echo ==========================================
echo   Frontend starting at http://localhost:3000
echo   Press Ctrl+C to stop
echo ==========================================
echo.

call npm start

pause
```

### `start.bat` (Launch Both)

```batch
@echo off
echo ==========================================
echo   X-Ray Analyzer - Full Application
echo ==========================================
echo.
echo Starting Backend and Frontend servers...
echo.

:: Start backend in new window
start "X-Ray Backend" cmd /k "%~dp0start_backend.bat"

:: Wait for backend to initialize
echo Waiting for backend to start...
timeout /t 8 /nobreak > nul

:: Start frontend in new window
start "X-Ray Frontend" cmd /k "%~dp0start_frontend.bat"

echo.
echo ==========================================
echo   Both servers are starting!
echo.
echo   Backend API:  http://localhost:8080
echo   Frontend UI:  http://localhost:3000
echo.
echo   Close this window when done.
echo ==========================================
pause
```

### `start_backend.ps1` (PowerShell Alternative)

```powershell
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  X-Ray Analyzer Backend (Windows CPU)" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "$PSScriptRoot\backend"

Write-Host "[1/4] Checking Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    Write-Host "      Creating virtual environment..." -ForegroundColor Gray
    python -m venv venv
}

Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host "[3/4] Installing dependencies (CPU-only PyTorch)..." -ForegroundColor Yellow
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
Write-Host "  Server starting at http://localhost:8080" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

python app.py
```

### `start_frontend.ps1` (PowerShell Alternative)

```powershell
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  X-Ray Analyzer Frontend (Windows)" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "$PSScriptRoot\frontend"

Write-Host "[1/2] Installing npm dependencies..." -ForegroundColor Yellow
npm install

Write-Host "[2/2] Starting React development server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Frontend starting at http://localhost:3000" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

npm start
```

---

## Step-by-Step Installation

### Step 1: Install Python

1. Download Python 3.10-3.12 (64-bit) from https://www.python.org/downloads/
2. Run the installer
3. **IMPORTANT:** Check ✅ "Add Python to PATH"
4. Click "Install Now"
5. Verify installation:
   ```cmd
   python --version
   ```

### Step 2: Install Node.js

1. Download Node.js LTS from https://nodejs.org/
2. Run the installer (accept defaults)
3. Verify installation:
   ```cmd
   node --version
   npm --version
   ```

### Step 3: Copy Project to Windows

Copy the entire `xray_web_app` folder to your Windows machine, for example:
```
C:\Users\YourName\Documents\xray_web_app\
```

### Step 4: Create Windows Startup Scripts

Create the `.bat` files listed above in the `xray_web_app` folder.

### Step 5: Apply Code Changes

Edit the Python files to update device detection (see [Detailed Changes Required](#detailed-changes-required)).

### Step 6: Copy Model Weights

Ensure these files exist:
```
backend\tb_model_weights\checkpoints\checkpoint_best.pth
backend\tb_model_weights\checkpoints\checkpoint_latest.pth
```

### Step 7: Run the Application

**Option A: Double-click `start.bat`**
- Opens two terminal windows (backend + frontend)
- Wait for both to fully start

**Option B: Run separately**
```cmd
# Terminal 1 - Backend
start_backend.bat

# Terminal 2 - Frontend
start_frontend.bat
```

### Step 8: Access the Application

Open your browser to:
- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:8080

---

## Performance Considerations

### CPU-Only Performance

Running on CPU without GPU acceleration will be **slower** than with MPS (Mac) or CUDA (NVIDIA):

| Operation | Expected Time (CPU) | With GPU |
|-----------|---------------------|----------|
| Model loading | 10-30 seconds | 5-15 seconds |
| Single X-ray analysis | 3-8 seconds | 0.5-2 seconds |
| GradCAM heatmap | 5-15 seconds | 1-3 seconds |
| PDF generation | 2-5 seconds | Same |

### Optimization Tips

1. **First run is slowest** - Models are cached after first load
2. **Close other applications** - Free up RAM for PyTorch
3. **Use SSD storage** - Faster model loading
4. **8GB+ RAM recommended** - ML models need memory

### Minimum System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB free | 10 GB free (SSD) |
| Display | 1366x768 | 1920x1080 |

---

## Troubleshooting

### Common Issues

#### 1. "python is not recognized as an internal command"

**Cause:** Python not added to PATH during installation.

**Solution:**
1. Reinstall Python and check "Add Python to PATH"
2. Or manually add to PATH:
   - Search "Environment Variables" in Windows
   - Edit "Path" under User variables
   - Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python312\`

#### 2. "npm is not recognized"

**Cause:** Node.js not installed or not in PATH.

**Solution:** Reinstall Node.js from https://nodejs.org/

#### 3. "Module not found: torch"

**Cause:** PyTorch not installed or wrong version.

**Solution:**
```cmd
venv\Scripts\activate.bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 4. "Port 8080/3000 already in use"

**Cause:** Another application using the port.

**Solution:**
```cmd
# Find what's using port 8080
netstat -ano | findstr :8080

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F
```

#### 5. PowerShell script won't run

**Cause:** Execution policy blocks scripts.

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 6. "Model file not found"

**Cause:** Checkpoint files missing or in wrong location.

**Solution:** Verify files exist at:
```
backend\tb_model_weights\checkpoints\checkpoint_best.pth
```

#### 7. Analysis is very slow

**Cause:** Running on CPU is inherently slower.

**Solution:**
- This is expected for CPU-only operation
- First analysis is slowest (model loading)
- Subsequent analyses will be faster
- Ensure no other heavy applications running

---

## Verification Checklist

After porting, verify each feature works:

- [ ] Backend starts without errors (`start_backend.bat`)
- [ ] Frontend starts and loads in browser (`start_frontend.bat`)
- [ ] Can upload PNG/JPG images
- [ ] Can upload DICOM (.dcm) files
- [ ] X-ray analysis returns pathology results
- [ ] TB/Pneumonia/Normal classification displays
- [ ] GradCAM heatmap generates (Deep Analysis)
- [ ] PDF report downloads successfully
- [ ] Analysis history shows previous results
- [ ] Can delete analysis from history

---

## File Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `start.bat` | Launch both servers |
| `start_backend.bat` | Start Flask backend |
| `start_frontend.bat` | Start React frontend |
| `start_backend.ps1` | PowerShell backend startup |
| `start_frontend.ps1` | PowerShell frontend startup |
| `backend/requirements-windows.txt` | CPU-only dependencies |

### Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `backend/xray_analyzer.py` | 77-82 | CPU device detection |
| `backend/gradcam_analyzer.py` | 37-40 | CPU device detection |
| `backend/tb_fine_tuning/config.py` | 17-24, 85-91 | Fix paths + CPU detection |

---

## Creating an Installable Executable

You can package this application as a standalone Windows executable that doesn't require users to install Python or Node.js. Here are the options:

### Option 1: PyInstaller + Electron (Recommended for Desktop App)

This creates a single installable application with both backend and frontend bundled.

#### Step 1: Package Backend with PyInstaller

```bash
# In backend directory with venv activated
pip install pyinstaller

# Create single executable
pyinstaller --onefile --name xray-backend app.py \
    --hidden-import=torchxrayvision \
    --hidden-import=torch \
    --hidden-import=sklearn \
    --add-data "tb_model_weights;tb_model_weights" \
    --add-data "clinical_knowledge.py;."
```

#### Step 2: Build Frontend for Production

```bash
# In frontend directory
npm run build
```

#### Step 3: Package with Electron (optional - for native app feel)

Create an Electron wrapper that:
1. Starts the backend executable
2. Serves the React build
3. Opens a native window

### Option 2: Docker Desktop (Easiest Distribution)

Package everything in Docker containers:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/ ./backend/
COPY frontend/build/ ./frontend/build/

RUN pip install -r backend/requirements-windows.txt
EXPOSE 8080

CMD ["python", "backend/app.py"]
```

Users install Docker Desktop, then run:
```cmd
docker-compose up
```

### Option 3: NSIS Installer (Windows Installer Package)

Create a proper Windows installer (.exe) using NSIS (Nullsoft Scriptable Install System):

1. **Bundle Python embedded** - Include Python 3.11 embedded distribution
2. **Include Node.js portable** - Bundle Node.js portable version
3. **Create installer script** - NSIS script that:
   - Extracts all files
   - Creates Start Menu shortcuts
   - Registers uninstaller

#### Sample NSIS Script Structure

```nsis
!include "MUI2.nsh"

Name "X-Ray Analyzer"
OutFile "XRayAnalyzer-Setup.exe"
InstallDir "$PROGRAMFILES\XRayAnalyzer"

Section "Install"
    SetOutPath $INSTDIR

    ; Copy Python embedded
    File /r "python-embed\*.*"

    ; Copy application files
    File /r "xray_web_app\*.*"

    ; Create shortcuts
    CreateShortcut "$DESKTOP\X-Ray Analyzer.lnk" "$INSTDIR\start.bat"

    ; Write uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd
```

### Option 4: Portable App (Recommended for Simplicity)

Create a self-contained portable folder that works without installation:

#### Structure
```
XRayAnalyzer-Portable/
├── python/                    # Python embedded (python-3.11.x-embed-amd64)
├── node/                      # Node.js portable
├── app/
│   ├── backend/
│   └── frontend/
├── Start-XRayAnalyzer.bat     # Main launcher
└── README.txt
```

#### Main Launcher (Start-XRayAnalyzer.bat)
```batch
@echo off
title X-Ray Analyzer

:: Set paths to embedded Python and Node
set PYTHON=%~dp0python\python.exe
set NODE=%~dp0node\node.exe
set NPM=%~dp0node\npm.cmd

:: Start backend
start "Backend" cmd /k "%PYTHON% %~dp0app\backend\app.py"

:: Wait for backend
timeout /t 5 /nobreak > nul

:: Start frontend (or serve static build)
start "Frontend" cmd /k "cd /d %~dp0app\frontend && %NPM% start"

echo Application starting...
echo Open http://localhost:3000 in your browser
pause
```

### Comparison of Packaging Options

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **PyInstaller + Electron** | Single native app, professional feel | Complex setup, large file size (~500MB+) | Commercial distribution |
| **Docker** | Easy to distribute, consistent environment | Requires Docker Desktop installation | Technical users, servers |
| **NSIS Installer** | Proper Windows installer, Start Menu integration | Requires NSIS knowledge | Wide distribution |
| **Portable App** | No installation needed, simple | Manual folder management | Quick sharing, USB drives |

### Recommended Approach: Portable App

For your use case (CPU-only, simple distribution), the **Portable App** approach is recommended:

1. **Download Python Embedded** from python.org (embeddable package)
2. **Download Node.js Portable** or pre-build the frontend
3. **Bundle everything** in a single folder
4. **Create a simple launcher** batch file

#### Pre-built Frontend Approach (Even Simpler)

Instead of requiring Node.js at runtime, build the React app once:

```bash
# On your dev machine
cd frontend
npm run build
```

Then serve the static `build/` folder directly from Flask. This eliminates the Node.js dependency entirely for end users.

#### Modified app.py to serve static frontend:

```python
from flask import Flask, send_from_directory

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)
```

This way, the entire app runs from a single Python process - no Node.js needed!

---

## Quick Portable Distribution Checklist

To create a portable Windows distribution:

- [ ] Build React frontend (`npm run build`)
- [ ] Modify Flask to serve static frontend
- [ ] Download Python 3.11 embeddable package
- [ ] Install dependencies to a local folder
- [ ] Copy model weights
- [ ] Create launcher batch file
- [ ] Test on a clean Windows machine
- [ ] Zip and distribute

---

*Document Version: 2.1 (CPU-Only Edition + Executable Packaging)*
*Last Updated: March 2026*
*Target Platform: Windows 10/11 (No NVIDIA GPU)*

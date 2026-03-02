# X-Ray Analyzer

AI-Powered Chest X-Ray Analysis Web Application

A professional medical imaging analysis tool that uses deep learning (DenseNet-121) to detect 18 different pathologies from chest radiographs.

## Features

- **AI-Powered Analysis**: Uses torchxrayvision DenseNet-121 model trained on multiple chest X-ray datasets
- **18 Pathology Detection**: Screens for Pneumonia, Cardiomegaly, Effusion, Mass, Nodule, and more
- **Clinical Recommendations**: Provides actionable insights based on findings
- **Professional UI**: Modern, responsive interface with gradient theming
- **Local Processing**: All analysis runs locally on your machine - no data leaves your computer
- **SQLite Database**: Stores analysis history locally for review

## Screenshots

The application features:
- Drag-and-drop image upload
- Real-time analysis progress indicator
- Detailed pathology breakdown with severity levels
- Clinical recommendations section
- Analysis history tracking

## Prerequisites

- **macOS** (with Apple Silicon M1/M2 for GPU acceleration, or Intel)
- **Python 3.9+**
- **Node.js 18+** and npm
- **4GB+ RAM** recommended

## Quick Start

### 1. Start the Backend Server

Open a terminal and run:

```bash
cd xray_web_app
./start_backend.sh
```

Or manually:

```bash
cd xray_web_app/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The backend will start at: `http://localhost:5000`

### 2. Start the Frontend

Open a **new terminal** and run:

```bash
cd xray_web_app
./start_frontend.sh
```

Or manually:

```bash
cd xray_web_app/frontend
npm install
npm start
```

The frontend will open at: `http://localhost:3000`

### 3. Use the Application

1. Open `http://localhost:3000` in your browser
2. Drag & drop a chest X-ray image (JPEG/PNG)
3. Click "Analyze X-Ray"
4. View the AI-generated diagnostic report
5. Click "Clear" to analyze another image

## Project Structure

```
xray_web_app/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── xray_analyzer.py       # ML model wrapper
│   ├── database.py            # SQLite database handler
│   ├── clinical_knowledge.py  # Medical knowledge base
│   ├── requirements.txt       # Python dependencies
│   └── uploads/               # Uploaded images storage
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/        # React UI components
│   │   ├── services/          # API service layer
│   │   ├── App.jsx            # Main application
│   │   ├── App.css            # Application styles
│   │   └── index.css          # Global styles
│   └── package.json
│
├── start_backend.sh           # Backend startup script
├── start_frontend.sh          # Frontend startup script
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload and analyze X-ray image |
| GET | `/api/analysis/:id` | Get analysis by ID |
| GET | `/api/history` | Get all previous analyses |
| DELETE | `/api/analysis/:id` | Delete an analysis |
| GET | `/api/health` | API health check |

## Detected Pathologies

The model screens for 18 pathologies:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural Thickening
13. Pneumonia
14. Pneumothorax
15. Enlarged Cardiomediastinum
16. Lung Opacity
17. Lung Lesion
18. Fracture

## Technical Details

### Backend
- **Framework**: Flask with CORS support
- **ML Model**: torchxrayvision DenseNet-121 (pretrained on CheXpert, MIMIC-CXR, NIH ChestX-ray14)
- **Database**: SQLite (local file-based)
- **GPU Support**: Apple Silicon MPS acceleration when available

### Frontend
- **Framework**: React 18
- **Styling**: Custom CSS with CSS variables
- **File Upload**: react-dropzone
- **HTTP Client**: Axios

## Configuration

### Analysis Threshold

The default threshold for abnormality detection is 50% (0.5). This can be adjusted in `backend/xray_analyzer.py`:

```python
self.threshold = 0.5  # Adjust this value (0.0 to 1.0)
```

### Maximum File Size

Default maximum upload size is 16MB. Adjust in `backend/app.py`:

```python
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

## Troubleshooting

### Backend won't start

1. Ensure Python 3.9+ is installed: `python3 --version`
2. Check if port 5000 is available
3. Verify all dependencies installed: `pip install -r requirements.txt`

### Frontend won't start

1. Ensure Node.js 18+ is installed: `node --version`
2. Check if port 3000 is available
3. Reinstall dependencies: `rm -rf node_modules && npm install`

### Analysis is slow

- First analysis may be slow as the model loads into memory
- Subsequent analyses will be faster
- Using Apple Silicon Mac provides GPU acceleration

### Connection refused error

Ensure the backend is running before using the frontend.

## Disclaimer

**IMPORTANT**: This tool is for research and educational purposes only. It should NOT be used for clinical diagnosis. Always consult with qualified healthcare professionals for medical advice and interpretation of radiological findings.

## License

MIT License

## Credits

- **torchxrayvision**: https://github.com/mlmed/torchxrayvision
- **DenseNet-121**: Pretrained on multiple chest X-ray datasets

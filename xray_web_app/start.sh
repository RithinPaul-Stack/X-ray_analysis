#!/bin/bash

# =====================================================
# X-Ray Analyzer - Startup Script (Mac)
# Single command to start the entire application
# =====================================================

echo ""
echo "=========================================="
echo "  X-Ray Analyzer"
echo "  AI-Powered Diagnostic Assistant"
echo "=========================================="
echo ""

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Checking dependencies..."
pip install -r requirements.txt --quiet

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Start the Flask server
echo ""
echo "=========================================="
echo "  Starting server..."
echo "  Open http://localhost:8080 in browser"
echo "=========================================="
echo ""

python app.py

#!/bin/bash

# =====================================================
# X-Ray Analyzer - Backend Startup Script (Mac)
# =====================================================

echo "=========================================="
echo "  X-Ray Analyzer Backend"
echo "=========================================="

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Start the Flask server
echo ""
echo "Starting Flask server..."
echo "=========================================="
python app.py

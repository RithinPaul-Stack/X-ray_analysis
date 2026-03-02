#!/bin/bash

# =====================================================
# X-Ray Analyzer - Frontend Startup Script (Mac)
# =====================================================

echo "=========================================="
echo "  X-Ray Analyzer Frontend"
echo "=========================================="

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo ""
    echo "Installing npm dependencies..."
    npm install
fi

# Start the React development server
echo ""
echo "Starting React development server..."
echo "=========================================="
npm start

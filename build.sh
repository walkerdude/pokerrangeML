#!/bin/bash
# Build script for Render deployment

echo "🚀 Starting build process..."

# Upgrade pip and install build tools
echo "📦 Upgrading pip and installing build tools..."
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

# Install requirements
echo "📋 Installing Python requirements..."
pip install -r requirements.txt

echo "✅ Build completed successfully!"

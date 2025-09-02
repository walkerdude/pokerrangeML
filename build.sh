#!/bin/bash
set -e

echo "🔧 Setting up Python 3.8 environment..."

# Force Python 3.8
export PATH="/opt/python/3.8.16/bin:$PATH"
python3.8 --version

echo "📦 Upgrading pip, setuptools, and wheel..."
python3.8 -m pip install --upgrade pip setuptools wheel

echo "📚 Installing requirements (Python 3.8 compatible)..."
python3.8 -m pip install -r requirements_py38.txt

echo "✅ Build completed successfully!"

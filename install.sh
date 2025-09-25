#!/bin/bash
# Installation script for PerfCount cross-platform prediction system

set -e

echo "Installing PerfCount Cross-Platform Prediction System"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements-core.txt

# Ask user if they want advanced features
echo ""
echo "Would you like to install advanced features? (y/n)"
echo "   This includes XGBoost, CatBoost, LightGBM, and PyTorch with GPU support"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Installing advanced dependencies..."
    pip install -r requirements-advanced.txt
    echo "Advanced features installed!"
else
    echo "Skipping advanced features. You can install them later with:"
    echo "   pip install -r requirements-advanced.txt"
fi

# Ask user if they want development tools
echo ""
echo "Would you like to install development tools? (y/n)"
echo "   This includes testing, linting, and documentation tools"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
    echo "Development tools installed!"
else
    echo "Skipping development tools. You can install them later with:"
    echo "   pip install -r requirements-dev.txt"
fi

# Test installation
echo ""
echo "Testing installation..."
python3 -c "
import pandas as pd
import numpy as np
import sklearn
print('Core dependencies working!')

try:
    import xgboost
    print('XGBoost available!')
except ImportError:
    print('XGBoost not installed')

try:
    import torch
    if torch.backends.mps.is_available():
        print('PyTorch with Apple ML accelerator (MPS) available!')
    else:
        print('PyTorch available but MPS not detected')
except ImportError:
    print('PyTorch not installed')
"

echo ""
echo "Installation complete!"
echo ""
echo "Usage:"
echo "   source venv/bin/activate  # Activate virtual environment"
echo "   python scripts/evaluation/cross_platform_prediction/cross_platform.py --help"
echo ""
echo "For GPU acceleration on Apple Silicon:"
echo "   python scripts/evaluation/cross_platform_prediction/test_gpu.py"
echo ""
echo "For development:"
echo "   pip install -r requirements-dev.txt"
echo ""
echo "Happy predicting!"

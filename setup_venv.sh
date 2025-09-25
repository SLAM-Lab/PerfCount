#!/bin/bash
# Setup script to create a pre-configured virtual environment for PerfCount

set -e

echo "Setting up PerfCount virtual environment..."
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "Python version: $python_version"

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements-core.txt

# Install advanced dependencies
echo "Installing advanced dependencies..."
pip install -r requirements-advanced.txt

# Install development tools
echo "Installing development tools..."
pip install -r requirements-dev.txt

# Test installation
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
    import catboost
    print('CatBoost available!')
except ImportError:
    print('CatBoost not installed')

try:
    import lightgbm
    print('LightGBM available!')
except ImportError:
    print('LightGBM not installed')

try:
    import torch
    if torch.backends.mps.is_available():
        print('PyTorch with Apple ML accelerator (MPS) available!')
    else:
        print('PyTorch available but MPS not detected')
except ImportError:
    print('PyTorch not installed')

# Note: Intel acceleration libraries only work on Intel processors
"

echo ""
echo "Virtual environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To test GPU acceleration (if on Apple Silicon):"
echo "  python scripts/evaluation/cross_platform_prediction/test_gpu.py"
echo ""
echo "To run cross-platform prediction:"
echo "  python scripts/evaluation/cross_platform_prediction/cross_platform.py --help"
echo ""
echo "Happy predicting!"

#!/bin/bash
# Simple activation script for PerfCount virtual environment

if [ -d "venv" ]; then
    echo "Activating PerfCount virtual environment..."
    source venv/bin/activate
    echo "Virtual environment activated!"
    echo ""
    echo "Available commands:"
    echo "  python scripts/evaluation/cross_platform_prediction/cross_platform.py --help"
    echo "  python scripts/evaluation/cross_platform_prediction/test_gpu.py"
    echo ""
    echo "To deactivate: deactivate"
else
    echo "Virtual environment not found!"
    echo "Run './setup_venv.sh' first to create the environment."
    exit 1
fi

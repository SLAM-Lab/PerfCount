#!/bin/bash
# Cleanup script for workload_forecasting directory
# This script removes temporary files and cache directories

echo "=========================================="
echo "Workload Forecasting Directory Cleanup"
echo "=========================================="
echo ""

# Remove Python cache directories
echo "1. Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "   ✓ Removed __pycache__ directories"

# Remove Python bytecode files
echo "2. Removing .pyc and .pyo files..."
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null
echo "   ✓ Removed Python bytecode files"

# Remove temporary directories
echo "3. Removing temporary directories..."
find . -type d -name "tmp*" -exec rm -rf {} + 2>/dev/null
echo "   ✓ Removed temporary directories"

# Remove temporary model files in tmp directories
echo "4. Removing temporary model files..."
find . -type f \( -name "*.keras" -o -name "*.h5" \) -path "*/tmp*" -delete 2>/dev/null
echo "   ✓ Removed temporary model files"

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Note: Log files and result directories were NOT removed."
echo "If you want to clean those, do so manually:"
echo "  - logs/ (11M)"
echo "  - parallel_arm_results/ (33M)"
echo "  - Various sweep result directories"
echo ""


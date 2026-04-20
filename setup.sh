#!/bin/bash
set -e

# DynaNFE Setup Script
# Sets up the environment for inference

echo "=========================================="
echo "DynaNFE Environment Setup"
echo "=========================================="

# Check Python version
python3 --version

# Setup openpi
echo "Setting up openpi..."
cd openpi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing openpi..."
pip install --upgrade pip
pip install -e .

cd ..

echo "=========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source openpi/.venv/bin/activate"
echo ""
echo "To run inference:"
echo "  bash scripts/run_inference.sh"
echo "=========================================="

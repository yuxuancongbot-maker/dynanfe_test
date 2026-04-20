#!/bin/bash
# DynaNFE Deployment Verification Script

echo "=========================================="
echo "DynaNFE Deployment Check"
echo "=========================================="

ERRORS=0

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION"
else
    echo "✗ Python3 not found"
    ERRORS=$((ERRORS + 1))
fi

# Check directory structure
echo -n "Checking directory structure... "
if [ -d "dynanfe" ] && [ -d "scripts" ] && [ -d "checkpoints" ]; then
    echo "✓"
else
    echo "✗ Missing directories"
    ERRORS=$((ERRORS + 1))
fi

# Check openpi
echo -n "Checking openpi... "
if [ -d "openpi" ] || [ -L "openpi" ]; then
    if [ -f "openpi/pyproject.toml" ]; then
        echo "✓"
    else
        echo "✗ openpi incomplete"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "✗ openpi not found"
    ERRORS=$((ERRORS + 1))
fi

# Check checkpoints
echo -n "Checking flow model checkpoint... "
if [ -f "checkpoints/flow_model/model.safetensors" ] || [ -L "checkpoints/flow_model" ]; then
    echo "✓"
else
    echo "✗ Flow model not found"
    ERRORS=$((ERRORS + 1))
fi

echo -n "Checking MAS Head checkpoint... "
if [ -f "checkpoints/mas_head/mas_head_best.pt" ] || [ -L "checkpoints/mas_head" ]; then
    echo "✓"
else
    echo "✗ MAS Head not found"
    ERRORS=$((ERRORS + 1))
fi

# Check core files
echo -n "Checking core DynaNFE code... "
if [ -f "dynanfe/models/pi0_dynanfe_adaflow_style.py" ]; then
    echo "✓"
else
    echo "✗ Core code missing"
    ERRORS=$((ERRORS + 1))
fi

echo -n "Checking inference script... "
if [ -f "scripts/inference.py" ]; then
    echo "✓"
else
    echo "✗ Inference script missing"
    ERRORS=$((ERRORS + 1))
fi

# Check virtual environment
echo -n "Checking virtual environment... "
if [ -d "openpi/.venv" ]; then
    echo "✓"
else
    echo "⚠ Not created yet (run setup.sh)"
fi

# Check CUDA
echo -n "Checking CUDA... "
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "✓ Driver $CUDA_VERSION"
else
    echo "⚠ nvidia-smi not found"
fi

echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed!"
    echo ""
    echo "Next steps:"
    echo "  1. Run setup.sh to install dependencies"
    echo "  2. Set DATA_DIR environment variable"
    echo "  3. Run scripts/run_inference.sh"
else
    echo "✗ $ERRORS error(s) found"
    echo ""
    echo "Please check DEPLOYMENT.md for troubleshooting"
fi
echo "=========================================="

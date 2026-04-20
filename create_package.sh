#!/bin/bash
# Create deployment package for transfer to another server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PACKAGE_NAME="dynanfe-release-$(date +%Y%m%d-%H%M%S).tar.gz"

echo "=========================================="
echo "Creating DynaNFE Deployment Package"
echo "=========================================="

# Check if we need to resolve symlinks
echo "Checking symlinks..."
if [ -L "openpi" ]; then
    echo "⚠ openpi is a symlink"
    echo "  You may want to copy the actual openpi directory before packaging"
fi

if [ -L "checkpoints/flow_model" ] || [ -L "checkpoints/mas_head" ]; then
    echo "⚠ checkpoints are symlinks"
    echo "  You may want to copy the actual checkpoint files before packaging"
fi

echo ""
read -p "Continue with packaging? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Creating package: $PACKAGE_NAME"
echo "This may take a while (checkpoints are large)..."

cd ..

tar -czf "$PACKAGE_NAME" \
    --exclude='dynanfe-release/openpi/.venv' \
    --exclude='dynanfe-release/openpi/wandb' \
    --exclude='dynanfe-release/openpi/outputs' \
    --exclude='dynanfe-release/openpi/.git' \
    --exclude='dynanfe-release/__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    dynanfe-release/

PACKAGE_SIZE=$(du -h "$PACKAGE_NAME" | cut -f1)

echo "=========================================="
echo "✓ Package created: $PACKAGE_NAME"
echo "  Size: $PACKAGE_SIZE"
echo ""
echo "Transfer to target server:"
echo "  scp $PACKAGE_NAME user@server:/path/to/"
echo ""
echo "On target server:"
echo "  tar -xzf $PACKAGE_NAME"
echo "  cd dynanfe-release"
echo "  bash check_deployment.sh"
echo "=========================================="

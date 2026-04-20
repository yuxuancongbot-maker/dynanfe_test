#!/bin/bash
set -e

# DynaNFE Inference Script
# Usage: bash scripts/run_inference.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate openpi venv
source openpi/.venv/bin/activate

# Default parameters
DATASET="${DATASET:-libero_plus}"
DATA_DIR="${DATA_DIR:-/path/to/libero_plus_lerobot}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
ETA="${ETA:-0.1}"
NFE_MAX="${NFE_MAX:-20}"

echo "=========================================="
echo "DynaNFE Inference"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Data Dir: $DATA_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Num Samples: $NUM_SAMPLES"
echo "Eta: $ETA"
echo "NFE Max: $NFE_MAX"
echo "=========================================="

python scripts/inference.py \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --checkpoint-path checkpoints/flow_model/model.safetensors \
    --mas-checkpoint checkpoints/mas_head/mas_head_best.pt \
    --batch-size "$BATCH_SIZE" \
    --num-samples "$NUM_SAMPLES" \
    --num-workers 0 \
    --eta "$ETA" \
    --nfe-max "$NFE_MAX"

echo "=========================================="
echo "Inference complete!"
echo "=========================================="

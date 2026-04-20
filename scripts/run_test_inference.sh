#!/bin/bash
set -e

cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/dynanfe

/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/dynanfe/openpi/.venv/bin/python3 test_inference.py \
    --dataset libero_plus \
    --data-dir /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/data/libero_plus_lerobot \
    --checkpoint-path ./checkpoints/pi05_liberoplus_l1_flow_pytorch/model.safetensors \
    --mas-checkpoint ./checkpoints/mas_head/mas_head_best.pt \
    --batch-size 8 \
    --num-samples 100 \
    --num-workers 0 \
    --eta 0.1 \
    --nfe-max 20

# DynaNFE Release Package

**Version**: 1.0  
**Date**: 2026-04-20  
**Purpose**: Inference-ready deployment package for DynaNFE

## What's Included

```
dynanfe-release/
├── dynanfe/                    # Core DynaNFE implementation
│   └── models/
│       └── pi0_dynanfe_adaflow_style.py
├── scripts/                    # Inference scripts
│   ├── inference.py
│   └── run_inference.sh
├── checkpoints/                # Model weights (symlinks)
│   ├── flow_model/            # L1-Flow model (~2.8GB)
│   └── mas_head/              # MAS Head (~7GB)
├── openpi/                     # OpenPI library (symlink)
├── README.md                   # Quick start guide
├── DEPLOYMENT.md               # Detailed deployment instructions
├── setup.sh                    # Environment setup
├── check_deployment.sh         # Verify deployment
├── create_package.sh           # Create transfer package
└── requirements.txt            # Python dependencies
```

## Quick Start (Local)

```bash
# 1. Check deployment
bash check_deployment.sh

# 2. Setup environment (if needed)
bash setup.sh

# 3. Run inference
export DATA_DIR=/path/to/libero_plus_lerobot
bash scripts/run_inference.sh
```

## Transfer to Another Server

```bash
# Option 1: Create package
bash create_package.sh
scp dynanfe-release-*.tar.gz user@server:/path/to/

# Option 2: Direct rsync
rsync -avz --progress \
  --exclude='openpi/.venv' \
  dynanfe-release/ \
  user@server:/path/to/dynanfe-release/
```

## Key Features

- **Adaptive NFE**: Dynamic step allocation based on uncertainty
- **L1-Flow Integration**: Direct endpoint prediction
- **AdaFlow-Style**: Variance-based step sizing
- **Efficient**: Reduces inference steps while maintaining accuracy

## Supported Datasets

- LIBERO (10 tasks)
- LIBERO+ (90+ tasks)

## System Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- ~10GB disk space (code + checkpoints)
- ~8GB GPU memory (batch_size=8)

## Documentation

- **README.md**: Quick start and basic usage
- **DEPLOYMENT.md**: Detailed deployment guide
- **check_deployment.sh**: Automated verification

## Notes

- Checkpoints and openpi use symlinks by default
- For independent deployment, copy actual files before packaging
- Virtual environment is excluded from packages (recreate on target)

## Contact

For issues or questions, refer to the main development repository.

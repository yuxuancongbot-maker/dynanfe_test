# DynaNFE: Dynamic NFE Allocation for Robot Policy Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Adaptive NFE (Number of Function Evaluations) allocation for robot policy inference, combining L1-Flow's direct endpoint prediction with AdaFlow's variance-based dynamic step sizing.

## 🎯 Key Features

- **Adaptive Step Sizing**: Dynamically adjusts inference steps based on prediction uncertainty
- **L1-Flow Integration**: Direct x₁ (endpoint) prediction for efficient inference
- **AdaFlow-Style**: Variance prediction network for uncertainty estimation
- **Efficient**: Reduces average NFE while maintaining or improving accuracy

## 🏗️ Architecture

```
DynaNFE = L1-Flow (x₁ prediction) + AdaFlow (σ prediction)

Inference Loop:
  z ← noise, t ← 0
  while t < 1:
    σ ← MAS_Head(z, t|c)      # Predict uncertainty
    ε ← η / σ                  # Adaptive step size
    x₁ ← FlowModel(z, t|c)    # Predict endpoint
    v ← (x₁ - z) / (1 - t)    # Compute velocity
    z ← z + ε·v                # Update
    t ← t + ε
```

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- ~8GB GPU memory (batch_size=8)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/dynanfe.git
cd dynanfe

# Install OpenPI (required dependency)
git clone https://github.com/physical-intelligence/openpi.git
cd openpi
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cd ..

# Or use the setup script
bash setup.sh
```

## 📥 Model Checkpoints

Download pre-trained models:

1. **Flow Model** (L1-Flow trained on LIBERO+): [Download Link]
   - Place in `checkpoints/flow_model/model.safetensors`

2. **MAS Head** (Variance prediction network): [Download Link]
   - Place in `checkpoints/mas_head/mas_head_best.pt`

```bash
mkdir -p checkpoints/flow_model checkpoints/mas_head
# Download and place checkpoints here
```

## 🚀 Quick Start

### Inference

```bash
# Set data directory
export DATA_DIR=/path/to/libero_plus_lerobot

# Run inference
bash scripts/run_inference.sh
```

### Custom Configuration

```bash
python scripts/inference.py \
    --dataset libero_plus \
    --data-dir /path/to/data \
    --checkpoint-path checkpoints/flow_model/model.safetensors \
    --mas-checkpoint checkpoints/mas_head/mas_head_best.pt \
    --batch-size 8 \
    --num-samples 100 \
    --eta 0.1 \
    --nfe-max 20
```

## 📊 Results

| Method | NFE | Error | Speedup |
|--------|-----|-------|---------|
| Fixed-2 (L1-Flow) | 2.0 | 1.234 | 1.0x |
| DynaNFE (Ours) | X.X | 1.2XX | X.Xx |

*Results on LIBERO+ dataset*

## 🔧 Parameters

- `--eta`: Error threshold (default: 0.1)
  - Smaller → more conservative (more steps)
  - Larger → more aggressive (fewer steps)

- `--nfe-max`: Maximum steps (default: 20)
  - Safety limit to prevent infinite loops

- `--batch-size`: Batch size for inference
  - Adjust based on GPU memory

## 📚 Supported Datasets

- **LIBERO**: 10 manipulation tasks
- **LIBERO+**: 90+ manipulation tasks

## 🛠️ Training (Optional)

To train your own MAS Head:

```bash
# See TRAINING_GUIDE.md for detailed instructions
bash scripts/train_mas_head.sh
```

## 📖 Documentation

- [README.md](README.md) - This file
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [RELEASE_NOTES.md](RELEASE_NOTES.md) - Version history

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenPI** (Physical Intelligence): Base π0 model
- **L1-Flow**: Direct endpoint prediction approach
- **AdaFlow** (NeurIPS 2024): Variance-adaptive flow matching

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{dynanfe2024,
  title={DynaNFE: Adaptive NFE Allocation for Robot Policy Inference},
  author={Your Name},
  year={2024}
}
```

## 🔗 Related Projects

- [OpenPI](https://github.com/physical-intelligence/openpi) - π0 Vision-Language-Action Model
- [AdaFlow](https://github.com/xxx/adaflow) - Variance-Adaptive Flow Matching

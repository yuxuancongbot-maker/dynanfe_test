# DynaNFE Deployment Guide

## 传输到新服务器

### 方法 1: 使用 rsync（推荐）

```bash
# 在源服务器上
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong

# 传输到目标服务器（排除大文件）
rsync -avz --progress \
  --exclude='openpi/.venv' \
  --exclude='openpi/wandb' \
  --exclude='openpi/outputs' \
  --exclude='__pycache__' \
  dynanfe-release/ \
  user@target-server:/path/to/dynanfe-release/
```

### 方法 2: 打包传输

```bash
# 在源服务器上
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong

# 打包（排除大文件）
tar -czf dynanfe-release.tar.gz \
  --exclude='openpi/.venv' \
  --exclude='openpi/wandb' \
  --exclude='openpi/outputs' \
  --exclude='__pycache__' \
  dynanfe-release/

# 传输
scp dynanfe-release.tar.gz user@target-server:/path/to/

# 在目标服务器上解压
tar -xzf dynanfe-release.tar.gz
```

## 在新服务器上设置

### 1. 解决符号链接

由于 checkpoints 和 openpi 使用了符号链接，需要处理：

**选项 A: 复制实际文件（推荐用于独立部署）**

```bash
cd dynanfe-release

# 删除符号链接
rm openpi checkpoints/flow_model checkpoints/mas_head

# 从原服务器复制实际文件
# openpi: 需要完整的 openpi 目录
# checkpoints: 只需要模型文件
```

**选项 B: 重新创建符号链接（如果文件在同一位置）**

```bash
cd dynanfe-release
ln -s /path/to/openpi openpi
ln -s /path/to/flow_model checkpoints/flow_model
ln -s /path/to/mas_head checkpoints/mas_head
```

### 2. 安装依赖

```bash
cd dynanfe-release
bash setup.sh
```

### 3. 准备数据

将数据集放置在合适的位置：
- LIBERO: `/path/to/libero/`
- LIBERO+: `/path/to/libero_plus_lerobot/`

### 4. 配置路径

编辑 `scripts/run_inference.sh`，修改 `DATA_DIR`：

```bash
DATA_DIR="${DATA_DIR:-/your/actual/path/to/libero_plus_lerobot}"
```

### 5. 运行测试

```bash
# 设置数据路径
export DATA_DIR=/path/to/libero_plus_lerobot

# 运行推理
bash scripts/run_inference.sh
```

## 文件清单

### 必需文件（需要传输）

```
dynanfe-release/
├── dynanfe/                          # 核心代码 (~50KB)
├── scripts/                          # 推理脚本 (~20KB)
├── openpi/                           # openpi 库 (~需要完整目录)
├── checkpoints/
│   ├── flow_model/model.safetensors # Flow model (~2.8GB)
│   └── mas_head/mas_head_best.pt    # MAS Head (~7GB)
├── README.md
├── setup.sh
└── requirements.txt
```

### 可选文件（不需要传输）

- `openpi/.venv/` - 虚拟环境（在新服务器重新创建）
- `openpi/wandb/` - 训练日志
- `openpi/outputs/` - 输出文件
- `data/` - 数据集（在新服务器单独准备）

## 磁盘空间需求

- **最小**: ~10GB（代码 + checkpoints）
- **推荐**: ~50GB（包含数据集）

## 故障排查

### 问题 1: 找不到 openpi 模块

```bash
# 确保 openpi 已安装
cd openpi
pip install -e .
```

### 问题 2: checkpoint 文件不存在

```bash
# 检查符号链接
ls -la checkpoints/

# 如果是断开的链接，需要复制实际文件
```

### 问题 3: CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvidia-smi

# 重新安装对应版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 性能优化

### 多 GPU 推理

目前脚本使用单 GPU，如需多 GPU：

```python
# 修改 scripts/inference.py
model = torch.nn.DataParallel(model)
```

### 批量大小调整

根据 GPU 内存调整：
- 8GB GPU: `--batch-size 4`
- 16GB GPU: `--batch-size 8`
- 24GB GPU: `--batch-size 16`

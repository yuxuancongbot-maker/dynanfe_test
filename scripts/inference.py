#!/usr/bin/env python3
"""
测试 DynaNFE 推理：对比固定步数 vs 自适应步数
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import safetensors.torch

# 添加 openpi 到路径
sys.path.insert(0, str(Path(__file__).parent / "openpi" / "src"))

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.models import model as _model

from dynanfe.models.pi0_dynanfe_adaflow_style import PI0PytorchWithDynaNFEAdaFlow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def obs_to_device(observation: _model.Observation, device: torch.device) -> _model.Observation:
    """将 Observation 移到 device"""
    def _to(x):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            return torch.as_tensor(x, device=device)
        return x

    return _model.Observation(
        image=_to(observation.image),
        wrist_image=_to(observation.wrist_image) if observation.wrist_image is not None else None,
        state=_to(observation.state),
        prompt=observation.prompt,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="libero_plus", choices=["libero_plus"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Flow model checkpoint")
    parser.add_argument("--mas-checkpoint", type=str, required=True, help="MAS Head checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=100, help="测试样本数")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.1, help="误差阈值")
    parser.add_argument("--nfe-max", type=int, default=20, help="最大步数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    train_config = _config.get_config("pi05_libero_plus_dynanfe")
    model_config = train_config.model

    # 设置 DynaNFE 参数
    model_config.use_dynanfe = True
    model_config.eta = args.eta
    model_config.nfe_max = args.nfe_max

    # ── 构建数据加载器 ────────────────────────────────────────────────────────
    data_config_factory = train_config.data
    object.__setattr__(data_config_factory, "repo_id", str(args.data_dir))
    data_config = data_config_factory.create(train_config.assets_dirs, model_config)

    loader = _data_loader.create_torch_data_loader(
        data_config=data_config,
        model_config=model_config,
        action_horizon=32,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        framework="pytorch",
        skip_norm_stats=False,
    )
    logger.info("DataLoader created")

    # ── 创建模型 ──────────────────────────────────────────────────────────────
    model = PI0PytorchWithDynaNFEAdaFlow(model_config)

    # 加载 flow model
    logger.info(f"Loading flow model from {args.checkpoint_path}")
    state_dict = safetensors.torch.load_file(str(args.checkpoint_path))
    model.load_state_dict(state_dict, strict=False)

    # 加载 MAS Head
    logger.info(f"Loading MAS Head from {args.mas_checkpoint}")
    mas_ckpt = torch.load(args.mas_checkpoint, map_location="cpu")
    model.load_state_dict(mas_ckpt["model_state_dict"], strict=False)

    model.to(device)
    model.eval()
    logger.info("Model loaded")

    # ── 推理测试 ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Testing DynaNFE Inference")
    logger.info("=" * 60)

    data_iter = iter(loader)
    total_samples = 0
    total_nfe_dynanfe = 0
    total_nfe_fixed = 0
    total_error_dynanfe = 0
    total_error_fixed = 0

    with torch.no_grad():
        while total_samples < args.num_samples:
            try:
                observation, actions_gt = next(data_iter)
            except StopIteration:
                break

            observation = obs_to_device(observation, device)
            actions_gt = torch.as_tensor(np.asarray(actions_gt), dtype=torch.float32).to(device)
            B = actions_gt.shape[0]

            # ===== 1. DynaNFE 推理（自适应步数）=====
            actions_dynanfe, nfe_info = model.sample_actions(
                device, observation, noise=None, num_steps=None, return_nfe=True
            )

            # 计算误差
            error_dynanfe = (actions_dynanfe - actions_gt).abs().sum(dim=(-1, -2)).mean().item()
            nfe_dynanfe = nfe_info['mean_nfe']

            # ===== 2. 固定步数推理（2 步，L1-Flow 默认）=====
            # 临时关闭 DynaNFE
            model.use_dynanfe = False
            actions_fixed = model.sample_actions(device, observation, noise=None, num_steps=2)
            model.use_dynanfe = True

            error_fixed = (actions_fixed - actions_gt).abs().sum(dim=(-1, -2)).mean().item()

            # 统计
            total_samples += B
            total_nfe_dynanfe += nfe_dynanfe * B
            total_nfe_fixed += B * 2
            total_error_dynanfe += error_dynanfe * B
            total_error_fixed += error_fixed * B

            if total_samples % 32 == 0:
                logger.info(
                    f"Samples: {total_samples}/{args.num_samples}  "
                    f"NFE(DynaNFE)={nfe_dynanfe:.2f}  "
                    f"Error(DynaNFE)={error_dynanfe:.4f}  "
                    f"Error(Fixed-2)={error_fixed:.4f}"
                )

    # ── 结果汇总 ──────────────────────────────────────────────────────────────
    avg_error_dynanfe = total_error_dynanfe / total_samples
    avg_error_fixed = total_error_fixed / total_samples
    avg_nfe_dynanfe = total_nfe_dynanfe / total_samples
    avg_nfe_fixed = total_nfe_fixed / total_samples

    logger.info("=" * 60)
    logger.info("Results:")
    logger.info(f"  Samples: {total_samples}")
    logger.info(f"  Fixed-2 steps:")
    logger.info(f"    - NFE: {avg_nfe_fixed:.2f}")
    logger.info(f"    - Error: {avg_error_fixed:.6f}")
    logger.info(f"  DynaNFE (adaptive):")
    logger.info(f"    - NFE: {avg_nfe_dynanfe:.2f}")
    logger.info(f"    - Error: {avg_error_dynanfe:.6f}")
    logger.info(f"  Speedup: {avg_nfe_fixed / avg_nfe_dynanfe:.2f}x")
    logger.info(f"  Error ratio: {avg_error_dynanfe / avg_error_fixed:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate utility-based soft/hard NFE labels for Router training.

For each sample and each NFE option k, the script computes:
  success_proxy_k = - mean(|a_pred_k - a_gt|)
  latency_k       = inference latency in milliseconds
  instability_k   = mean(||a_t - a_{t-1}||_2)

Then per sample (across k) it min-max normalizes each component and builds
utility:
  U_k = alpha * success_norm_k - beta * latency_norm_k - gamma * instability_norm_k

Soft labels are produced by softmax(U / tau), hard labels by argmax(U).

Output .pt keys:
  labels            (N,) long hard labels
  labels_soft       (N, K) float soft labels
  utilities         (N, K) float
  success_proxy     (N, K) float
  latency_ms        (N, K) float
  instability       (N, K) float
  nfe_options       list[int]
  label_counts      list[int]
  num_samples       int
  alpha,beta,gamma,tau,seed

Usage:
    python scripts/generate_nfe_utility_labels.py \
        --config pi05_libero \
        --checkpoint-dir /path/to/checkpoints/pi05_libero \
        --output /path/to/nfe_utility_labels.pt
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "openpi" / "src"))

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def _sample_fixed_nfe(model, state, prefix_pad_masks, past_key_values, noise, nfe, device):
    """Run fixed-NFE L1-flow style sampling."""
    bsize = noise.shape[0]
    dt = 1.0 / float(nfe)

    z = noise.clone()
    for step in range(nfe):
        t = torch.full((bsize,), step * dt, device=device, dtype=torch.float32)
        x1_pred = model.denoise_step(state, prefix_pad_masks, past_key_values, z, t)
        if step == nfe - 1:
            z = x1_pred
        else:
            v = (x1_pred - z) / (1.0 - t[:, None, None]).clamp(min=1e-6)
            z = z + dt * v
    return z


def _clone_kv_cache(past_key_values):
    """Clone KV cache to avoid cross-call side effects."""
    if isinstance(past_key_values, (list, tuple)):
        return type(past_key_values)(
            tuple(t.clone() for t in layer_kv) for layer_kv in past_key_values
        )
    elif hasattr(past_key_values, "key_cache"):
        past_key_values.key_cache = [k.clone() for k in past_key_values.key_cache]
        past_key_values.value_cache = [v.clone() for v in past_key_values.value_cache]
    return past_key_values


def _instability(actions):
    """actions: (B, H, D) -> (B,) mean temporal L2 delta."""
    if actions.shape[1] < 2:
        return torch.zeros(actions.shape[0], device=actions.device, dtype=torch.dtype)
    delta = actions[:, 1:, :] - actions[:, :-1, :]
    return torch.norm(delta, p=2, dim=-1).mean(dim=-1)


def _minmax_per_row(x, eps=1e-8):
    """x: (B, K) -> min-max normalized over K per sample."""
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    return (x - x_min) / (x_max - x_min + eps)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from pi0_pytorch."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def main():
    parser = argparse.ArgumentParser(description="Generate utility-based NFE labels")
    parser.add_argument("--config", type=str, required=True,
                        help="Config name (e.g., pi05_libero)")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Base flow checkpoint dir (model.safetensors)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for nfe_utility_labels.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nfe-options", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--tau", type=float, default=0.7)

    args = parser.parse_args()

    if args.tau <= 0:
        raise ValueError(f"tau must be > 0, got {args.tau}")

    device = torch.device(args.device)

    # Load config and model
    config = _config.get_config(args.config)
    model_config = config.model

    # Enable PyTorch model loading
    pytorch_weight_path = getattr(model_config, "pytorch_weight_path", None) or \
        str(Path(args.checkpoint_dir) / "model.safetensors")

    logger.info(f"Loading model from {pytorch_weight_path}")

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import safetensors.torch

    model = PI0Pytorch(config=model_config)
    safetensors.torch.load_model(model, pytorch_weight_path, strict=False)
    model.to(device)
    model.eval()
    logger.info("Model loaded.")

    # Create data loader
    jax.config.update("jax_platform_name", "cpu")

    import jax
    import jax.numpy as jnp
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        shuffle=False,
    )

    logger.info(f"Dataset: {data_loader.num_examples} samples")
    logger.info(
        f"Utility weights: alpha={args.alpha:.3f}, beta={args.beta:.3f}, "
        f"gamma={args.gamma:.3f}, tau={args.tau:.3f}"
    )

    all_success = []
    all_latency = []
    all_instab = []
    all_utility = []
    all_labels_soft = []
    all_labels_hard = []

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    data_iter = iter(data_loader)

    with torch.no_grad():
        for batch_idx, (observation, actions) in enumerate(tqdm(data_iter, desc="Generating utility labels")):
            # Convert JAX observation to PyTorch format
            bsize = actions.shape[0]

            # Build PyTorch observation
            torch_images = {}
            for k, v in observation.images.items():
                # v is (B, H, W, C) numpy -> convert to (B, C, H, W) torch
                v_np = np.asarray(v)
                if v_np.ndim == 4 and v_np.shape[-1] == 3:
                    v_np = np.transpose(v_np, (0, 3, 1, 2))
                torch_images[k] = torch.from_numpy(v_np).float().to(device)

            torch_masks = {k: torch.ones(v.shape[0], dtype=torch.bool, device=device) for k, v in torch_images.items()}

            state = torch.from_numpy(np.asarray(observation.state)).float().to(device)
            actions_t = torch.from_numpy(np.asarray(actions)).float().to(device)

            # Build observation dict for model
            from openpi.models import model as _model
            torch_obs = _model.Observation(
                images=torch_images,
                image_masks=torch_masks,
                state=state,
                tokenized_prompt=None,
                tokenized_prompt_mask=None,
            )

            noise = torch.randn(
                bsize,
                model.config.action_horizon,
                model.config.action_dim,
                device=device,
                generator=rng,
            )

            images, img_masks, lang_tokens, lang_masks, state_prep = model._preprocess_observation(
                torch_obs, train=False
            )

            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

            model_dtype = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            prefix_embs = prefix_embs.to(dtype=model_dtype)
            prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=model_dtype)

            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            success_cols = []
            latency_cols = []
            instab_cols = []

            for nfe in args.nfe_options:
                kv = _clone_kv_cache(past_key_values)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()

                pred = _sample_fixed_nfe(
                    model=model,
                    state=state_prep,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=kv,
                    noise=noise,
                    nfe=nfe,
                    device=device,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                ms = (time.perf_counter() - t0) * 1000.0

                err = (pred - actions_t).abs().mean(dim=(-1, -2))
                success_proxy = -err
                instability = _instability(pred)

                success_cols.append(success_proxy)
                latency_cols.append(torch.full((bsize,), ms, device=device, dtype=torch.float32))
                instab_cols.append(instability)

            success = torch.stack(success_cols, dim=1)
            latency = torch.stack(latency_cols, dim=1)
            instability = torch.stack(instab_cols, dim=1)

            success_n = _minmax_per_row(success)
            latency_n = _minmax_per_row(latency)
            instability_n = _minmax_per_row(instability)

            utility = (
                args.alpha * success_n
                - args.beta * latency_n
                - args.gamma * instability_n
            )

            labels_soft = torch.softmax(utility / args.tau, dim=-1)
            labels_hard = utility.argmax(dim=-1)

            all_success.append(success.cpu())
            all_latency.append(latency.cpu())
            all_instab.append(instability.cpu())
            all_utility.append(utility.cpu())
            all_labels_soft.append(labels_soft.cpu())
            all_labels_hard.append(labels_hard.cpu())

    success_all = torch.cat(all_success, dim=0)
    latency_all = torch.cat(all_latency, dim=0)
    instab_all = torch.cat(all_instab, dim=0)
    utility_all = torch.cat(all_utility, dim=0)
    labels_soft_all = torch.cat(all_labels_soft, dim=0)
    labels_hard_all = torch.cat(all_labels_hard, dim=0).long()

    num_classes = len(args.nfe_options)
    label_counts = torch.bincount(labels_hard_all, minlength=num_classes).tolist()
    total = int(labels_hard_all.numel())

    logger.info("Label distribution:")
    for i, nfe_val in enumerate(args.nfe_options):
        pct = 100.0 * label_counts[i] / max(total, 1)
        logger.info(f"  class {i} (NFE={nfe_val}): {label_counts[i]} ({pct:.1f}%)")

    logger.info(
        f"Avg components | success={success_all.mean().item():.4f} "
        f"latency_ms={latency_all.mean().item():.2f} instability={instab_all.mean().item():.4f}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "labels": labels_hard_all,
            "labels_soft": labels_soft_all,
            "utilities": utility_all,
            "success_proxy": success_all,
            "latency_ms": latency_all,
            "instability": instab_all,
            "nfe_options": args.nfe_options,
            "label_counts": label_counts,
            "num_samples": total,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "tau": args.tau,
            "seed": args.seed,
        },
        output_path,
    )
    logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
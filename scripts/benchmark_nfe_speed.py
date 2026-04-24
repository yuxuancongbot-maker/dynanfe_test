#!/usr/bin/env python3
"""
NFE speed benchmark: compare latency across different NFE modes with/without torch.compile.

Uses dummy random observations (no dataset needed). Monkey-patches compute_sigma()
to simulate fixed or random step counts without a trained MAS head.

Usage:
    python scripts/benchmark_nfe_speed.py \
        --checkpoint-dir /path/to/pi05_libero \
        --batch-size 1 \
        --num-iters 50 \
        --compile-mode default
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "openpi" / "src"))

from openpi.models_pytorch.pi0_dynanfe import PI0PytorchWithDynaNFE
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
import safetensors.torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dummy observation
# ---------------------------------------------------------------------------

def make_dummy_observation(batch_size: int, device: torch.device):
    """Create a random observation with the same shapes as real LIBERO data."""
    B = batch_size

    def rand_img():
        return torch.randn(B, 3, 224, 224, device=device)

    images = {
        "base_0_rgb": rand_img(),
        "left_wrist_0_rgb": rand_img(),
        "right_wrist_0_rgb": torch.zeros(B, 3, 224, 224, device=device),
    }
    image_masks = {
        "base_0_rgb": torch.ones(B, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(B, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.zeros(B, dtype=torch.bool, device=device),
    }
    state = torch.randn(B, 32, device=device)
    tokenized_prompt = torch.randint(0, 30000, (B, 200), device=device, dtype=torch.long)
    tokenized_prompt_mask = torch.ones(B, 200, dtype=torch.bool, device=device)

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


# ---------------------------------------------------------------------------
# Sigma patchers
# ---------------------------------------------------------------------------

def make_fixed_sigma(model, target_nfe):
    """Return a fake compute_sigma that yields exactly target_nfe steps."""
    eta = model.eta

    def fake_sigma(prefix_pooled, noisy_actions, timestep):
        B = noisy_actions.shape[0]
        return torch.full((B,), eta * target_nfe, device=noisy_actions.device)

    return fake_sigma


def make_random_sigma(model, nfe_low, nfe_high):
    """Return a fake compute_sigma that yields nfe_low~nfe_high steps per sample."""
    eta = model.eta

    def fake_sigma(prefix_pooled, noisy_actions, timestep):
        B = noisy_actions.shape[0]
        target = torch.randint(nfe_low, nfe_high + 1, (B,), device=noisy_actions.device).float()
        return eta * target

    return fake_sigma


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def create_dynanfe_config(nfe_max=16, use_router=False, nfe_options=(1, 2, 4)):
    """Create Pi0Config with DynaNFE fields injected."""
    config = Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    object.__setattr__(config, "use_dynanfe", True)
    object.__setattr__(config, "eta", 0.1)
    object.__setattr__(config, "nfe_max", nfe_max)
    object.__setattr__(config, "mas_hidden_dim", 256)
    object.__setattr__(config, "use_nfe_router", use_router)
    object.__setattr__(config, "nfe_options", nfe_options)
    object.__setattr__(config, "router_hidden_dim", 256)
    return config


def load_model(checkpoint_dir: str, device: torch.device, nfe_max: int = 16,
               use_router: bool = False, nfe_options: tuple = (1, 2, 4)):
    """Load PI0PytorchWithDynaNFE from safetensors checkpoint."""
    model_config = create_dynanfe_config(nfe_max, use_router, nfe_options)
    model = PI0PytorchWithDynaNFE(model_config)

    weight_path = Path(checkpoint_dir) / "model.safetensors"
    logger.info(f"Loading base flow weights from {weight_path}")
    safetensors.torch.load_model(model, str(weight_path), strict=False)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def make_fixed_router(model, target_class_idx):
    """Return a fake nfe_router.forward that always predicts a fixed class."""
    def fake_forward(prefix_pooled):
        B = prefix_pooled.shape[0]
        logits = torch.zeros(B, len(model.nfe_options), device=prefix_pooled.device)
        logits[:, target_class_idx] = 10.0  # strong preference
        return logits
    return fake_forward


def benchmark_mode(
    model,
    observation,
    device,
    mode: str,
    num_iters: int,
    warmup: int,
    use_compile: bool,
    compile_mode: str,
) -> dict:
    """Run benchmark for a single NFE mode.

    Supports: fixed-N, random-L-H, router-N modes.
    Returns dict with mode, compile, mean_nfe, latencies.
    """
    is_router_mode = mode.startswith("router-")
    uses_adaptive = False

    if mode == "fixed-2":
        model.use_dynanfe = False
        model.use_nfe_router = False
    elif mode.startswith("fixed-"):
        target_nfe = int(mode.split("-")[1])
        model.use_dynanfe = True
        model.use_nfe_router = False
        model.nfe_max = max(target_nfe, model.nfe_max)
        model.compute_sigma = make_fixed_sigma(model, target_nfe)
        uses_adaptive = True
    elif mode.startswith("random-"):
        parts = mode.split("-")
        nfe_low, nfe_high = int(parts[1]), int(parts[2])
        model.use_dynanfe = True
        model.use_nfe_router = False
        model.nfe_max = max(nfe_high, model.nfe_max)
        model.compute_sigma = make_random_sigma(model, nfe_low, nfe_high)
        uses_adaptive = True
    elif is_router_mode:
        target_nfe = int(mode.split("-")[1])
        model.use_dynanfe = False
        model.use_nfe_router = True
        try:
            class_idx = list(model.nfe_options).index(target_nfe)
        except ValueError:
            raise ValueError(f"NFE={target_nfe} not in nfe_options={model.nfe_options}")
        model.nfe_router.forward = make_fixed_router(model, class_idx)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Optionally compile
    if use_compile and (uses_adaptive or is_router_mode):
        model.enable_torch_compile(mode=compile_mode, dynamic=False)

    # Warmup
    logger.info(f"  Warmup ({warmup} iters)...")
    for _ in range(warmup):
        model.sample_actions(device, observation, return_nfe=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    nfe_list = []

    for i in range(num_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        result = model.sample_actions(device, observation, return_nfe=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if isinstance(result, tuple):
            _, nfe_info = result
            nfe_list.append(nfe_info["mean_nfe"])
        else:
            nfe_list.append(2.0)

        latencies.append((t1 - t0) * 1000)

    return {
        "mode": mode,
        "compile": use_compile,
        "latencies": latencies,
        "nfe_list": nfe_list,
    }


def print_results(results: list[dict]):
    """Print benchmark results as a formatted table."""
    header = f"{'Mode':<14} | {'Compile':<8} | {'NFE':>5} | {'Mean(ms)':>10} | {'Std(ms)':>9} | {'P50(ms)':>9} | {'P95(ms)':>9}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    for r in results:
        lats = np.array(r["latencies"])
        nfe_arr = np.array(r["nfe_list"])
        print(
            f"{r['mode']:<14} | {'on' if r['compile'] else 'off':<8} | "
            f"{nfe_arr.mean():>5.1f} | "
            f"{lats.mean():>10.1f} | {lats.std():>9.1f} | "
            f"{np.percentile(lats, 50):>9.1f} | {np.percentile(lats, 95):>9.1f}"
        )
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NFE speed benchmark")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to base flow checkpoint dir (model.safetensors)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile-mode", type=str, default=None,
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (None = skip compile tests)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--modes", type=str,
                        default="fixed-2,fixed-4,fixed-8,fixed-16,random-2-8",
                        help="Comma-separated NFE modes (fixed-N, random-L-H, router-N)")
    parser.add_argument("--nfe-options", type=int, nargs="+", default=[1, 2, 4],
                        help="NFE options for router modes (default: 1 2 4)")
    args = parser.parse_args()

    device = torch.device(args.device)
    modes = [m.strip() for m in args.modes.split(",")]

    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Modes: {modes}")
    logger.info(f"Compile: {args.compile_mode}")
    logger.info(f"Iters: {args.num_iters} (warmup: {args.warmup})")

    observation = make_dummy_observation(args.batch_size, device)
    all_results = []

    for mode in modes:
        needs_router = mode.startswith("router-")

        # --- Without compile ---
        logger.info(f"[{mode}] no compile")
        model_fresh = load_model(args.checkpoint_dir, device, nfe_max=16,
                                 use_router=needs_router,
                                 nfe_options=tuple(args.nfe_options))
        r = benchmark_mode(model_fresh, observation, device, mode,
                           args.num_iters, args.warmup, use_compile=False, compile_mode="default")
        all_results.append(r)
        del model_fresh
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # --- With compile ---
        if args.compile_mode is not None:
            logger.info(f"[{mode}] compile={args.compile_mode}")
            model_fresh = load_model(args.checkpoint_dir, device, nfe_max=16,
                                     use_router=needs_router,
                                     nfe_options=tuple(args.nfe_options))
            r = benchmark_mode(model_fresh, observation, device, mode,
                               args.num_iters, args.warmup, use_compile=True,
                               compile_mode=args.compile_mode)
            all_results.append(r)
            del model_fresh
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print_results(all_results)


if __name__ == "__main__":
    main()
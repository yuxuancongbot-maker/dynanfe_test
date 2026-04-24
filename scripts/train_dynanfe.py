#!/usr/bin/env python3
"""
Stage 2 training for DynaNFE: train MAS Head or NFE Router.

This script:
1. Loads a pre-trained PI0Pytorch flow model (from safetensors)
2. Freezes the flow model
3. Trains either:
   - MAS Head (AdaFlow mode): Gaussian NLL loss on variance prediction
   - NFE Router (Router mode): Cross-entropy loss on utility-based soft labels

Usage:
    # Train MAS Head
    python scripts/train_dynanfe.py \
        --checkpoint-dir /path/to/pi05_libero \
        --stage2_type mas_head \
        --dataset libero_plus \
        --data-dir /path/to/libero_datasets \
        --norm-stats-path /path/to/norm_stats.json \
        --dynanfe_stage2_checkpoint /tmp/mas_head_output

    # Train NFE Router
    python scripts/train_dynanfe.py \
        --checkpoint-dir /path/to/pi05_libero \
        --stage2_type router \
        --dataset libero_plus \
        --data-dir /path/to/libero_datasets \
        --norm-stats-path /path/to/norm_stats.json \
        --nfe_labels_path /path/to/nfe_utility_labels.pt \
        --dynanfe_stage2_checkpoint /tmp/router_output
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add openpi src to path
OPENPI_ROOT = Path(__file__).resolve().parent.parent / "openpi"
sys.path.insert(0, str(OPENPI_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 training for DynaNFE")

    # Model / checkpoint
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to Stage 1 checkpoint (contains model.safetensors)")

    # Stage 2 type
    parser.add_argument("--stage2_type", type=str, required=True,
                        choices=["mas_head", "router"],
                        help="Which head to train: mas_head or router")

    # Data
    parser.add_argument("--dataset", type=str, required=True, choices=["libero", "libero_plus"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--norm-stats-path", type=str, required=True)
    parser.add_argument("--libero-subsets", type=str, nargs="*", default=None)

    # NFE labels (router only)
    parser.add_argument("--nfe_labels_path", type=str, default=None,
                        help="Path to utility-based NFE labels (.pt file from generate_nfe_utility_labels.py)")

    # Training hyperparams
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-train-steps", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    # Output
    parser.add_argument("--dynanfe_stage2_checkpoint", type=str, required=True,
                        help="Output directory for Stage 2 checkpoints")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # DynaNFE config
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--nfe-max", type=int, default=4)
    parser.add_argument("--mas-hidden-dim", type=int, default=256)
    parser.add_argument("--use-nfe-router", action="store_true")
    parser.add_argument("--nfe-options", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--router-hidden-dim", type=int, default=256)

    return parser.parse_args()


def load_norm_stats(path):
    import json
    with open(path) as f:
        return json.load(f)


# Minimal data loading (avoid importing dynanfe datasets to keep it standalone)
ACTION_HORIZON = 10


class LiberoHDF5Dataset:
    """Simple HDF5 dataset for LIBERO."""
    def __init__(self, data_dir, action_horizon=10, subsets=None):
        import h5py
        self.data_dir = Path(data_dir)
        self.action_horizon = action_horizon

        # Find all .h5 files
        if subsets is not None:
            h5_files = [self.data_dir / f"{s}.hdf5" for s in subsets]
        else:
            h5_files = list(self.data_dir.glob("*.hdf5"))

        self.episodes = []
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                for ep_key in f.keys():
                    self.episodes.append((h5_file, ep_key))

        logger.info(f"Loaded {len(self.episodes)} episodes from {len(h5_files)} files")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        import h5py
        h5_file, ep_key = self.episodes[idx]
        with h5py.File(h5_file, "r") as f:
            ep = f[ep_key]
            # Get observations
            images = {
                "base_0_rgb": ep["observation/image"][:],
                "left_wrist_0_rgb": ep["observation/wrist_image"][:],
                "right_wrist_0_rgb": np.zeros_like(ep["observation/image"][:]),
            }
            state = ep["observation/state"][:]
            prompt = ep["prompt"][()] if isinstance(ep["prompt"][()], str) else ep["prompt"][()].decode()

            # Actions: shape (T, action_dim)
            actions = ep["actions"][:]

            # Sample a random chunk of action_horizon steps
            if len(actions) > self.action_horizon:
                start = np.random.randint(0, len(actions) - self.action_horizon)
            else:
                start = 0
            action_chunk = actions[start:start + self.action_horizon]
            if action_chunk.shape[0] < self.action_horizon:
                pad = self.action_horizon - action_chunk.shape[0]
                action_chunk = np.concatenate([action_chunk, np.zeros((pad, action_chunk.shape[1]))], axis=0)

        return {
            "images": images,
            "state": state,
            "prompt": prompt,
            "actions": action_chunk.astype(np.float32),
        }


class LiberoPlusDataset(LiberoHDF5Dataset):
    """LIBERO+ dataset."""
    def __init__(self, data_dir, action_horizon=10):
        super().__init__(data_dir, action_horizon, subsets=None)
        # Override: LIBERO+ has different structure
        import h5py
        self.episodes = []
        data_dir = Path(data_dir)

        h5_files = list(data_dir.glob("**/*.hdf5"))
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                for ep_key in f.keys():
                    self.episodes.append((h5_file, ep_key))

        logger.info(f"Loaded {len(self.episodes)} episodes from {len(h5_files)} LIBERO+ files")


class NFELabelsDataset(Dataset):
    """Dataset that pairs observations with pre-computed NFE utility labels."""
    def __init__(self, base_dataset, nfe_labels_path):
        self.base_dataset = base_dataset
        data = torch.load(nfe_labels_path)
        self.labels_soft = data["labels_soft"]  # (N, K)
        self.labels_hard = data["labels"]       # (N,)
        self.utilities = data["utilities"]       # (N, K)

        if len(self.base_dataset) != len(self.labels_soft):
            logger.warning(f"Dataset size mismatch: {len(self.base_dataset)} samples vs {len(self.labels_soft)} labels")

    def __len__(self):
        return min(len(self.base_dataset), len(self.labels_soft))

    def __getitem__(self, idx):
        obs = self.base_dataset[idx]
        return obs, self.labels_soft[idx], self.labels_hard[idx]


def collate_fn(batch):
    """Collate a batch of observations."""
    images_dict = {}
    states = []
    prompts = []
    actions = []

    for obs in batch:
        for k, v in obs["images"].items():
            if k not in images_dict:
                images_dict[k] = []
            images_dict[k].append(v)
        states.append(obs["state"])
        prompts.append(obs["prompt"])
        actions.append(obs["actions"])

    return {
        "images": {k: np.stack(v) for k, v in images_dict.items()},
        "state": np.stack(states),
        "prompt": prompts,
        "actions": np.stack(actions),
    }


def make_observation(obs_dict, device):
    """Convert dict to observation format expected by model."""
    from openpi.models import model as _model

    images = {}
    for k, v in obs_dict["images"].items():
        # HWC -> CHW
        if v.ndim == 4 and v.shape[-1] == 3:
            v = np.transpose(v, (0, 3, 1, 2))
        images[k] = torch.from_numpy(v).float()

    state = torch.from_numpy(np.asarray(obs_dict["state"])).float()
    prompt = obs_dict["prompt"]

    # Build tokenized prompt placeholder (will be tokenized by model)
    return _model.Observation(
        images=images,
        image_masks={k: torch.ones(v.shape[0], dtype=torch.bool) for k, v in images.items()},
        state=state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )


def load_pytorch_model(checkpoint_dir, config, device):
    """Load PI0PytorchWithDynaNFE from safetensors checkpoint."""
    from openpi.models_pytorch.pi0_dynanfe import PI0PytorchWithDynaNFE
    import safetensors.torch

    model = PI0PytorchWithDynaNFE(config)
    weight_path = Path(checkpoint_dir) / "model.safetensors"
    safetensors.torch.load_model(model, str(weight_path), strict=False)
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {weight_path}")
    return model


def freeze_flow_model(model):
    """Freeze all parameters except MAS head / NFE router."""
    for name, param in model.named_parameters():
        if "mas_head" in name or "nfe_router" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def create_model_config(args):
    """Create Pi0Config with DynaNFE fields."""
    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(
        pi05=True,
        action_horizon=ACTION_HORIZON,
        discrete_state_input=False,
        pytorch_compile_mode=None,  # Disable compile during training
    )

    # Inject DynaNFE fields
    object.__setattr__(config, "use_dynanfe", True)
    object.__setattr__(config, "eta", args.eta)
    object.__setattr__(config, "nfe_max", args.nfe_max)
    object.__setattr__(config, "mas_hidden_dim", args.mas_hidden_dim)
    object.__setattr__(config, "use_nfe_router", args.use_nfe_router)
    object.__setattr__(config, "nfe_options", tuple(args.nfe_options))
    object.__setattr__(config, "router_hidden_dim", args.router_hidden_dim)

    return config


def train_mas_head(model, train_loader, optimizer, device, num_steps, log_interval):
    """Train MAS Head with Gaussian NLL loss."""
    model.train()

    pbar = tqdm(range(num_steps), desc="Training MAS Head")
    step = 0
    iter_loader = iter(train_loader)

    while step < num_steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        # Move to device
        observation = make_observation(batch, device)
        actions = torch.from_numpy(np.asarray(batch["actions"])).float().to(device)

        # Forward pass
        loss, info = model.forward_stage2(observation, actions)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_interval == 0:
            logger.info(
                f"Step {step}: mas_loss={info['mas_loss']:.4f}, "
                f"mean_sigma={info['mean_sigma']:.4f}, mean_error={info['mean_error']:.4f}, "
                f"correlation={info['correlation']:.4f}, est_nfe={info['estimated_nfe']:.2f}"
            )

        pbar.update(1)
        step += 1

    pbar.close()
    return model


def train_router(model, train_loader, optimizer, device, num_steps, log_interval):
    """Train NFE Router with soft label cross-entropy loss."""
    model.train()

    pbar = tqdm(range(num_steps), desc="Training NFE Router")
    step = 0
    iter_loader = iter(train_loader)

    while step < num_steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        obs_dict, labels_soft, labels_hard = batch
        observation = make_observation(obs_dict, device)
        labels_soft = labels_soft.to(device)
        labels_hard = labels_hard.to(device)

        # Forward pass
        loss, info = model.forward_router_stage2(
            observation,
            nfe_labels=labels_hard,
            nfe_labels_soft=labels_soft,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_interval == 0:
            logger.info(
                f"Step {step}: router_loss={info['router_loss']:.4f}, "
                f"accuracy={info['accuracy']:.4f}, mean_pred_nfe={info['mean_predicted_nfe']:.2f}"
            )
            for nfe_val in model.nfe_options:
                pct = info.get(f"class_{nfe_val}_pct", 0)
                logger.info(f"  class_{nfe_val}_pct={pct:.3f}")

        pbar.update(1)
        step += 1

    pbar.close()
    return model


def save_stage2_checkpoint(model, output_dir, stage2_type):
    """Save only MAS head or router weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {}
    if stage2_type == "mas_head":
        for name, param in model.named_parameters():
            if "mas_head" in name:
                state_dict[name] = param.detach().cpu()
        torch.save(state_dict, output_dir / "mas_head.pt")
        logger.info(f"Saved MAS head to {output_dir / 'mas_head.pt'}")

    elif stage2_type == "router":
        for name, param in model.named_parameters():
            if "nfe_router" in name:
                state_dict[name] = param.detach().cpu()
        torch.save(state_dict, output_dir / "nfe_router.pt")
        logger.info(f"Saved NFE Router to {output_dir / 'nfe_router.pt'}")

    # Save full model state dict for convenience
    torch.save(model.state_dict(), output_dir / "full_model.pt")
    logger.info(f"Saved full model to {output_dir / 'full_model.pt'}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Create model config
    model_config = create_model_config(args)

    # Load model
    model = load_pytorch_model(args.checkpoint_dir, model_config, device)

    # Freeze flow model
    freeze_flow_model(model)

    # Create dataset
    if args.dataset == "libero":
        raw_dataset = LiberoHDF5Dataset(args.data_dir, action_horizon=ACTION_HORIZON, subsets=args.libero_subsets)
    else:
        raw_dataset = LiberoPlusDataset(args.data_dir, action_horizon=ACTION_HORIZON)

    class WrappedDataset(Dataset):
        def __len__(self):
            return len(raw_dataset)
        def __getitem__(self, i):
            return raw_dataset[i]

    if args.stage2_type == "router":
        assert args.nfe_labels_path is not None, "--nfe_labels_path required for router training"
        dataset = NFELabelsDataset(WrappedDataset(), args.nfe_labels_path)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: tuple(zip(*batch)),
            pin_memory=True,
            drop_last=True,
        )
    else:
        dataset = WrappedDataset()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    logger.info(f"Dataset: {len(dataset)} samples, {len(loader)} batches")

    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Train
    if args.stage2_type == "mas_head":
        model = train_mas_head(model, loader, optimizer, device, args.num_train_steps, args.log_interval)
    elif args.stage2_type == "router":
        model = train_router(model, loader, optimizer, device, args.num_train_steps, args.log_interval)

    # Save
    save_stage2_checkpoint(model, args.dynanfe_stage2_checkpoint, args.stage2_type)
    logger.info("Done!")


if __name__ == "__main__":
    main()
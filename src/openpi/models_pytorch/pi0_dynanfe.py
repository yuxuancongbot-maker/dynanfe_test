"""
DynaNFE: AdaFlow-Style Adaptive NFE Allocation for pi0.5

Core ideas (AdaFlow, NeurIPS 2024):
- Per-sample adaptive step sizes via variance prediction sigma(x_t, t|c)
- Step size: epsilon = eta / sigma
- Batch-parallel with early exit when all samples reach t=1

This module extends PI0Pytorch with:
1. MAS Head: predicts sigma for adaptive step sizing
2. NFE Router: classifies samples into discrete NFE levels
3. Adaptive sampling: replaces fixed-step flow matching loop
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

logger = logging.getLogger(__name__)


class MASHeadAdaFlow(nn.Module):
    """AdaFlow-Style Variance Prediction Head.

    Predicts log(sigma) conditioned on (prefix_embedding, noisy_actions, timestep).
    Output in log-space for numerical stability.
    """

    def __init__(self, emb_dim, action_dim, action_horizon, hidden_dim=256):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.time_embed_dim = 128

        input_dim = emb_dim + action_dim * action_horizon + self.time_embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if i == len(self.net) - 1:
                    nn.init.constant_(layer.bias, 0.0)
                else:
                    nn.init.zeros_(layer.bias)

        logger.info(f"MAS Head initialized: {sum(p.numel() for p in self.parameters())} parameters")

    def timestep_embedding(self, timesteps, dim):
        """Sinusoidal timestep embedding."""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, prefix_pooled, noisy_actions, timestep):
        """
        Args:
            prefix_pooled: (B, emb_dim)
            noisy_actions: (B, action_horizon, action_dim)
            timestep: (B,) in [0, 1]

        Returns:
            log_sqrt_var: (B,) — log(sigma)
        """
        B = prefix_pooled.shape[0]
        t_emb = self.timestep_embedding(timestep, self.time_embed_dim)
        action_flat = noisy_actions.reshape(B, -1)
        x = torch.cat([prefix_pooled, action_flat, t_emb], dim=-1)
        log_sqrt_var = self.net(x).squeeze(-1)
        return log_sqrt_var


class NFERouter(nn.Module):
    """NFE classification router.

    Predicts which discrete NFE level to use based on post-transformer
    prefix representation.
    """

    def __init__(self, emb_dim, nfe_options=(1, 2, 4), hidden_dim=256):
        super().__init__()
        self.nfe_options = nfe_options
        num_classes = len(nfe_options)

        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        logger.info(
            f"NFE Router initialized: {sum(p.numel() for p in self.parameters())} params, "
            f"nfe_options={nfe_options}"
        )

    def forward(self, prefix_pooled):
        """(B, emb_dim) -> (B, num_classes) logits."""
        return self.net(prefix_pooled)


class PI0PytorchWithDynaNFE(PI0Pytorch):
    """pi0.5 + DynaNFE (AdaFlow-Style Adaptive NFE).

    Inference modes:
      - use_dynanfe=True: AdaFlow adaptive step sizing via MAS head
      - use_nfe_router=True: discrete NFE classification via Router

    Training (Stage 2):
      - forward_stage2: train MAS head with Gaussian NLL loss
      - forward_router_stage2: train NFE Router with cross-entropy loss
    """

    def __init__(self, config):
        super().__init__(config)

        self.use_dynanfe = getattr(config, "use_dynanfe", False)
        self.use_nfe_router = getattr(config, "use_nfe_router", False)

        emb_dim = self.paligemma_with_expert.paligemma.config.text_config.hidden_size

        if self.use_nfe_router:
            nfe_options = getattr(config, "nfe_options", (1, 2, 4))
            router_hidden_dim = getattr(config, "router_hidden_dim", 256)

            self.nfe_router = NFERouter(
                emb_dim=emb_dim,
                nfe_options=nfe_options,
                hidden_dim=router_hidden_dim,
            )
            self.nfe_options = list(nfe_options)

            logger.info(f"NFE Router enabled: nfe_options={nfe_options}")

        elif self.use_dynanfe:
            mas_hidden_dim = getattr(config, "mas_hidden_dim", 256)

            self.mas_head = MASHeadAdaFlow(
                emb_dim=emb_dim,
                action_dim=self.config.action_dim,
                action_horizon=self.config.action_horizon,
                hidden_dim=mas_hidden_dim,
            )

            self.eta = getattr(config, "eta", 0.1)
            self.nfe_max = getattr(config, "nfe_max", 4)
            self.nfe_min_step = 1.0 / self.nfe_max

            logger.info(f"DynaNFE enabled: eta={self.eta}, nfe_max={self.nfe_max}")

    def compute_sigma(self, prefix_pooled, noisy_actions, timestep):
        """Predict sigma(x_t, t|c) via MAS head."""
        log_sqrt_var = self.mas_head(prefix_pooled, noisy_actions, timestep)
        return torch.exp(log_sqrt_var)

    def _encode_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Encode prefix (images + language) and compute KV cache.

        This is the most expensive part of inference (~80% of total time).
        Extracted as a separate method so torch.compile can optimize it.

        Returns:
            prefix_pooled: (B, emb_dim) mean-pooled post-transformer prefix
            prefix_pad_masks: (B, prefix_len) padding masks
            past_key_values: KV cache for subsequent denoise steps
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=model_dtype)
        prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=model_dtype)

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        mask_expanded = prefix_pad_masks.unsqueeze(-1).float()
        prefix_pooled = (prefix_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        return prefix_pooled, prefix_pad_masks, past_key_values

    def _get_prefix_pooled(self, observation):
        """Compute post-transformer prefix_pooled for Router/MAS head training.

        Returns prefix_pooled, prefix_embs, prefix_pad_masks, prefix_att_masks, state.
        All tensors are detached (no grad through the transformer).
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )

            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

            model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            embs = prefix_embs.to(dtype=model_dtype)
            prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=model_dtype)

            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )

            mask_expanded = prefix_pad_masks.unsqueeze(-1).float()
            prefix_pooled = (prefix_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        return prefix_pooled, prefix_embs, prefix_pad_masks, prefix_att_masks, state

    def _clone_kv_cache(self, past_key_values):
        """Clone KV cache to avoid cross-call side effects."""
        if isinstance(past_key_values, (list, tuple)):
            return type(past_key_values)(
                tuple(t.clone() for t in layer_kv) for layer_kv in past_key_values
            )
        elif hasattr(past_key_values, "key_cache"):
            past_key_values.key_cache = [k.clone() for k in past_key_values.key_cache]
            past_key_values.value_cache = [v.clone() for v in past_key_values.value_cache]
        return past_key_values

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=None, return_nfe=False):
        """Inference entry point. Dispatches to Router, AdaFlow, or base L1-flow.

        When use_nfe_router=True, uses base class sample_actions with router-determined NFE.
        When use_dynanfe=True, uses AdaFlow adaptive stepping.
        Otherwise, uses base model behavior (L1-flow or standard flow matching).
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        if self.use_nfe_router:
            # Router mode: predict NFE from prefix, then use base class sample_actions with compiled internals
            images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
                observation, train=False
            )
            prefix_pooled, prefix_pad_masks, past_key_values = self._encode_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            # Clone to avoid CUDAGraph buffer reuse issues
            prefix_pooled = prefix_pooled.clone()
            prefix_pad_masks = prefix_pad_masks.clone()
            past_key_values = self._clone_kv_cache(past_key_values)

            # Compute nfe BEFORE calling compiled methods (avoids graph break from .cpu())
            logits = self.nfe_router(prefix_pooled)
            nfe_class = logits.argmax(dim=-1)
            nfe_idx = int(nfe_class.max().detach().item())
            nfe = self.nfe_options[nfe_idx]

            if return_nfe:
                nfe_info = {
                    "num_steps_taken": torch.full((bsize,), float(nfe), device=device),
                    "mean_nfe": float(nfe),
                    "max_nfe": float(nfe),
                    "min_nfe": float(nfe),
                    "nfe_class": nfe_class,
                }
                actions = self._sample_with_router(
                    device,
                    prefix_pooled,
                    prefix_pad_masks,
                    past_key_values,
                    state,
                    noise,
                    bsize,
                    return_nfe=False,
                    num_steps=nfe,
                )
                return actions, nfe_info
            else:
                return self._sample_with_router(
                    device,
                    prefix_pooled,
                    prefix_pad_masks,
                    past_key_values,
                    state,
                    noise,
                    bsize,
                    return_nfe=False,
                    num_steps=nfe,
                )

        elif self.use_dynanfe:
            # AdaFlow mode: adaptive stepping
            images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
                observation, train=False
            )
            prefix_pooled, prefix_pad_masks, past_key_values = self._encode_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_pooled = prefix_pooled.clone()
            prefix_pad_masks = prefix_pad_masks.clone()
            past_key_values = self._clone_kv_cache(past_key_values)
            return self._sample_adaptive(
                device, prefix_pooled, prefix_pad_masks, past_key_values,
                state, noise, bsize, return_nfe,
            )
        else:
            # Use base model behavior (L1-flow or standard flow matching)
            if self.l1_flow:
                num_steps = 2
            return super().sample_actions(device, observation, noise, num_steps=num_steps or 10)

    def _sample_with_router(self, device, prefix_pooled, prefix_pad_masks,
                            past_key_values, state, noise, bsize, return_nfe, num_steps=None):
        """Router mode: predict NFE class, run fixed-step uniform denoising."""
        if num_steps is not None:
            nfe = int(num_steps)
            nfe_class = torch.full((bsize,), -1, dtype=torch.long, device=device)
        else:
            logits = self.nfe_router(prefix_pooled)  # (B, num_classes)
            nfe_class = logits.argmax(dim=-1)         # (B,)
            # For batch: use max NFE to ensure quality for all samples
            nfe_idx = int(nfe_class.max().detach().item())
            nfe = self.nfe_options[nfe_idx]

        dt = 1.0 / nfe
        z = noise.clone()

        for step in range(nfe):
            t = torch.full((bsize,), step * dt, device=device)
            x1_pred = self.denoise_step(
                state, prefix_pad_masks, past_key_values, z, t
            )
            if step == nfe - 1:
                z = x1_pred
            else:
                v = (x1_pred - z) / (1.0 - t[:, None, None]).clamp(min=1e-6)
                z = z + dt * v

        if return_nfe:
            nfe_info = {
                "num_steps_taken": torch.full((bsize,), float(nfe), device=device),
                "mean_nfe": float(nfe),
                "max_nfe": float(nfe),
                "min_nfe": float(nfe),
                "nfe_class": nfe_class,
            }
            return z, nfe_info
        return z

    def _sample_adaptive(self, device, prefix_pooled, prefix_pad_masks,
                         past_key_values, state, noise, bsize, return_nfe):
        """AdaFlow mode: adaptive step sizes via MAS head sigma predictions."""
        z = noise.clone()
        current_t = torch.zeros(bsize, device=device)
        nfe_per_sample = torch.zeros(bsize, device=device)

        for _ in range(self.nfe_max):
            remaining = (1.0 - current_t).clamp(min=0.0)

            if remaining.max().item() < 1e-6:
                break

            sigma = self.compute_sigma(prefix_pooled, z, current_t)

            step_size = (self.eta / sigma).clamp(max=remaining)
            step_size = torch.where(
                (remaining < self.nfe_min_step) & (remaining > 1e-6),
                remaining,
                step_size,
            )
            active = (step_size > 0).float()

            x1_pred = self.denoise_step(
                state, prefix_pad_masks, past_key_values, z, current_t
            )

            v = (x1_pred - z) / (1.0 - current_t[:, None, None]).clamp(min=1e-6)
            z = z + step_size[:, None, None] * v

            current_t = current_t + step_size
            nfe_per_sample = nfe_per_sample + active

        mean_nfe = nfe_per_sample.mean().item()
        max_nfe = nfe_per_sample.max().item()
        min_nfe = nfe_per_sample.min().item()
        logger.info(f"AdaFlow sampling: mean_NFE={mean_nfe:.1f}, max={max_nfe:.0f}, min={min_nfe:.0f}")

        if return_nfe:
            nfe_info = {
                "num_steps_taken": nfe_per_sample,
                "mean_nfe": mean_nfe,
                "max_nfe": max_nfe,
                "min_nfe": min_nfe,
            }
            return z, nfe_info
        return z

    def forward_stage2(self, observation, actions):
        """Stage 2 training: freeze flow model, train MAS head.

        Loss: Gaussian NLL = (error^2 / 2*sigma^2) + log(sigma)
        where error = ||x1_pred - actions||_L1
        """
        prefix_pooled, prefix_embs, prefix_pad_masks, prefix_att_masks, state = (
            self._get_prefix_pooled(observation)
        )

        B = actions.shape[0]
        t = torch.rand(B, device=actions.device)
        noise = torch.randn_like(actions)

        # Flow matching interpolation: z_t = t*actions + (1-t)*noise
        z_t = t[:, None, None] * actions + (1 - t[:, None, None]) * noise

        # Frozen flow model prediction
        with torch.no_grad():
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, z_t, t
            )

            model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            prefix_embs = prefix_embs.to(dtype=model_dtype)
            suffix_embs = suffix_embs.to(dtype=model_dtype)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            bsize = prefix_pad_masks.shape[0]
            prefix_att_masks_zero = torch.zeros(
                bsize, prefix_pad_masks.shape[1],
                dtype=torch.bool, device=prefix_pad_masks.device,
            )
            att_masks = torch.cat([prefix_att_masks_zero, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
            att_2d_masks_4d = att_2d_masks_4d.to(dtype=model_dtype)

            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = suffix_out[:, -self.config.action_horizon:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            x1_pred = self.action_out_proj(suffix_out)

        # MAS head variance prediction
        log_sqrt_var = self.mas_head(prefix_pooled, z_t, t)
        log_sqrt_var = torch.clamp(log_sqrt_var, min=-3.0, max=5.0)

        # L1 prediction error
        error = (x1_pred - actions).abs().sum(dim=(-1, -2))
        error = torch.clamp(error, max=100.0)

        # Gaussian NLL loss
        sigma_sq = torch.exp(2 * log_sqrt_var)
        mas_loss = (error.pow(2) / (2.0 * sigma_sq)) + log_sqrt_var
        mas_loss = mas_loss.mean()

        with torch.no_grad():
            sigma_pred = torch.exp(log_sqrt_var)
            mean_sigma = sigma_pred.mean().item()
            mean_error = error.mean().item()

            sigma_centered = sigma_pred - sigma_pred.mean()
            error_centered = error - error.mean()
            correlation = (sigma_centered * error_centered).sum() / (
                torch.sqrt((sigma_centered ** 2).sum() * (error_centered ** 2).sum()) + 1e-8
            )

            calibration_ratio = mean_sigma / (mean_error + 1e-8)
            step_sizes = self.eta / sigma_pred
            mean_step = step_sizes.mean().item()
            estimated_nfe = 1.0 / (mean_step + 1e-8)

        info = {
            "mas_loss": mas_loss.item(),
            "mean_sigma": mean_sigma,
            "mean_error": mean_error,
            "correlation": correlation.item(),
            "calibration": calibration_ratio,
            "mean_step": mean_step,
            "estimated_nfe": estimated_nfe,
        }

        return mas_loss, info

    def forward_router_stage2(self, observation, nfe_labels=None, nfe_labels_soft=None, class_weights=None):
        """Router Stage 2 training: classification loss on NFE labels.

        Supports both hard labels (cross-entropy) and soft labels (soft CE).
        """
        prefix_pooled = self._get_prefix_pooled(observation)[0]

        logits = self.nfe_router(prefix_pooled)  # (B, num_classes)

        class_weights_local = None
        if class_weights is not None:
            class_weights_local = class_weights.to(device=logits.device, dtype=logits.dtype)
            class_weights_local = class_weights_local[:len(self.nfe_options)]

        if nfe_labels_soft is not None:
            target = nfe_labels_soft.to(device=logits.device, dtype=logits.dtype)
            log_probs = F.log_softmax(logits, dim=-1)
            if class_weights_local is not None:
                weighted_target = target * class_weights_local.unsqueeze(0)
                denom = weighted_target.sum(dim=-1).clamp(min=1e-9)
                loss = -((weighted_target * log_probs).sum(dim=-1) / denom).mean()
            else:
                loss = -(target * log_probs).sum(dim=-1).mean()
            metric_labels = nfe_labels if nfe_labels is not None else target.argmax(dim=-1)
        else:
            if nfe_labels is None:
                raise ValueError("Either nfe_labels_soft or nfe_labels must be provided")
            metric_labels = nfe_labels
            loss = F.cross_entropy(logits, nfe_labels, weight=class_weights_local)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == metric_labels).float().mean()
            nfe_values = torch.tensor(self.nfe_options, device=preds.device, dtype=torch.float32)
            predicted_nfe = nfe_values[preds]

        info = {
            "router_loss": loss.item(),
            "accuracy": accuracy.item(),
            "mean_predicted_nfe": predicted_nfe.mean().item(),
        }
        for i, nfe_val in enumerate(self.nfe_options):
            info[f"class_{nfe_val}_pct"] = (preds == i).float().mean().item()

        return loss, info

    def enable_torch_compile(self, mode="default", dynamic=False):
        """Compile key inference methods for faster execution.

        Compiles: _encode_prefix, denoise_step, compute_sigma.
        The outer adaptive loop stays as Python for dynamic break support.
        """
        compile_kwargs = dict(mode=mode, dynamic=dynamic)
        self._encode_prefix = torch.compile(self._encode_prefix, **compile_kwargs)
        self.denoise_step = torch.compile(self.denoise_step, **compile_kwargs)
        self.compute_sigma = torch.compile(self.compute_sigma, **compile_kwargs)
        logger.info(f"torch.compile enabled: mode={mode}, dynamic={dynamic}")
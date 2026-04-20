"""
DynaNFE: AdaFlow-Style Adaptive NFE Allocation for π0.5

完全按照 AdaFlow 的实现方式：
- 每个样本独立的自适应步长
- 动态方差预测 σ(x_t, t|c)
- 批量并行计算
- 提前退出机制

参考:
- AdaFlow (NeurIPS 2024): adaflow/policy/adaflow_unet_image_policy.py
- 理论: Proposition 3.3 误差上界 W₂² ≤ ε² · σ²
"""
import logging
import torch
from torch import nn
import torch.nn.functional as F

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

logger = logging.getLogger(__name__)


class MASHeadAdaFlow(nn.Module):
    """
    AdaFlow-Style Variance Prediction Head

    输入:
        - prefix_embedding: (B, emb_dim) - VLM 特征
        - noisy_actions: (B, action_horizon, action_dim) - 当前噪声动作
        - timestep: (B,) - 当前时间步

    输出:
        - log_sqrt_var: (B,) - log(√σ²)，用于数值稳定性

    设计依据:
    - AdaFlow: 方差预测依赖 (x_t, t, c)
    - 使用 log-space 避免数值问题
    """
    def __init__(self, emb_dim, action_dim, action_horizon, hidden_dim=256):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Time embedding (类似 Diffusion 的 sinusoidal embedding)
        self.time_embed_dim = 128

        # 输入: prefix_emb + action_emb + time_emb
        input_dim = emb_dim + action_dim * action_horizon + self.time_embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # 输出 log(√σ²)
        )

        # 初始化：最后一层偏置设为小的正值
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if i == len(self.net) - 1:  # 最后一层
                    # 初始化 bias 为 0.0，使得初始 log(σ) ≈ 0，即 σ ≈ 1.0
                    nn.init.constant_(layer.bias, 0.0)
                else:
                    nn.init.zeros_(layer.bias)

        logger.info(f"MAS Head (AdaFlow-style) initialized: {sum(p.numel() for p in self.parameters())} parameters")

    def timestep_embedding(self, timesteps, dim):
        """
        Sinusoidal timestep embedding (from Diffusion models)

        Args:
            timesteps: (B,) in [0, 1]
            dim: embedding dimension

        Returns:
            emb: (B, dim)
        """
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
            log_sqrt_var: (B,) - log(√σ²)
        """
        B = prefix_pooled.shape[0]

        # Time embedding
        t_emb = self.timestep_embedding(timestep, self.time_embed_dim)  # (B, 128)

        # Flatten actions
        action_flat = noisy_actions.reshape(B, -1)  # (B, H*D)

        # Concatenate all features
        x = torch.cat([prefix_pooled, action_flat, t_emb], dim=-1)

        # Predict log(√σ²)
        log_sqrt_var = self.net(x).squeeze(-1)

        return log_sqrt_var


class PI0PytorchWithDynaNFEAdaFlow(PI0Pytorch):
    """
    π0.5 + DynaNFE (AdaFlow-Style)

    核心特点:
    1. 自适应步长: ε_t = η / σ(x_t, t|c)
    2. 动态方差预测: 每步都预测 σ
    3. 批量并行: 所有样本一起计算，但时间独立
    4. 提前退出: 样本到达 t=1 后停止

    训练策略:
        阶段 1: 训练 base flow model (不带 MAS Head)
        阶段 2: 冻结 flow model，训练 MAS Head

    推理策略:
        while t < 1:
            σ = MAS_Head(c, x_t, t)
            ε = η / σ
            x_{t+ε} = x_t + ε · v(x_t, t|c)
            t = t + ε
    """
    def __init__(self, config):
        super().__init__(config)

        # DynaNFE 配置
        self.use_dynanfe = getattr(config, "use_dynanfe", False)

        if self.use_dynanfe:
            # MAS Head (AdaFlow-style)
            emb_dim = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
            mas_hidden_dim = getattr(config, "mas_hidden_dim", 256)

            self.mas_head = MASHeadAdaFlow(
                emb_dim=emb_dim,
                action_dim=self.config.action_dim,
                action_horizon=self.config.action_horizon,
                hidden_dim=mas_hidden_dim
            )

            # 超参数
            self.eta = getattr(config, "eta", 1)  # 误差阈值
            self.nfe_max = getattr(config, "nfe_max", 5)  # 最大步数
            self.nfe_min_step = 1.0 / getattr(config, "nfe_max_theoretical", 5)  # 最小步长
            self.mas_loss_weight = getattr(config, "mas_loss_weight", 1.0)

            logger.info(f"DynaNFE (AdaFlow-style) enabled: eta={self.eta}, nfe_max={self.nfe_max}")

    def compute_sigma(self, prefix_pooled, noisy_actions, timestep):
        """
        计算方差预测 σ(x_t, t|c)

        Args:
            prefix_pooled: (B, emb_dim)
            noisy_actions: (B, action_horizon, action_dim)
            timestep: (B,) in [0, 1]

        Returns:
            sigma: (B,) - 预测的标准差
        """
        log_sqrt_var = self.mas_head(prefix_pooled, noisy_actions, timestep)
        sigma = torch.exp(log_sqrt_var)  # √σ²
        return sigma

    def forward_stage2(self, observation, actions):
        """
        阶段 2 训练: 冻结 flow model，训练 MAS Head

        使用 AdaFlow 的损失函数 (Gaussian NLL):
            L = (||v_true - v_pred||² / (2σ²)) + log(σ²)

        Args:
            observation: 观测数据
            actions: ground truth 动作

        Returns:
            mas_loss: MAS Head 的损失
            info: 训练信息字典
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True
        )

        # Embed prefix (VLM 部分，已冻结)
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )

        # Mean pooling
        mask_expanded = prefix_pad_masks.unsqueeze(-1).float()
        prefix_pooled = (prefix_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        B = actions.shape[0]

        # 采样 t ~ Uniform[0, 1]
        t = torch.rand(B, device=actions.device)

        # 采样噪声
        noise = torch.randn_like(actions)

        # 构造 z_t = t·a + (1-t)·z_0
        z_t = t[:, None, None] * actions + (1 - t[:, None, None]) * noise

        # 真实速度场: v* = a - z_0
        v_true = actions - noise

        # 预测速度场 (冻结)
        with torch.no_grad():
            # 这里需要调用 flow model 预测 v
            # 由于 π0.5 直接预测 x₁，我们需要转换
            # v = (x₁_pred - z_t) / (1 - t)

            # 构造 suffix
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, z_t, t
            )

            if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            # 构造 attention mask
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)

            bsize = prefix_pad_masks.shape[0]
            prefix_att_masks_zero = torch.zeros(
                bsize, prefix_pad_masks.shape[1],
                dtype=torch.bool, device=prefix_pad_masks.device
            )
            att_masks = torch.cat([prefix_att_masks_zero, suffix_att_masks], dim=1)

            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            # Forward
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

        # MAS Head 预测方差
        log_sqrt_var = self.mas_head(prefix_pooled, z_t, t)

        # Clamp log_sqrt_var 防止 σ 过大或过小
        # log(σ) ∈ [-3, 1] => σ ∈ [0.05, 2.72]
        log_sqrt_var = torch.clamp(log_sqrt_var, min=-3.0, max=1.0)

        # L1-Flow loss: 直接预测 x₁（终点）
        # error = ||x₁_pred - actions||
        error = (x1_pred - actions).abs().sum(dim=(-1, -2))  # (B,) L1 距离

        # Clip error 防止极端异常值
        error = torch.clamp(error, max=100.0)

        # Gaussian NLL loss (AdaFlow 公式)
        # L = (error² / (2σ²)) + log(σ²) / 2
        # 由于 log_sqrt_var = log(σ)，所以:
        # L = (error² / (2σ²)) + log(σ)
        # 注意: error 已经是 L1 距离，需要平方
        sigma_sq = torch.exp(2 * log_sqrt_var)  # σ²
        mas_loss = (error.pow(2) / (2.0 * sigma_sq)) + log_sqrt_var
        mas_loss = mas_loss.mean()

        # 统计信息
        with torch.no_grad():
            sigma_pred = torch.exp(log_sqrt_var)
            mean_sigma = sigma_pred.mean().item()
            mean_error = error.mean().item()  # L1 error，不需要 sqrt

            # 计算 σ 和 error 的相关性（Pearson correlation）
            # 这是判断 MAS Head 是否有效训练的关键指标
            sigma_centered = sigma_pred - sigma_pred.mean()
            error_centered = error - error.mean()
            correlation = (sigma_centered * error_centered).sum() / (
                torch.sqrt((sigma_centered ** 2).sum() * (error_centered ** 2).sum()) + 1e-8
            )

            # 校准误差：σ 和 error 的比值
            # 理想情况：ratio ≈ 1（σ 准确估计 error）
            calibration_ratio = mean_sigma / (mean_error + 1e-8)

            # 计算对应的步长 ε = η / σ
            step_sizes = self.eta / sigma_pred  # (B,)
            mean_step = step_sizes.mean().item()
            # 估算 NFE = 1 / mean_step（从 t=0 到 t=1 需要多少步）
            estimated_nfe = 1.0 / (mean_step + 1e-8)

        info = {
            'mas_loss': mas_loss.item(),
            'mean_sigma': mean_sigma,
            'mean_error': mean_error,
            'correlation': correlation.item(),
            'calibration': calibration_ratio,
            'mean_step': mean_step,
            'estimated_nfe': estimated_nfe,
        }

        return mas_loss, info

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=None, return_nfe=False):
        """
        AdaFlow-Style 自适应推理

        算法:
            z ← noise
            t ← 0
            while t < 1:
                σ ← MAS_Head(c, z, t)
                ε ← clip(η/σ, ε_min, 1-t)
                x₁ ← FlowModel(z, t|c)
                v ← (x₁ - z) / (1 - t)
                z ← z + ε·v
                t ← t + ε

        Args:
            device: 设备
            observation: 观测数据
            noise: 初始噪声 (可选)
            num_steps: 忽略 (DynaNFE 自适应决定)
            return_nfe: 是否返回 NFE 信息

        Returns:
            actions: (B, action_horizon, action_dim)
            如果 return_nfe=True，返回 (actions, nfe_info)
        """
        if not self.use_dynanfe:
            # 回退到原始 π0.5 推理
            return super().sample_actions(device, observation, noise, num_steps=2)

        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        # ===== 1. Embed prefix + KV cache =====
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # Mean pooling (用于 MAS Head)
        mask_expanded = prefix_pad_masks.unsqueeze(-1).float()
        prefix_pooled = (prefix_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # 计算 KV cache
        prefix_att_2d_masks = self._make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        # ===== 2. AdaFlow 自适应采样 =====
        z = noise.clone()
        current_t = torch.zeros(bsize, device=device)

        valid_action = torch.zeros_like(noise)
        valid_action_found = torch.zeros(bsize, dtype=torch.bool, device=device)
        num_steps_taken = torch.zeros(bsize, device=device)

        for i in range(self.nfe_max):
            # 预测方差
            sigma = self.compute_sigma(prefix_pooled, z, current_t)

            # 计算自适应步长: ε = η / σ
            step_size = self.eta / sigma

            # Clip 步长
            step_size = torch.clamp(
                step_size,
                min=self.nfe_min_step,
                max=1.0 - current_t
            )

            # 预测 x₁
            x1_pred = self._denoise_step(
                state, prefix_pad_masks, past_key_values, z, current_t
            )

            # 计算速度场: v = (x₁ - z) / (1 - t)
            v = (x1_pred - z) / (1 - current_t[:, None, None] + 1e-8)

            # Euler step: z ← z + ε·v
            z = z + step_size[:, None, None] * v

            # 更新时间
            current_t = current_t + step_size

            # 检查哪些样本已经到达 t=1
            if current_t.max() >= 1.0:
                mask = (current_t >= 1.0) & (~valid_action_found)
                valid_action[mask] = z[mask].clone()
                valid_action_found[mask] = True
                num_steps_taken[mask] = i + 1

            # 如果所有样本都完成了，提前退出
            if valid_action_found.all():
                break

        # 统计信息
        mean_nfe = num_steps_taken.float().mean().item()
        max_nfe = num_steps_taken.max().item()
        min_nfe = num_steps_taken.min().item()
        logger.debug(f"AdaFlow sampling: mean_NFE={mean_nfe:.2f}, max_NFE={max_nfe}, min_NFE={min_nfe}")

        if return_nfe:
            nfe_info = {
                'num_steps_taken': num_steps_taken,  # (B,) 每个样本的步数
                'mean_nfe': mean_nfe,
                'max_nfe': max_nfe,
                'min_nfe': min_nfe,
            }
            return valid_action, nfe_info

        return valid_action

    def _denoise_step(self, state, prefix_pad_masks, past_key_values, noisy_actions, timestep):
        """
        单步去噪 (辅助函数)

        Args:
            state: 状态
            prefix_pad_masks: prefix padding mask
            past_key_values: KV cache
            noisy_actions: 噪声动作
            timestep: 时间步 (B,)

        Returns:
            x1_pred: 预测的干净动作
        """
        # Embed suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, noisy_actions, timestep
        )

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # 构造 attention mask
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)

        # prefix_att_masks 全 0 (full attention)
        bsize = prefix_pad_masks.shape[0]
        prefix_att_masks = torch.zeros(
            bsize, prefix_pad_masks.shape[1],
            dtype=torch.bool, device=prefix_pad_masks.device
        )
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = self._make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Forward
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        x1_pred = self.action_out_proj(suffix_out)

        return x1_pred

    def _make_att_2d_masks(self, pad_masks, att_masks):
        """辅助函数: 构造 2D attention mask"""
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        return make_att_2d_masks(pad_masks, att_masks)

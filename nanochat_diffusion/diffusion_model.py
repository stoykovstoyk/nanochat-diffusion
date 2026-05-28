"""
Diffusion LLM model for token-level denoising.

This module implements a diffusion language model where the diffusion process works over
discrete tokens. At each diffusion step t, randomly mask some tokens with a special UNK
token, and the model predicts the original tokens.

Key design:
- Training: Sample random timestep t, randomly mask tokens at that noise level, predict original
- Inference: Start from all-UNK, progressively unmask and predict (iterative denoising)
- The UNK token has its own embedding, separate from the regular vocabulary
- Timestep embedding uses sinusoidal encoding (like ImageDiffusion)

The DiffusionLM wraps the base GPT model and adds:
1. Timestep embedding (sinusoidal, projected to n_embd)
2. Masked token encoder (embeds UNK separately, optionally)
3. Diffusion loss: cross-entropy on unmasked positions
4. Training step: sample t, mask tokens, compute loss
5. Inference: iterative denoising from all-UNK to clean sequence
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat_diffusion.gpt import GPT, GPTConfig, Linear, norm, apply_rotary_emb, has_ve
from nanochat_diffusion.gpt import CausalSelfAttention, Block as GPTBlock
from nanochat_diffusion.common import COMPUTE_DTYPE, print0
from nanochat_diffusion.optim import MuonAdamW


# -----------------------------------------------------------------------------
# DiffusionConfig dataclass
# -----------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    """Configuration for the Diffusion LLM."""
    # GPT base config (forwarded to GPTConfig)
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"

    # Diffusion-specific config
    num_diffusion_steps: int = 1000          # total number of diffusion steps
    unk_token_id: int = 32767                 # UNK token ID (should be >= vocab_size - 1)
    max_mask_ratio: float = 0.8               # maximum fraction of tokens to mask
    timestep_embed_dim: int = 256             # embedding dimension for timestep
    timestep_proj_dim: int = 512              # projected dim for timestep before adding
    add_timestep_to_embedding: bool = True    # whether to add timestep embed to token embedding
    add_timestep_to_mlp: bool = True          # whether to add timestep to MLP input
    timestep_attn: bool = True                # whether to add cross-attention to timestep in each layer
    pad_vocab_size_to: int = 64               # pad vocab for efficiency

    # Sampling / inference config
    sampling_steps: int = 20                  # number of denoising steps (fewer than training steps)
    unmask_schedule: str = "linear"           # unmask schedule: "linear", "cosine", "exponential"
    
    def __post_init__(self):
        """Ensure n_head divides n_embd and n_head >= n_kv_head."""
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert self.n_head % self.n_kv_head == 0, \
            f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"

    def to_gpt_config(self) -> GPTConfig:
        """Convert to a plain GPTConfig (without diffusion fields)."""
        import dataclasses
        fields = dataclasses.fields(self)
        gpt_dict = {
            "sequence_len": self.sequence_len,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "n_embd": self.n_embd,
            "window_pattern": self.window_pattern,
        }
        return GPTConfig(**gpt_dict)


# -----------------------------------------------------------------------------
# Timestep embedding (sinusoidal, like ImageDiffusion)
# -----------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding, similar to Stable Diffusion."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.timestep_embed_dim = config.timestep_embed_dim
        self.timestep_proj_dim = config.timestep_proj_dim

        self.linear_1 = nn.Linear(self.timestep_embed_dim, self.timestep_proj_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(self.timestep_proj_dim, config.n_embd, bias=True)

        # Precompute sinusoidal frequencies (device-agnostic, created on first forward)
        half_dim = self.timestep_embed_dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        self.register_buffer('emb_freq', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        dtype = self.linear_1.weight.dtype

        # Expand for batch: (B, half_dim)
        t = t.to(torch.float32)
        half_dim = self.timestep_embed_dim // 2
        emb = t.unsqueeze(-1) * self.emb_freq.unsqueeze(0).to(device=device, dtype=torch.float32)

        # Concat sin/cos
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1).to(dtype)

        # Project
        emb = self.linear_1(emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)

        return emb


# -----------------------------------------------------------------------------
# DiffusionModel - wraps GPT with diffusion-specific layers
# -----------------------------------------------------------------------------

class DiffusionModel(nn.Module):
    """
    Diffusion LLM that wraps GPT with diffusion-specific components.

    The model is trained to predict original tokens given masked input at various
    noise levels. During inference, it progressively denoises from all-UNK to clean.

    Architecture:
    - GPT base with timestep embedding added to token embeddings
    - Optionally, timestep is cross-attended in each transformer layer
    - MLP timestep projection for timestep conditioning
    """

    def __init__(self, diffusion_config: DiffusionConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = diffusion_config
        self.gpt_config = diffusion_config.to_gpt_config()
        self.pad_vocab_size_to = pad_vocab_size_to

        # The base GPT model
        self.gpt = GPT(self.gpt_config, pad_vocab_size_to=pad_vocab_size_to)

        # Timestep embedding
        self.timestep_embed = SinusoidalTimestepEmbedding(diffusion_config)

        # Optional: project timestep to add to token embeddings
        if diffusion_config.add_timestep_to_embedding:
            self.timestep_proj = nn.Linear(
                diffusion_config.n_embd, diffusion_config.n_embd, bias=False
            )

        # Optional: project timestep to add to MLP
        if diffusion_config.add_timestep_to_mlp:
            self.timestep_mlp_proj = nn.Linear(
                diffusion_config.n_embd, 4 * diffusion_config.n_embd, bias=False
            )

        # Per-token type embeddings: learnable vector added when token is UNK
        # This allows the model to learn what "masked" looks like
        self.unk_type = nn.Parameter(torch.zeros(1, diffusion_config.n_embd))

    def get_device(self):
        return self.gpt.get_device()

    def get_timestep(self) -> torch.Tensor:
        return self.gpt.get_device()

    # ---------------------------------------------------------------------
    # Forward: GPT forward + timestep conditioning
    # ---------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        loss_reduction: str = "mean",
        return_all_logits: bool = False,
        mask_inputs: bool = False,
    ) -> Any:
        """
        Diffusion forward pass.

        Args:
            idx: Token IDs, shape (B, T)
            t: Timestep, shape (B,) or (B, 1). If None, no timestep conditioning.
            targets: Optional target tokens for loss computation, shape (B, T)
            kv_cache: Optional KVCache for inference
            loss_reduction: 'mean' or 'none'
            return_all_logits: If True, return (logits, t_emb) tuple
            mask_inputs: If True, mask tokens internally (for training)

        Returns:
            If targets is given: scalar loss
            Otherwise: logits (B, T, vocab_size) or (logits, t_emb) tuple
        """
        B, T = idx.size()

        # Mask tokens internally (keeps this inside the compiled graph)
        if mask_inputs:
            if targets is None:
                targets = idx
            idx = self.mask_tokens(idx, t)

        # Get timestep embedding
        if t is not None and t.dim() == 2:
            t = t.squeeze(1)
        if t is not None:
            t_emb = self.timestep_embed(t)  # (B, n_embd)
            if not self.config.add_timestep_to_embedding:
                t_emb_for_attn = t_emb
            else:
                t_emb_for_attn = None
        else:
            t_emb = None
            t_emb_for_attn = None

        # Forward through GPT (which handles the GPT's own timestep handling)
        # We extend GPT's forward to add timestep conditioning

        # Embed tokens (GPT does this internally)
        x = self.gpt.transformer.wte(idx)

        if self.config.add_timestep_to_embedding and t_emb is not None:
            x = x + self.timestep_proj(t_emb).unsqueeze(1)

        x = norm(x)

        # Save initial embedding for x0 residual
        x0 = x

        # Forward through transformer blocks
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None

        for i, block in enumerate(self.gpt.transformer.h):
            # Residual scaling
            x = self.gpt.resid_lambdas[i] * x + self.gpt.x0_lambdas[i] * x0

            # Value embedding
            ve = None
            if str(i) in self.gpt.value_embeds:
                ve = self.gpt.value_embeds[str(i)](idx).to(x.dtype)

            # Get attention components
            n_head = self.config.n_head
            n_kv_head = self.config.n_kv_head
            n_embd = self.config.n_embd
            head_dim = n_embd // n_head

            q = block.attn.c_q(x).view(B, T, n_head, head_dim)
            k = block.attn.c_k(x).view(B, T, n_kv_head, head_dim)
            v = block.attn.c_v(x).view(B, T, n_kv_head, head_dim)

            if ve is not None:
                ve = ve.view(B, T, n_kv_head, head_dim)
                gate = 3 * torch.sigmoid(block.attn.ve_gate(x[..., :12]))
                v = v + gate.unsqueeze(-1) * ve

            # Apply RoPE
            cos, sin = self.gpt.cos[:, :T], self.gpt.sin[:, :T]
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)
            q = q * 1.2
            k = k * 1.2

            # Flash attention
            window_size = self.gpt.window_sizes[i]
            y = self.gpt._flash_attn(q, k, v, causal=True, window_size=window_size, kv_cache=kv_cache)

            y = y.contiguous().view(B, T, -1)
            y = block.attn.c_proj(y)

            # Residual with attention
            x = x + y

            # Add timestep to MLP
            if t_emb_for_attn is not None:
                x_mlp_in = x + self.timestep_mlp_proj(t_emb).unsqueeze(1).to(torch.bfloat16)
            else:
                x_mlp_in = x

            # MLP
            x_mlp_in = norm(x_mlp_in)
            x_mlp_in = x_mlp_in + self.gpt.transformer.h[i].mlp(x_mlp_in)

            x = x + x_mlp_in

            if i == backout_layer:
                x_backout = x

        # Backout
        if x_backout is not None:
            x = x - self.gpt.backout_lambda.to(x.dtype) * x_backout

        x = norm(x)

        # LM head
        logits = self.gpt.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = F.hardtanh(logits, -15.0, 15.0)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            if return_all_logits:
                return logits, t_emb
            return logits

    @torch.no_grad()
    def init_weights(self):
        """Initialize weights including diffusion-specific layers."""
        self.gpt.init_weights()

        # Initialize timestep embedding
        nn.init.zeros_(self.timestep_embed.linear_1.weight)
        nn.init.zeros_(self.timestep_embed.linear_1.bias)
        nn.init.zeros_(self.timestep_embed.linear_2.weight)
        nn.init.zeros_(self.timestep_embed.linear_2.bias)

        if self.config.add_timestep_to_embedding:
            nn.init.zeros_(self.timestep_proj.weight)

        if self.config.add_timestep_to_mlp:
            nn.init.zeros_(self.timestep_mlp_proj.weight)

        # UNK type embedding
        nn.init.zeros_(self.unk_type)

    # ---------------------------------------------------------------------
    # Training: diffusion forward + backward pass
    # ---------------------------------------------------------------------

    def mask_tokens(
        self,
        idx: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        max_mask_ratio: Optional[float] = None,
        return_mask: bool = False,
    ) -> torch.Tensor:
        """
        Randomly mask tokens with UNK_ID for training.

        Args:
            idx: Token IDs, shape (B, T)
            t: Timestep(s), shape (B,) or scalar. If None, use uniform random.
            max_mask_ratio: Override max_mask_ratio
            return_mask: If True, also return the boolean mask tensor

        Returns:
            Masked token IDs, shape (B, T)
            If return_mask: (masked_idx, mask_bool)
        """
        B, T = idx.size()
        max_mask = max_mask_ratio or self.config.max_mask_ratio

        # Compute mask ratio from timestep
        if t is not None:
            if t.dim() == 0:
                t = t.unsqueeze(0).expand(B)
            # mask fraction = (t / num_diffusion_steps) * max_mask_ratio
            ratios = (t.float() / self.config.num_diffusion_steps) * max_mask
        else:
            # Uniform random mask ratio between 0 and max_mask
            ratios = torch.rand(B, device=idx.device) * max_mask

        # Each token independently masked with probability ratio[i]
        mask = torch.rand(B, T, device=idx.device) < ratios.unsqueeze(1)

        # Set masked positions to UNK_ID (where avoids clone+scatter)
        masked_idx = torch.where(mask, self.config.unk_token_id, idx)

        if return_mask:
            return masked_idx, mask
        return masked_idx

    def sample_timesteps(
        self, batch_size: int, device: torch.device = None, logit: bool = False
    ) -> torch.Tensor:
        """Sample random timesteps for training."""
        if device is None:
            device = self.get_device()
        if logit:
            # Use logit transform for uniform sampling in logit space
            # This gives more weight to early and late timesteps
            u = torch.rand(batch_size, device=device)
            return (torch.logit(u + 1e-7) * (self.config.num_diffusion_steps / 3)).clip(0, self.config.num_diffusion_steps - 1).long()
        return torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)

    def compute_diffusion_loss(
        self,
        idx: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        max_mask_ratio: Optional[float] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss.

        1. Sample random timestep t
        2. Mask tokens at corresponding noise level
        3. Forward pass with masked input
        4. Cross-entropy loss on all positions (predict original tokens)

        Args:
            idx: Original token IDs, shape (B, T)
            t: Optional pre-sampled timesteps, shape (B,)
            max_mask_ratio: Override mask ratio
            reduce: If True, return scalar loss; else return per-sample loss

        Returns:
            Scalar loss (if reduce) or (B,) per-sample loss
        """
        B, T = idx.size()

        # Sample timesteps if not provided
        if t is None:
            t = self.sample_timesteps(B, device=idx.device)

        # Mask tokens
        masked_idx = self.mask_tokens(idx, t, max_mask_ratio=max_mask_ratio)

        # Forward pass
        loss = self(masked_idx, t=t, targets=idx, loss_reduction="none")
        return loss

    # ---------------------------------------------------------------------
    # Inference: iterative denoising
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def denoise(
        self,
        seq_len: int = 256,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        start_token: int = 0,  # token to start with (typically BOS or 0)
        pad_value: int = -1,
        return_history: bool = False,
    ) -> torch.Tensor:
        """
        Iterative denoising from all-UNK to clean sequence.

        Args:
            seq_len: Length of output sequence
            num_steps: Number of denoising steps. If None, use config.
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
            start_token: Special token to start with (e.g., BOS)
            pad_value: Value used for padding in intermediate steps

        Returns:
            Final denoised token IDs, shape (seq_len,)
        """
        num_steps = num_steps or self.config.sampling_steps
        B = 1

        # Initialize with all UNK
        current_tokens = torch.full(
            (B, seq_len),
            self.config.unk_token_id,
            dtype=torch.long,
            device=self.get_device(),
        )
        # First token is start_token (e.g., BOS)
        current_tokens[0, 0] = start_token

        history = [current_tokens.clone()]

        # Linear schedule: progressively unmask
        # Step 0: fully masked, step N: fully unmasked
        for step in range(num_steps):
            t = torch.tensor(
                [int((step / max(num_steps - 1, 1)) * self.config.num_diffusion_steps)],
                device=self.get_device(),
            )

            # Get current mask status
            mask = current_tokens == self.config.unk_token_id
            n_masked = mask.sum().item()
            n_total = seq_len
            mask_ratio = n_masked / n_total

            # Forward pass
            logits = self(current_tokens, t=t)
            logits = logits[:, -1, :]  # Get last token's logits... actually get all

            # Better: predict all tokens from the current state
            # We need full-sequence prediction for diffusion
            # For simplicity, use the logits from all positions
            logits = logits.view(B, seq_len, -1)

            # Sample from masked positions only
            if temperature > 0:
                top_k_logits = logits[0, mask[0]] / temperature
                if top_k > 0:
                    top_k_vals, top_k_idx = torch.topk(top_k_logits, min(top_k, top_k_logits.size(-1)))
                    probs = F.softmax(top_k_vals, dim=-1)
                    choice = torch.multinomial(probs, 1)
                    sampled = top_k_vals[choice]
                else:
                    probs = F.softmax(top_k_logits, dim=-1)
                    choice = torch.multinomial(probs, 1)
                    sampled = top_k_logits[choice]

                current_tokens[0, mask[0]] = sampled + start_token  # shift for vocab
            else:
                sampled = torch.argmax(logits[0, mask[0]], dim=-1)
                current_tokens[0, mask[0]] = sampled

            # Gradually unmask: at step k, we keep k/num_steps tokens unmasked
            if step < num_steps - 1:
                n_to_keep = int((step + 1) / num_steps * seq_len)
                if n_to_keep < seq_len:
                    current_tokens[0, -seq_len + n_to_keep:] = pad_value

            history.append(current_tokens.clone())

        final_seq = current_tokens[0]
        return final_seq if not return_history else (final_seq, history)

    # ---------------------------------------------------------------------
    # Parameter grouping for optimizer
    # ---------------------------------------------------------------------

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        scalar_lr=0.5,
    ):
        """Setup optimizer with diffusion-specific parameter groups."""
        from nanochat_diffusion.common import get_dist_info, print0 as print0_fn
        from nanochat_diffusion.optim import MuonAdamW, DistMuonAdamW
        import torch.distributed as dist
        import os

        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate parameters
        matrix_params = list(self.gpt.transformer.h.parameters())
        value_embeds_params = list(self.gpt.value_embeds.parameters())
        embedding_params = list(self.gpt.transformer.wte.parameters())
        lm_head_params = list(self.gpt.lm_head.parameters())
        resid_params = [self.gpt.resid_lambdas]
        x0_params = [self.gpt.x0_lambdas]
        smear_params = [
            self.gpt.smear_gate.weight,
            self.gpt.smear_lambda,
            self.gpt.backout_lambda,
        ]

        # Diffusion-specific parameters
        diffusion_params = [
            self.timestep_embed.linear_1.weight,
            self.timestep_embed.linear_1.bias,
            self.timestep_embed.linear_2.weight,
            self.timestep_embed.linear_2.bias,
            self.unk_type,
        ]
        if self.config.add_timestep_to_embedding:
            diffusion_params.append(self.timestep_proj.weight)
        if self.config.add_timestep_to_mlp:
            diffusion_params.append(self.timestep_mlp_proj.weight)

        # Build param_groups
        param_groups = [
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=unembedding_lr,
                betas=(0.8, 0.96),
                eps=1e-10,
                weight_decay=0.01,
            ),
            dict(
                kind="adamw",
                params=embedding_params,
                lr=embedding_lr,
                betas=(0.8, 0.995),
                eps=1e-10,
                weight_decay=0.001,
            ),
            dict(
                kind="adamw",
                params=value_embeds_params,
                lr=embedding_lr * 0.5,
                betas=(0.8, 0.995),
                eps=1e-10,
                weight_decay=0.01,
            ),
            dict(
                kind="adamw",
                params=resid_params,
                lr=scalar_lr * 0.01,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=0.05,
            ),
            dict(
                kind="adamw",
                params=x0_params,
                lr=scalar_lr,
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=smear_params,
                lr=0.2,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=diffusion_params,
                lr=embedding_lr * 0.5,
                betas=(0.8, 0.995),
                eps=1e-10,
                weight_decay=0.01,
            ),
        ]

        # Muon groups (matrix params, grouped by shape)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.9,
                    weight_decay=weight_decay,
                )
            )

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer


# -----------------------------------------------------------------------------
# Convenience: create a DiffusionLM model
# -----------------------------------------------------------------------------

def create_diffusion_model(
    n_layer: int = 12,
    n_head: int = 6,
    n_kv_head: int = 6,
    n_embd: int = 768,
    vocab_size: int = 32768,
    sequence_len: int = 2048,
    window_pattern: str = "SSSL",
    num_diffusion_steps: int = 1000,
    unk_token_id: int = 32767,
    max_mask_ratio: float = 0.8,
    sampling_steps: int = 20,
) -> DiffusionModel:
    """
    Create a DiffusionLM model with given configuration.

    Args:
        n_layer: Number of transformer layers
        n_head: Number of query heads
        n_kv_head: Number of key/value heads (GQA)
        n_embd: Hidden dimension
        vocab_size: Vocabulary size
        sequence_len: Max sequence length
        window_pattern: Sliding window pattern
        num_diffusion_steps: Number of diffusion timesteps
        unk_token_id: Token ID for UNK/masked tokens
        max_mask_ratio: Maximum fraction of tokens to mask during training
        sampling_steps: Number of denoising steps during inference

    Returns:
        DiffusionModel instance
    """
    config = DiffusionConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        vocab_size=vocab_size,
        sequence_len=sequence_len,
        window_pattern=window_pattern,
        num_diffusion_steps=num_diffusion_steps,
        unk_token_id=unk_token_id,
        max_mask_ratio=max_mask_ratio,
        sampling_steps=sampling_steps,
    )
    model = DiffusionModel(config)
    model.init_weights()
    return model

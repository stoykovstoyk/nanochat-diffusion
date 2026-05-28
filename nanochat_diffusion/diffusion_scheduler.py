"""
Diffusion noise schedules and masking strategies.
"""

import torch
import math
from typing import Optional, Tuple, List


class NoiseSchedule:
    """Base class for noise schedules."""
    
    def __init__(self, num_steps: int = 1000, max_mask: float = 0.8):
        self.num_steps = num_steps
        self.max_mask = max_mask
    
    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        """Get mask ratio for timestep t."""
        raise NotImplementedError
    
    def get_timesteps(self, num_steps: Optional[int] = None) -> torch.Tensor:
        """Get timesteps for inference."""
        raise NotImplementedError


class LinearNoiseSchedule(NoiseSchedule):
    """Linear noise schedule - mask ratio increases linearly with t."""
    
    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        return (t.float() / self.num_steps) * self.max_mask
    
    def get_timesteps(self, num_steps: Optional[int] = None) -> torch.Tensor:
        n = num_steps or self.num_steps
        return torch.linspace(0, self.num_steps, n + 1, dtype=torch.long)[1:]


class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule - matches the cosine schedule used in DDPM."""
    
    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        t_frac = t.float() / self.num_steps
        ratio = self.max_mask * (1 + math.cos(math.pi * t_frac / 2)) / 2
        return ratio
    
    def get_timesteps(self, num_steps: Optional[int] = None) -> torch.Tensor:
        n = num_steps or self.num_steps
        return torch.linspace(0, self.num_steps, n + 1, dtype=torch.long)[1:]


class ExponentialNoiseSchedule(NoiseSchedule):
    """Exponential noise schedule - masks increase exponentially."""
    
    def __init__(self, num_steps=1000, max_mask=0.8, alpha=2.0):
        super().__init__(num_steps, max_mask)
        self.alpha = alpha
    
    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        t_frac = t.float() / self.num_steps
        ratio = self.max_mask * (torch.exp(self.alpha * t_frac) - 1) / (torch.exp(self.alpha) - 1)
        return ratio
    
    def get_timesteps(self, num_steps: Optional[int] = None) -> torch.Tensor:
        n = num_steps or self.num_steps
        return torch.linspace(0, self.num_steps, n + 1, dtype=torch.long)[1:]


class ConstantMaskSchedule(NoiseSchedule):
    """Constant mask ratio - always mask the same fraction."""
    
    def __init__(self, num_steps=1000, max_mask=0.5):
        super().__init__(num_steps, max_mask)
    
    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.max_mask)
    
    def get_timesteps(self, num_steps: Optional[int] = None) -> torch.Tensor:
        n = num_steps or self.num_steps
        return torch.full((n,), self.num_steps, dtype=torch.long)


def create_noise_schedule(name: str = "linear", **kwargs) -> NoiseSchedule:
    """Factory function to create noise schedules."""
    schedules = {
        "linear": LinearNoiseSchedule,
        "cosine": CosineNoiseSchedule,
        "exponential": ExponentialNoiseSchedule,
        "constant": ConstantMaskSchedule,
    }
    schedule_class = schedules.get(name.lower())
    if schedule_class is None:
        raise ValueError(f"Unknown noise schedule: {name}")
    return schedule_class(**kwargs)


def mask_tokens_simple(
    idx: torch.Tensor,
    t: torch.Tensor,
    unk_token_id: int = 4095,
    max_mask_ratio: float = 0.8,
    schedule: Optional[NoiseSchedule] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask tokens with UNK_ID for diffusion training.
    
    Args:
        idx: Token IDs, shape (B, T)
        t: Timestep(s), shape (B,) or scalar
        unk_token_id: The UNK token ID to use for masking
        max_mask_ratio: Maximum fraction of tokens to mask
        schedule: Optional noise schedule for mask ratio
    
    Returns:
        (masked_idx, mask_bool) where mask_bool indicates which tokens are masked
    """
    B, T = idx.size()
    
    # Get mask ratio from timestep
    if schedule is not None:
        mask_ratio = schedule.get_mask_ratio(t)
    else:
        mask_ratio = (t.float() / 1000) * max_mask
    
    # Expand for broadcasting: (B, 1)
    mask_ratio = mask_ratio.view(B, 1)
    
    # Each token independently masked with probability mask_ratio
    mask = torch.rand(B, T, device=idx.device) < mask_ratio
    
    # Set masked positions to UNK_ID
    masked_idx = idx.clone()
    masked_idx[mask] = unk_token_id
    
    return masked_idx, mask


def unmask_tokens_progressively(
    idx: torch.Tensor,
    mask: torch.Tensor,
    predicted_tokens: torch.Tensor,
    unmask_ratio: float = 0.05,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Progressive unmasking for diffusion inference.
    
    Takes a fraction of currently-masked tokens and replaces them with
    predicted values from the previous step.
    
    Args:
        idx: Current token sequence (B, T)
        mask: Current mask (B, T) - True means masked
        predicted_tokens: Predicted tokens from previous step (B, T)
        unmask_ratio: Fraction of masked tokens to unmask per step
        deterministic: If True, unmask the highest-confidence tokens
    
    Returns:
        (updated_idx, updated_mask)
    """
    B, T = idx.size()
    
    # Number of tokens to unmask
    num_masked = mask.sum(dim=1, keepdim=True)  # (B, 1)
    num_to_unmask = torch.clamp(
        (num_masked * unmask_ratio).long(), min=1, max=num_masked
    )
    
    if deterministic:
        # Unmask highest confidence (this would need confidence scores)
        # For now, just use random
        pass
    
    # Randomly select masked tokens to unmask
    mask_probs = torch.where(mask, torch.rand_like(mask, dtype=torch.float32), -1.0)
    _, top_k = torch.topk(mask_probs, num_to_unmask.squeeze(1), dim=1)
    top_k_expanded = top_k.unsqueeze(2).expand(-1, -1, 1)  # (B, k, 1)
    
    # Create new mask
    updated_mask = mask.clone()
    updated_idx = idx.clone()
    
    # Unmask selected tokens and update with predicted values
    batch_idx = torch.arange(B).unsqueeze(1).expand_as(top_k)
    updated_idx[batch_idx, top_k] = predicted_tokens[batch_idx, top_k]
    updated_mask[batch_idx, top_k] = False
    
    return updated_idx, updated_mask


def generate_unmask_schedule(
    num_steps: int = 20,
    schedule_type: str = "linear",
    initial_mask: float = 1.0,
    final_mask: float = 0.05,
) -> List[float]:
    """Generate a schedule for progressive unmasking."""
    steps = list(range(num_steps + 1))
    
    if schedule_type == "linear":
        unmask_ratios = [initial_mask + (final_mask - initial_mask) * (t / num_steps) for t in steps[1:]]
    elif schedule_type == "cosine":
        unmask_ratios = [
            initial_mask + (final_mask - initial_mask) * (1 + math.cos(math.pi * t / num_steps)) / 2
            for t in steps[1:]
        ]
    elif schedule_type == "exponential":
        alpha = 2.0
        unmask_ratios = [
            initial_mask + (final_mask - initial_mask) * (torch.exp(alpha * t / num_steps) - 1) / (torch.exp(alpha) - 1)
            for t in steps[1:]
        ]
    else:
        raise ValueError(f"Unknown unmask schedule: {schedule_type}")
    
    return unmask_ratios

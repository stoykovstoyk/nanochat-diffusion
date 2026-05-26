"""
Diffusion sampler for diffusion LLM inference.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from nanochat_diffusion.diffusion_scheduler import (
    mask_tokens_simple,
    generate_unmask_schedule,
    LinearNoiseSchedule,
)
from nanochat_diffusion.diffusion_model import DiffusionModel


class DiffusionSampler:
    """
    Sampler for diffusion LLM inference.
    
    Implements iterative denoising where:
    1. Start with all tokens as UNK
    2. Progressive unmasking: predict masked tokens and update
    3. Repeat until all tokens are determined
    """
    
    def __init__(self, model: DiffusionModel, verbose: bool = False):
        self.model = model
        self.verbose = verbose
    
    @torch.no_grad()
    def sample(
        self,
        seq_len: int = 256,
        num_samples: int = 1,
        num_steps: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        start_token: int = 0,
        unmask_schedule: Optional[List[float]] = None,
        mask_ratio_schedule: Optional[List[float]] = None,
        return_history: bool = False,
        progress_callback=None,
    ) -> torch.Tensor:
        """
        Sample from diffusion LLM.
        
        Args:
            seq_len: Sequence length to generate
            num_samples: Number of samples to generate
            num_steps: Number of denoising steps
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            start_token: Start token ID (e.g., BOS)
            unmask_schedule: Custom unmask schedule
            mask_ratio_schedule: Custom mask ratio schedule
            return_history: If True, return history of intermediate states
            progress_callback: Callback for progress updates
        
        Returns:
            Generated token IDs, shape (num_samples, seq_len)
        """
        device = self.model.get_device()
        dtype = next(self.model.parameters()).dtype
        
        # Initialize with all UNK tokens
        B = num_samples
        T = seq_len
        unk_id = self.model.config.unk_token_id
        
        # Start with all UNK
        current_tokens = torch.full(
            (B, T), unk_id, dtype=torch.long, device=device
        )
        
        # Mark first token as start_token (BOS)
        current_tokens[:, 0] = start_token
        
        # Track which positions are determined (not UNK)
        is_determined = torch.zeros(B, T, dtype=torch.bool, device=device)
        is_determined[:, 0] = True  # First token is always determined
        
        # Generate unmask schedule if not provided
        if unmask_schedule is None:
            unmask_schedule = generate_unmask_schedule(num_steps=num_steps)
        
        history = [] if return_history else None
        
        for step in range(num_steps):
            if progress_callback:
                progress_callback(step, num_steps)
            
            # Current number of determined positions
            num_determined = is_determined.sum(dim=1).item()
            num_total = T
            
            # Fraction to unmask this step
            unmask_frac = unmask_schedule[step] if step < len(unmask_schedule) else 0.0
            
            # Get number of positions to unmask
            num_to_unmask = max(1, int((num_total - num_determined) * unmask_frac))
            
            # Find UNK positions
            unk_mask = (current_tokens == unk_id) & (~is_determined)
            num_unk = unk_mask.sum(dim=1).item()
            
            if num_unk == 0:
                break  # All tokens determined
            
            # Sample positions to unmask
            num_unmask = min(num_to_unmask, num_unk)
            unmask_indices = []
            for b in range(B):
                unk_pos = torch.where(unk_mask[b])[0]
                if len(unk_pos) > 0:
                    chosen = torch.randperm(len(unk_pos), device=device)[:num_unmask]
                    unmask_indices.append(unk_pos[chosen])
            
            # Forward pass to predict all positions
            # Use all-UNK sequence
            input_tokens = current_tokens.clone()
            logits = self.model(input_tokens)  # (B, T, vocab_size)
            
            # Sample from logits for UNK positions
            for b in range(B):
                if b < len(unmask_indices):
                    positions = unmask_indices[b]
                    
                    # Get logits for these positions
                    b_logits = logits[b, positions]  # (num_unmask, vocab_size)
                    
                    # Apply temperature
                    if temperature > 0:
                        b_logits = b_logits / temperature
                        
                        # Top-k filtering
                        if top_k > 0:
                            top_k_values, _ = torch.topk(b_logits, min(top_k, b_logits.size(-1)))
                            min_val = top_k_values[:, -1].unsqueeze(1)
                            mask = b_logits < min_val
                            b_logits = b_logits.masked_fill(mask, -1e10)
                        
                        # Top-p filtering
                        if top_p > 0:
                            probs = F.softmax(b_logits, dim=-1)
                            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                            cumsum = torch.cumsum(sorted_probs, dim=-1)
                            mask = cumsum < top_p
                            mask[:, -1] = True  # Always keep the last one
                            b_logits = b_logits.masked_fill(mask, -1e10)
                        
                        probs = F.softmax(b_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, 1).squeeze(1)
                    else:
                        next_tokens = torch.argmax(b_logits, dim=-1)
                    
                    # Update tokens
                    current_tokens[b, positions] = next_tokens
                    is_determined[b, positions] = True
            
            # Store history
            if return_history and history is not None:
                history.append(current_tokens.clone())
        
        if return_history and history is not None:
            return current_tokens, history
        
        return current_tokens
    
    @torch.no_grad()
    def generate(
        self,
        prompt: List[int],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        seed: int = 42,
    ) -> List[int]:
        """
        Generate text by autoregressive sampling using diffusion LLM.
        
        This works by:
        1. Encode prompt
        2. For each new token:
           - Pad to full sequence with UNK
           - Run diffusion sampling (one step)
           - Extract new token from sampled sequence
        """
        import random
        device = self.model.get_device()
        
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        
        # Start with prompt
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
        current_tokens = prompt_tensor.tolist()
        
        for _ in range(max_tokens):
            # Pad to sequence length with UNK
            seq_len = len(current_tokens) + max(1, max_tokens - len(current_tokens))
            padded = current_tokens + [self.model.config.unk_token_id] * (seq_len - len(current_tokens))
            
            input_seq = torch.tensor([padded], dtype=torch.long, device=device)
            logits = self.model(input_seq)
            
            # Get next token from UNK positions
            last_pos = len(current_tokens) - 1
            logits_last = logits[0, last_pos]
            
            if temperature > 0:
                logits_last = logits_last / temperature
                if top_k > 0:
                    top_k_values, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                    min_val = top_k_values[-1]
                    mask = logits_last < min_val
                    logits_last = logits_last.masked_fill(mask, -1e10)
                probs = F.softmax(logits_last, dim=-1)
                next_token = torch.multinomial(probs, 1, generator=rng).item()
            else:
                next_token = torch.argmax(logits_last).item()
            
            current_tokens.append(next_token)
            
            # Check for stop token
            if next_token == 0:  # BOS as stop
                break
        
        return current_tokens[len(prompt):]

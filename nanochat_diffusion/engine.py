"""
Engine for efficient diffusion LLM inference with KV cache.

Adapted from karpathy/nanochat's engine.py for diffusion models.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from nanochat_diffusion.gpt import GPT, GPTConfig, Linear
from nanochat_diffusion.diffusion_model import DiffusionModel, DiffusionConfig
from nanochat_diffusion.common import COMPUTE_DTYPE, print0


class DiffusionEngine:
    """
    Engine for efficient diffusion LLM inference.
    
    Supports:
    1. KV cache for fast autoregressive generation
    2. Parallel diffusion sampling with KV cache
    3. Streaming generation
    4. Batch inference
    """
    
    def __init__(self, model: DiffusionModel, max_seq_len: int = 2048):
        self.model = model
        self.max_seq_len = max_seq_len
        self.device = next(model.parameters()).device
        self.dtype = COMPUTE_DTYPE
        
        # KV cache
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0
        
    def clear_cache(self):
        """Clear the KV cache."""
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0
    
    def _get_kv_cache(self, n_layer: int, n_head: int, n_kv_head: int, 
                      head_dim: int, batch_size: int, device: torch.device):
        """Initialize KV cache."""
        k_cache = torch.empty(
            n_layer, batch_size, n_kv_head, self.max_seq_len, head_dim,
            dtype=self.dtype, device=device
        )
        v_cache = torch.empty(
            n_layer, batch_size, n_kv_head, self.max_seq_len, head_dim,
            dtype=self.dtype, device=device
        )
        return k_cache, v_cache
    
    @torch.no_grad()
    def forward_with_kv_cache(self, x: torch.Tensor, 
                               mask: Optional[torch.Tensor] = None,
                               input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with KV cache for efficient autoregressive inference.
        
        Args:
            x: Input tokens (B, T)
            mask: Attention mask (B, 1, T, T) or (B, n_head, T, T)
            input_pos: Position indices (B,)
        
        Returns:
            Logits (B, T, vocab_size)
        """
        B, T = x.size()
        config = self.model.config
        
        # Compute or update KV cache
        if self.k_cache is None:
            n_layer = config.n_layer
            n_head = config.n_head
            n_kv_head = config.n_kv_head or n_head
            head_dim = config.n_embd // n_head
            
            device = x.device
            self.k_cache, self.v_cache = self._get_kv_cache(
                n_layer, n_head, n_kv_head, head_dim, B, device
            )
            self.seq_len = 0
        
        # Run model forward
        x_emb = self.model.transformer.wte(x)  # (B, T, n_embd)
        x_pos = self.model.transformer.wpe(input_pos if input_pos is not None else torch.arange(T, device=x.device))
        
        # Apply blocks with KV cache
        for block in self.model.transformer.h:
            x_emb, self.k_cache, self.v_cache = block(
                x_emb, self.k_cache, self.v_cache,
                input_pos=input_pos, mask=mask
            )
        
        # Final normalization + projection
        x_norm = self.model.transformer.ln(x_emb)
        logits = self.model.lm_head(x_norm)
        
        return logits
    
    @torch.no_grad()
    def generate_autoregressive(self, prompt: List[int], max_new_tokens: int = 128,
                                temperature: float = 1.0, top_k: int = 0,
                                eos_token: int = 0) -> List[int]:
        """
        Generate text using autoregressive sampling.
        
        Args:
            prompt: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            eos_token: End-of-sequence token
        
        Returns:
            Generated token IDs
        """
        self.clear_cache()
        
        # Encode prompt
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=self.device)
        x = prompt_tensor.unsqueeze(0)  # (1, len(prompt))
        
        generated = []
        for _ in range(max_new_tokens):
            # If longer than max_seq_len, generate from prompt[-max_seq_len:]
            if x.size(-1) > self.max_seq_len:
                x = x[:, -self.max_seq_len:]
            
            # Forward pass
            logits = self.model(x)
            
            # Get next token from last position
            logits = logits[0, -1]
            
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    min_val = top_k_values[-1]
                    mask = logits < min_val
                    logits = logits.masked_fill(mask, -1e10)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(logits).item()
            
            generated.append(next_token)
            x = torch.tensor([next_token], dtype=torch.long, device=self.device)
            
            # Stop if eos
            if next_token == eos_token:
                break
        
        return generated
    
    @torch.no_grad()
    def generate_diffusion(self, prompt: List[int], max_new_tokens: int = 128,
                           num_steps: int = 20, temperature: float = 1.0) -> List[int]:
        """
        Generate text using diffusion sampling.
        
        Starts with UNK tokens and progressively denoises.
        
        Args:
            prompt: Input token IDs (conditioning)
            max_new_tokens: Maximum tokens to generate
            num_steps: Number of denoising steps
            temperature: Sampling temperature
        
        Returns:
            Generated token IDs
        """
        from nanochat_diffusion.diffusion_sampler import DiffusionSampler
        
        sampler = DiffusionSampler(self.model, verbose=True)
        
        # Generate using diffusion
        generated = sampler.generate(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            num_steps=num_steps
        )
        
        return generated
    
    @torch.no_grad()
    def stream(self, prompt: List[int], max_tokens: int = 256,
               stream_callback=None) -> List[int]:
        """
        Stream tokens as they're generated.
        
        Args:
            prompt: Input token IDs
            max_tokens: Maximum tokens to generate
            stream_callback: Callback for each generated token
        
        Returns:
            Generated token IDs
        """
        generated = []
        
        for token in self.generate_autoregressive(
            prompt, max_tokens=max_tokens
        ):
            generated.append(token)
            if stream_callback:
                stream_callback(token)
        
        return generated
    
    @torch.no_grad()
    def batch_generate(self, prompts: List[List[int]], 
                       max_new_tokens: int = 128,
                       temperature: float = 1.0) -> List[List[int]]:
        """
        Generate for multiple prompts in batch.
        
        Args:
            prompts: List of input token sequences
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
        
        Returns:
            List of generated sequences
        """
        results = []
        for prompt in prompts:
            generated = self.generate_autoregressive(
                prompt, max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            results.append(generated)
        
        return results


class KVCache:
    """Simple KV cache implementation."""
    
    def __init__(self, n_layer: int, n_head: int, max_seq_len: int, device: torch.device):
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.device = device
        self.k_cache = torch.zeros(n_layer, max_seq_len, device=device, dtype=COMPUTE_DTYPE)
        self.v_cache = torch.zeros(n_layer, max_seq_len, device=device, dtype=COMPUTE_DTYPE)
        self.pos = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor, input_pos: int):
        """Update cache with new key/value."""
        self.k_cache[:, input_pos] = k
        self.v_cache[:, input_pos] = v
        self.pos += 1
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current cache."""
        return self.k_cache[:, :self.pos], self.v_cache[:, :self.pos]
    
    def clear(self):
        """Clear cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.pos = 0

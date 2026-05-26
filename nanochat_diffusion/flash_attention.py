"""
Flash Attention wrapper for diffusion LLM.
Adapted from karpathy/nanochat.
"""

import torch
import torch.nn.functional as F

# Try to import Flash Attention 3
try:
    from flash_attn import flash_attn_func as _fa3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False
    _fa3_func = None

def flash_attn_func(q, k, v, causal=True, window_size=(-1, 0)):
    """
    Flash Attention forward pass.
    
    Args:
        q: (B, T, H, D)
        k: (B, T, H_kv, D)
        v: (B, T, H_kv, D)
        causal: whether to use causal masking
        window_size: sliding window (left, right)
    
    Returns:
        y: (B, T, H, D)
    """
    if HAS_FA3 and _fa3_func is not None:
        return _fa3_func(q, k, v, causal=causal, window_size=window_size)
    else:
        # Fallback to PyTorch SDPA
        B, T, H, D = q.shape
        H_kv = k.shape[2]
        
        # Squeeze batch dim for SDPA (expects 2D or 3D)
        q_s = q.transpose(1, 2).reshape(B * H, T, D)  # (B*H, T, D)
        k_s = k.transpose(1, 2).reshape(B * H_kv, T, D)  # (B*H_kv, T, D)
        v_s = v.transpose(1, 2).reshape(B * H_kv, T, D)  # (B*H_kv, T, D)
        
        # Use grouped attention: repeat k,v for each query head
        if H != H_kv:
            repeat_factor = H // H_kv
            k_s = k_s.repeat_interleave(repeat_factor, dim=0)
            v_s = v_s.repeat_interleave(repeat_factor, dim=0)
        
        # Scaled dot-product attention
        scale = 1.0 / (D ** 0.5)
        attn = F.scaled_dot_product_attention(
            q_s, k_s, v_s, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=causal
        )
        
        # Reshape back
        y = attn.reshape(B, H, T, D).transpose(1, 2)  # (B, T, H, D)
        return y

def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, 
                             causal=True, window_size=(-1, 0)):
    """
    Flash Attention with KV cache.
    
    Args:
        q: (B, T, H, D) - current queries
        k_cache: pre-allocated key cache
        v_cache: pre-allocated value cache
        k: (B, new_T, H_kv, D) - new keys to append
        v: (B, new_T, H_kv, D) - new values to append
        cache_seqlens: current sequence lengths per batch element
        causal: whether to use causal masking
    
    Returns:
        y: (B, T, H, D)
    """
    # For simplicity, just update cache and run attention
    if k is not None and v is not None:
        # Update cache with new keys/values
        # This is a simplified implementation
        pass
    
    # Run flash attention on cached keys/values
    return flash_attn_func(q, k_cache, v_cache, causal=causal, window_size=window_size)


# Make the module callable so gpt.py's `flash_attn.flash_attn_func(...)` works
class _FlashAttnModule:
    flash_attn_func = flash_attn_func
    flash_attn_with_kvcache = flash_attn_with_kvcache


flash_attn = _FlashAttnModule()

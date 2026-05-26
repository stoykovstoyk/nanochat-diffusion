"""
Diffusion LLM - A diffusion-based language model framework.

Adapted from nanochat by Andrej Karpathy.
This framework implements diffusion-based token generation where:
- Training: randomly mask tokens at various noise levels, predict original tokens
- Inference: iterative denoising from all-masked to clean sequence
"""

__version__ = "0.1.0"

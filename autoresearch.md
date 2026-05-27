# Autoresearch: Optimize Diffusion Training Speed on DGX Spark (GB10)

## Objective
Optimize the diffusion LLM training speed on the DGX Spark (NVIDIA GB10 Blackwell GPU).
Benchmark: depth=8, n_embd=512, batch=8, seq_len=256, 100 iterations.

## Metrics
- **Primary**: training_time_s (seconds, lower is better)
- **Secondary**: final_loss (should stay below ~4.0 for meaningful training)

## How to Run
```
./autoresearch.sh
```
Outputs `METRIC training_time_s=<seconds>`.

## Files in Scope
- `scripts/diffusion_train.py` — Training loop, data loading, autocast, GradScaler
- `nanochat_diffusion/diffusion_model.py` — Diffusion model forward pass
- `nanochat_diffusion/gpt.py` — Base GPT model (Linear, attention, etc.)
- `nanochat_diffusion/common.py` — Dtype detection, compute init, get_peak_flops
- `nanochat_diffusion/dataloader.py` — Data loading pipeline (parquet reader)
- `nanochat_diffusion/checkpoint_manager.py` — Checkpoint save/load
- `pyproject.toml` — Dependency management, torch version

## Off Limits
- `nanochat_diffusion/tokenizer.py`, `flash_attention.py`, `engine.py`, `optim.py`
- `data/` files and sample generation scripts

## Constraints
- Training must converge (final loss < 4.0)
- No new external dependencies
- Must work with `uv sync --extra gpu` (cu128 torch)

## What's Been Tried (this session)

### DGX Spark GB10 Optimizations (baseline: 17.785s)
- **Remove GradScaler/fp16 autocast** → 12.051s (-32.2%). Biggest win. fp16 autocast triggers slow cc12.1 fallback paths. fp32 is native and faster.
- **Remove epoch loop, gradient clipping, zero_grad(set_to_none)** → 17.013s (-4.3%)
- **torch.where in mask_tokens** → 11.781s (-1.5%)
- **Cache sinusoidal frequency buffer** → 11.965s (-0.7%)
- **torch.compile**: crashed — Triton doesn't support sm_121a (Blackwell cc12.1)
- **bf16 autocast**: 16.138s — slower than fp32 on cc12.1 fallback
- **Data pre-generation**: no gain — dataloader is not bottleneck
- **Limited dataloader threads**: no gain
- **Increased batch size**: 3x slower (more tokens)
- **Total improvement: 33.8%** (17.785s → 11.781s)

## Key Insights
- GB10 (Blackwell cc12.1) + PyTorch 2.9.1+cu128 (max cc12.0) = slow fallback GPU kernels
- Basic 512-dim 8-layer transformer takes 86ms/step — fundamental limit
- fp32 >> bf16 > fp16 on this fallback path
- To go faster, need PyTorch with native Blackwell support

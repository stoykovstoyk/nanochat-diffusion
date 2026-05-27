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
(None yet — fresh start on DGX Spark GB10)

## Key Insights (this session)
(None yet)

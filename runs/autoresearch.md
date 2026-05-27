# Autoresearch: Optimize Diffusion Training Speed

## Objective
Optimize the diffusion LLM training throughput (seconds per epoch, lower is better).
The benchmark runs a small model (depth=8, n_embd=512) on CPU for 100 training iterations
with synthetic data, measuring total wall time. Each improvement should maintain
correct training behavior (loss decreases, no crashes).

## Metrics
- **Primary**: training_time_s (seconds, lower is better)
- **Secondary**: avg_loss (final average loss, should stay below 8.0 for non-trivial training)

## How to Run
```
./autoresearch.sh
```
Outputs `METRIC training_time_s=<seconds>`.

## Files in Scope
- `scripts/diffusion_train.py` — Main training script. Has synthetic dataset, training loop.
- `nanochat_diffusion/checkpoint_manager.py` — Checkpoint save/load. Uses `time.time()` now.
- `nanochat_diffusion/diffusion_model.py` — Diffusion model with timestep conditioning.
- `nanochat_diffusion/diffusion_scheduler.py` — Noise schedule and masking.
- `nanochat_diffusion/common.py` — Common utilities, dtype detection, logging.
- `nanochat_diffusion/gpt.py` — Base GPT model, attention, MLP, rotary embeddings.
- `nanochat_diffusion/optim.py` — AdamW and Muon optimizers with fused kernels.
- `nanochat_diffusion/engine.py` — Inference engine (not used in training).
- `nanochat_diffusion/dataloader.py` — Data loading pipeline.

## Off Limits
- `nanochat_diffusion/tokenizer.py` — Tokenizer code. Don't modify.
- `nanochat_diffusion/flash_attention.py` — Flash attention wrapper. Only relevant for CUDA.
- Data files and sample generation scripts.

## Constraints
- Training must produce reasonable loss values (< 8.0 on avg)
- No new external dependencies
- Must run on CPU (no CUDA required)
- Changes should not break eval/inference functionality

## What's Been Tried
- Baseline: vanilla run with depth=8, n_embd=512, CPU, 100 iterations

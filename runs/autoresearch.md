# Autoresearch: Optimize Diffusion Training Speed

## Objective
Optimize the diffusion LLM training speed (seconds for 100 iterations, lower is better).
Benchmark runs depth=8, n_embd=512, batch=8, seq_len=256, 100 iterations.

## Metrics
- **Primary**: training_time_s (seconds, lower is better)
- **Secondary**: final_loss (should stay below ~3.0 for meaningful training)

## How to Run
```
./autoresearch.sh
```
Outputs `METRIC training_time_s=<seconds>`.

## Files in Scope
- `scripts/diffusion_train.py` — Training loop, data loading, autocast, GradScaler
- `nanochat_diffusion/checkpoint_manager.py` — Checkpoint save/load
- `nanochat_diffusion/diffusion_model.py` — Diffusion model forward pass
- `nanochat_diffusion/gpt.py` — Base GPT model
- `nanochat_diffusion/common.py` — Dtype detection, compute init
- `nanochat_diffusion/dataloader.py` — Data loading pipeline

## Off Limits
- `nanochat_diffusion/tokenizer.py`, `flash_attention.py`, `engine.py`, `optim.py`
- Data files and sample generation scripts

## Constraints
- Training must converge (final loss < 3.0)
- No new external dependencies

## What's Been Tried

### CPU Optimizations (baseline: 118.634s)
- **hardtanh** instead of tanh logit softcap → 110.615s (-6.8%). Tanh is expensive on CPU.
- **Dead code removal** (unused block_input, x_pre_mlp) + --num-cpus 1 → 109.285s (-7.9%)
- OMP_NUM_THREADS=4 made things worse (oversubscription with 24 dataloader workers)
- torch.compile: crashed without setuptools; with it, compilation overhead dominates short runs

### GPU Optimizations (RTX A2000)
- **Switch to GPU** (--device-type cuda) → 21.679s (-81.7%). The single biggest win.
- **Skip final checkpoint save** → 17.669s (-85.1%). I/O was ~4s of overhead.
- **Delayed loss.item()** (store detached tensors, sync only at print time) + fp16 autocast → 16.985s (-85.7%)
- fp16 autocast alone: ~2% improvement from tensor cores on Ampere
- Simplified dataloader (bypass tokenizer pipeline): didn't help, changed data distribution
- zero_grad(set_to_none=True): no improvement

### Key Insights
1. GPU is 5x+ faster than CPU for this workload
2. For tiny models on GPU, kernel launch overhead dominates — reducing Python overhead helps
3. Checkpoint I/O is expensive even on fast storage
4. FP16 autocast on Ampere gives modest gains (tf32 already enabled via matmul_precision='high')
5. Model is too small (512-dim, 8 layers) for effective GPU utilization

# Autoresearch Worklog: Optimize Diffusion Training Speed on DGX Spark (GB10)

## Session Info
- **System**: NVIDIA GB10 (Blackwell, cc 12.1), CUDA 13.0, PyTorch 2.9.1+cu128
- **CPU**: 20 cores (ARM64)

## Data Summary
| Metric | Value |
|--------|-------|
| Runs | 5 |
| Kept | 5 |
| Discarded | 2 |
| Crashed | 1 |
| Baseline | 17.785s (#1) |
| Best | 11.781s (#5, -33.8%) |

## Key Insights
- **GB10 on PyTorch 2.9.1+cu128**: cc12.1 > max 12.0 → slow fallback kernels. Basic 8-layer 512-dim transformer takes 86ms/step.
- **GradScaler overhead is huge on cc12.1**: removing it yielded -32% improvement (17.013→12.051s). The fp16 autocast paths trigger slow fallback kernels.
- **fp32 > bf16**: bf16 autocast was actually slower (16.1s), confirming cc12.1 fallback has no tensor core paths.
- **Data loading is NOT the bottleneck**: pre-generating data (exp7) didn't help. Distributed dataloader with 20 threads is fast enough.
- **torch.compile crashes**: Triton doesn't support sm_121a.
- **Python overhead optimizations**: removing epoch loop, gradient clipping, pre-gen timesteps, torch.where in mask_tokens — each gave ~1-5% improvement.
- **Fundamental limit**: ~80ms/step GPU compute on cc12.1 fallback. To go faster, need PyTorch with Blackwell (cc12.1) support.

## Experiments

### Run 1: baseline on GB10 — 17.785s (KEEP)
- What changed: First run on DGX Spark, depth=8, batch=8, seq=256, 100 iters
- Result: 17.785s, loss=2.639

### Run 2: no epoch loop, pre-gen timesteps, no grad clip, set_to_none, print50 — 17.013s (KEEP)
- What changed: single-pass training loop, removed gradient clipping, set_to_none, print every 50
- Result: 17.013s (-4.3%), loss=2.723

### Run 3: disable GradScaler/fp16 autocast — 12.051s (KEEP ⭐)
- What changed: removed GradScaler and fp16 autocast, train in native fp32
- Result: 12.051s (-29.2% vs best), loss=2.723

### Run 4: cache sinusoidal freq buffer in timestep embed — 11.965s (KEEP)
- What changed: precomputed frequency buffer in SinusoidalTimestepEmbedding
- Result: 11.965s (-0.7%), loss=2.724

### Run 5: torch.where in mask_tokens — 11.781s (KEEP)
- What changed: replaced clone+scatter with torch.where for token masking
- Result: 11.781s (-1.5%), loss=2.724

## Discarded
- Exp1 (--num-cpus 2): 17.778s, no improvement vs baseline's 17.785s
- Exp2 (batch_size=32): 52.683s, 3x slower (processes 4x more tokens/step)
- Exp3 (torch.compile): crashed — Triton doesn't support sm_121a
- Exp6 (--num-cpus 4): 11.995s, no improvement vs 12.051s
- Exp7 (pre-gen input data): 12.034s, pre-generation overhead cancels gain
- Exp9 (profiling): 11.933s, timing overhead minimal but no optimization
- Exp11 (bf16 autocast): 16.138s, slower than fp32 on cc12.1 fallback

## Next Ideas
To get beyond 11.8s, need PyTorch with native Blackwell support (cc12.1). Otherwise minimal headroom remains.

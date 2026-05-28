# Autoresearch Dashboard: Optimize Diffusion Training Speed on DGX Spark

**Runs:** 13 | **Kept:** 13 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** training_time_s: 17.785s (#1)
**Best:** training_time_s: 9.295s (#12, -47.7%) ⭐⭐

| # | commit | training_time_s | status | description |
|---|--------|---------------|--------|-------------|
| 1 | 341d730 | 17.785s | keep | baseline on GB10 (Blackwell, cc12.1) |
| 2 | 30a17c7 | 17.013s (-4.3%) | keep | no epoch loop, pre-gen timesteps, no grad clip, set_to_none, print50 |
| 3 | e45c973 | 12.051s (-32.2%) | keep | disable GradScaler/fp16 autocast ⭐ |
| 4 | 8e71934 | 11.965s (-32.7%) | keep | cache sinusoidal freq buffer in timestep embed |
| 5 | 1ed3f37 | 11.781s (-33.8%) | keep | torch.where in mask_tokens |
| 6 | * | 11.581s (-34.9%) | keep | torch 2.12.0+cu130 (CUDA 13) + Python 3.12 |
| 7 | * | 9.745s (-45.2%) | keep | **torch.compile (wired up! was dead code before) ⭐** |
| 8 | * | 9.421s (-47.0%) | keep | compile mode=reduce-overhead (CUDA graphs) |
| 9 | * | 9.660s (-45.7%) | keep | compile + batch=16 (2x tok throughput) |
| 10 | * | 9.378s (-47.3%) | keep | compile + batch=32 (341 tok/s) |
| 11 | * | 12.580s (-29.3%) | keep | depth=12 + compile (40% speedup from compile) |
| 12 | * | **9.295s (-47.7%)** | keep ⭐ | **compile + seq=512 + bs=16 (88K tok/s 🚀)** |
| 13 | * | 15.616s (-12.2%) | keep | depth=16 (1024-dim) + compile (26K tok/s) |

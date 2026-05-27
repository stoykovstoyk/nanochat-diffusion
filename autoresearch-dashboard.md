# Autoresearch Dashboard: Optimize Diffusion Training Speed on DGX Spark

**Runs:** 5 | **Kept:** 5 | **Discarded:** 2 | **Crashed:** 1
**Baseline:** training_time_s: 17.785s (#1)
**Best:** training_time_s: 11.781s (#5, -33.8%)

| # | commit | training_time_s | status | description |
|---|--------|---------------|--------|-------------|
| 1 | 341d730 | 17.785s | keep | baseline on GB10 (Blackwell, cc12.1) |
| 2 | 30a17c7 | 17.013s (-4.3%) | keep | no epoch loop, pre-gen timesteps, no grad clip, set_to_none, print50 |
| 3 | e45c973 | 12.051s (-32.2%) | keep | disable GradScaler/fp16 autocast ⭐ |
| 4 | 8e71934 | 11.965s (-32.7%) | keep | cache sinusoidal freq buffer in timestep embed |
| 5 | 1ed3f37 | 11.781s (-33.8%) | keep | torch.where in mask_tokens |

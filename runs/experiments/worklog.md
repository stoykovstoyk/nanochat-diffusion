# Autoresearch Worklog: Optimize Diffusion Training Speed

## Session Summary
- **Started:** 2026-05-27
- **Primary Metric:** training_time_s (seconds, lower is better)
- **Model:** depth=8, n_embd=512, seq_len=256, batch=8, GPU (RTX A2000), 100 iterations

## Key Insights
- tanh logit softcap is expensive on CPU (~7% of training time); hardtanh is equivalent and free
- OMP_NUM_THREADS=4 oversubscribes with 24 dataloader workers (150s vs 119s)
- GPU training is ~5x faster than CPU for this workload (109s → 22s)
- The model is tiny (depth=8, n_embd=512), so GPU kernel launch overhead is significant

## Next Ideas
- Increase batch size to improve GPU utilization
- Try torch.compile on GPU (setuptools required)
- Increase model size to reduce kernel launch overhead relative to compute

---

### Run 1: baseline — training_time_s=118.634 (KEEP)
- Timestamp: 2026-05-27 11:38
- What changed: Initial baseline run for training speed
- Result: training_time_s=118.634, final_loss=0.965

### Run 2: hardtanh instead of tanh — training_time_s=110.615 (KEEP)
- Timestamp: 2026-05-27 11:49
- What changed: Replaced `15.0 * torch.tanh(logits / 15.0)` with `F.hardtanh(logits, -15, 15)`
- Result: training_time_s=110.615 (-6.8%), final_loss=0.885
- Insight: tanh is expensive on CPU; hardtanh gives same logit bounding for free

### Run 3: dead code removal + --num-cpus 1 — training_time_s=109.285 (KEEP)
- Timestamp: 2026-05-27 11:57
- What changed: Removed unused block_input/x_pre_mlp in diffusion forward; set --num-cpus 1
- Result: training_time_s=109.285 (-7.9% vs baseline), final_loss=0.820
- Insight: Reduced dataloader thread contention helps slightly on CPU

### Run 4: GPU training (RTX A2000) — training_time_s=21.679 (KEEP)
- Timestamp: 2026-05-27 12:15
- What changed: Switched from --device-type cpu to cuda
- Result: training_time_s=21.679 (-81.7% vs baseline!), final_loss=1.044
- Insight: GPU gives 5x speedup. Next: increase batch size for better GPU utilization

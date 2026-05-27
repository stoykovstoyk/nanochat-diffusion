# Autoresearch Worklog: Optimize Diffusion Training Speed

## Session Summary
- **Started:** 2026-05-27
- **Primary Metric:** training_time_s (seconds, lower is better)
- **Model:** depth=8, n_embd=512, seq_len=256, batch=8, CPU, 100 iterations

## Key Insights
- (to be filled as experiments progress)

## Next Ideas
- (to be filled as experiments progress)

---

### Run 1: baseline — training_time_s=118.634 (KEEP)
- Timestamp: 2026-05-27 11:38
- What changed: Initial baseline run with depth=8, n_embd=512, CPU, 100 iters
- Result: training_time_s=118.634, final_loss=0.965
- Insight: Model converges well on synthetic data. Most time likely in forward/backward of transformer.
- Next: Try torch.compile for CPU inference, or reduce OMP_NUM_THREADS / tune parallelism

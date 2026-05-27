# Autoresearch Worklog: Optimize Diffusion Training Speed on DGX Spark (GB10)

## Session Info
- **System**: NVIDIA GB10 (Blackwell, cc 12.1), CUDA 13.0, PyTorch 2.9.1+cu128
- **GPU memory**: Not supported for querying from nvidia-smi (GB10 unified memory?)
- **CPU**: 20 cores (ARM64)

## Data Summary
| Metric | Value |
|--------|-------|
| Runs | 0 |
| Kept | 0 |
| Discarded | 0 |
| Crashed | 0 |
| Baseline | TBD |
| Best | TBD |

## Key Insights
- GB10 baseline: 17.785s — roughly on par with RTX A2000 best (16.985s), despite cc12.1 warning
- Loss converges well: 2.639 final

## Next Ideas
- Check if torch.compile works (might hit cc12.1 issues)
- Try disabling wandb import entirely
- Try reducing dataloader parallelism (oversubscription with 20 threads on 20-core system)
- Try bf16 if supported (cc12.1 should support it)

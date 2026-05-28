# Autoresearch Worklog: Improve Training Loss for Diffusion LLM

## Session Info
- **Goal**: Achieve loss < 4.0 at 2000 training iterations
- **Model**: depth=8, n_embd=512, vocab_size=4096, BPE tokenizer
- **Hardware**: DGX Spark (GB10, CUDA 13.0)

## Data Summary
| Metric | Value |
|--------|-------|
| Runs | 3 |
| Kept | 1 |
| Best | 4.87 (#3, -8.5% vs baseline) |
| Baseline | 5.32 (#1) |

## Experiments

### Run 1: Baseline — constant LR 4e-4, linear noise, max_mask=0.8 — 5.32 (KEEP)
- What changed: First experiment with fixed tokenizer and all-positions loss
- Config: lr=4e-4, noise=linear, max_mask=0.8, no warmup, no scheduler
- Result: loss 5.32 @ 2000 iters (7.0→5.32), loss 5.79 @ 500

### Run 2: Cosine LR scheduler — 5.44 (DISCARD)
- What changed: Added cosine LR decay with 100-step warmup
- Config: lr=4e-4, noise=linear, max_mask=0.8, warmup=100, cosine decay
- Result: loss 5.44 @ 2000 iters (worse than constant LR)

### Run 3: Cosine noise, lower mask ratio, higher LR — 4.87 (KEEP ⭐)
- What changed: noise=cosine, max_mask=0.5, lr=8e-4, warmup=100
- Result: loss 4.87 @ 2000 iters, loss 5.36 @ 500
- Insight: Easier noise schedule + higher LR gives faster, better convergence

## Next Ideas
- Try gradient clipping to enable even higher LR
- Sweep max_mask_ratio (0.4, 0.5, 0.6)
- Try constant noise schedule with mask_ratio=0.5
- GPT-2 style Adam (beta1=0.9, beta2=0.98)

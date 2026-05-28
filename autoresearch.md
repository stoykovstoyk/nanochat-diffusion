# Autoresearch: Improve Training Loss for Diffusion LLM

## Objective
The diffusion LLM loss plateaus too high (~5.3 after 2000 steps for vocab=4096).
Find training config changes that achieve lower loss faster (target <4.0 at 2000 steps).

## Metrics
- **Primary**: loss at step 2000 (lower is better)
- **Secondary**: loss at step 500 (early convergence speed)

## How to Run (Best Config)
```bash
python -m scripts.diffusion_train \
    --depth 8 --max-seq-len 512 --device-batch-size 16 \
    --num-iterations 2000 --lr 1e-3 --warmup-iters 100 --grad-clip 1.0 \
    --vocab-size 4096 --unk-token-id 4095 --max-mask-ratio 0.15 \
    --noise-schedule cosine
```

## Files in Scope
- `scripts/diffusion_train.py` — Training loop, LR scheduler, optimizer config
- `nanochat_diffusion/diffusion_model.py` — Diffusion model forward, loss computation
- `nanochat_diffusion/diffusion_scheduler.py` — Noise schedules
- `nanochat_diffusion/gpt.py` — Transformer backbone
- `nanochat_diffusion/tokenizer.py` — BPE tokenizer

## Constraints
- Max 2000 iterations per test
- Must use BPE tokenizer (vocab_size=4096, UNK=4095)
- Must converge (loss trending down)
- Training on single GPU only

## What's Been Tried

### Best So Far: loss 4.18 @ 2000 iters (max_mask=0.15)
- cosine noise schedule, max_mask_ratio=0.15, lr=1e-3, warmup=100, grad_clip=1.0
- 4.18 at step 2000, 4.90 at step 500

### Mask ratio sweep results (cosine noise, LR 1e-3, grad_clip 1.0, warmup 100)
| max_mask | Loss @ 2000 |
|----------|------------|
| 0.8 | 5.32 |
| 0.5 | 4.87 |
| 0.4 | 4.55 |
| 0.3 | 4.41 |
| 0.25 | 4.36 |
| 0.2 | 4.25 |
| 0.15 | 4.18 ⭐ |

### Discarded
- mask-only loss: loss stuck at ~7.0 (no improvement)
- cosine LR decay: worse than constant LR (5.44 vs 5.32 with cosine, 5.30 with const)
- LR 1e-3 without gradient clipping: unstable (loss spikes)

## Key Insights
1. Lower max_mask_ratio = faster learning (model has more context)
2. Cosine noise schedule helps vs linear (more gradual difficulty)
3. Gradient clipping enables higher LR (1e-3) without divergence
4. All-positions loss works better than mask-only for this architecture
5. Constant LR or mild cosine decay works best
6. BPE tokenizer needs ByteLevel decoder for lossless roundtrip

## Next Ideas
- Try max_mask=0.1 to see if trend continues
- Try GPT-2 style Adam (beta1=0.9, beta2=0.98)
- Try larger model (depth=12, n_embd=768) for better capacity
- Try gradient accumulation for more stable gradients
- Train longer (5000+ steps with cosine decay)

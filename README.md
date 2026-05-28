# nanochat-diffusion

A diffusion-based language model (dLLM) framework. Instead of autoregressive next-token prediction,
the model masks tokens at various noise levels and learns to denoise them in parallel.

Built on the GPT architecture from [karpathy/nanochat](https://github.com/karpathy/nanochat).

---

## How It Works

**Training:**
1. Take a sequence of tokens, randomly mask a fraction with `UNK`
2. Feed the partially-masked sequence through the transformer
3. Predict the original tokens — loss computed on all positions
4. Noise level `t` (0–1000) controls the mask ratio via a noise schedule

**Inference (progressive denoising):**
1. Start with all tokens as `UNK`
2. Each step: forward pass predicts all positions simultaneously
3. Replace `UNK` tokens with the highest-confidence predictions
4. Repeat 10–20 steps until all tokens are determined

All positions are predicted in parallel per step — generation takes O(steps) instead of O(sequence_length).

---

## Quick Start

### Install

```bash
git clone https://github.com/stoykovstoyk/nanochat-diffusion.git
cd nanochat-diffusion

# CUDA GPU
uv sync --extra gpu

# CPU only
uv sync --extra cpu

source .venv/bin/activate
```

### Download Data & Train Tokenizer

Downloads ~50k FineWeb articles (80 MB) and trains a BPE tokenizer:

```bash
python -m scripts.download_dataset --num-examples 50000
```

### Train a Model

Best config (loss 4.18 after 2000 steps on DGX Spark):

```bash
python -m scripts.diffusion_train \
    --depth 8 --max-seq-len 512 --device-batch-size 16 \
    --num-iterations 2000 --lr 1e-3 --warmup-iters 100 --grad-clip 1.0 \
    --vocab-size 4096 --unk-token-id 4095 --max-mask-ratio 0.15 \
    --noise-schedule cosine
```

For Blackwell GPUs (GB10/DGX Spark, sm_121) with `torch.compile`:

```bash
TRITON_CUDA_ARCH=sm_120 python -m scripts.diffusion_train \
    --depth 8 --max-seq-len 512 --device-batch-size 16 \
    --num-iterations 2000 --lr 1e-3 --warmup-iters 100 --grad-clip 1.0 \
    --vocab-size 4096 --unk-token-id 4095 --max-mask-ratio 0.15 \
    --noise-schedule cosine --compile --custom-rmsnorm
```

### Generate Text

```bash
python -m scripts.diffusion_infer \
    --prompt "The future of AI is" --max-tokens 256 --num-steps 20
```

---

## Training

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth` | `8` | Transformer layers (`n_embd = depth × aspect_ratio`) |
| `--aspect-ratio` | `64` | Width multiplier |
| `--max-seq-len` | `1024` | Maximum sequence length |
| `--device-batch-size` | `16` | Per-device batch size |
| `--num-iterations` | `-1` | Training iterations (`-1` = auto from target flops) |
| `--lr` | `4e-4` | Learning rate |
| `--warmup-iters` | `50` | LR linear warmup steps |
| `--grad-clip` | `0.0` | Gradient norm clipping (0 = disabled) |
| `--weight-decay` | `0.1` | AdamW weight decay |
| `--beta1` | `0.8` | AdamW beta1 |
| `--beta2` | `0.95` | AdamW beta2 |
| `--compile` | `False` | Enable `torch.compile` (CUDA only) |
| `--compile-mode` | `reduce-overhead` | `default`, `reduce-overhead`, `max-autotune` |
| `--fullgraph` | `False` | Full CUDA graph capture |
| `--custom-rmsnorm` | `False` | Use custom Triton RMS norm (2.8x faster) |
| `--attention-backend` | `auto` | SDPA backend: `auto`, `math`, `flash`, `mem_efficient`, `cudnn` |
| `--num-diffusion-steps` | `1000` | Training timesteps |
| `--sampling-steps` | `20` | Inference denoising steps |
| `--max-mask-ratio` | `0.8` | Max fraction of tokens masked (0.15 recommended) |
| `--noise-schedule` | `linear` | `linear`, `cosine`, `exponential`, `constant` |
| `--vocab-size` | `4096` | BPE vocabulary size (4096 = BOS + 4094 BPE + UNK) |
| `--unk-token-id` | `4095` | UNK sentinel (outside BPE vocab, within padded range) |
| `--save-every` | `1000` | Save checkpoint every N steps |
| `--resume` | `""` | Resume from step number (`latest` for most recent) |
| `--device-type` | `auto` | Device: `cuda`, `cpu`, `mps` |
| `--num-cpus` | `all` | CPU threads for tokenization |
| `--run` | `diffusion_demo` | W&B run name |

### Learning Tips

- **Lower `--max-mask-ratio`** (0.15–0.3) gives faster convergence — the model has more visible context
- **`--noise-schedule cosine`** is better than `linear` — more gradual difficulty increase
- **`--grad-clip 1.0`** prevents loss spikes and enables higher LR
- **`--lr 1e-3` + `--warmup-iters 100`** works well with gradient clipping
- **BPE tokenizer** is trained automatically from the first `--vocab-size` parquet file in `data/`

### Examples

**Quick test (100 iters, depth=4):**
```bash
python -m scripts.diffusion_train \
    --depth 4 --max-seq-len 256 --device-batch-size 4 \
    --num-iterations 100 --max-mask-ratio 0.15 --noise-schedule cosine
```

**Medium training (5000 iters):**
```bash
python -m scripts.diffusion_train \
    --depth 8 --max-seq-len 512 --device-batch-size 16 \
    --num-iterations 5000 --lr 1e-3 --warmup-iters 100 --grad-clip 1.0 \
    --vocab-size 4096 --max-mask-ratio 0.15 --noise-schedule cosine
```

**Recommended production config (depth=12):**
```bash
python -m scripts.diffusion_train \
    --depth 12 --max-seq-len 1024 --device-batch-size 8 \
    --num-iterations 20000 --lr 8e-4 --warmup-iters 200 --grad-clip 1.0 \
    --vocab-size 4096 --max-mask-ratio 0.2 --noise-schedule cosine \
    --compile --save-every 1000
```

**Resume from checkpoint:**
```bash
python -m scripts.diffusion_train --resume latest
python -m scripts.diffusion_train --resume 5000
```

---

## Inference

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | `""` | Text prompt to continue |
| `--max-tokens` | `128` | Max tokens to generate (beyond prompt) |
| `--num-steps` | `20` | Denoising steps (more = better quality) |
| `--temperature` | `0.8` | Sampling temperature |
| `--top-k` | `40` | Top-k filtering (0 = disabled) |
| `--checkpoint-step` | `latest` | Checkpoint to load |
| `--output-file` | `""` | Save generated text to file |

```bash
python -m scripts.diffusion_infer \
    --prompt "Hello world" --max-tokens 64 --num-steps 15

python -m scripts.diffusion_infer \
    --prompt "Once upon a time" --max-tokens 512 \
    --num-steps 30 --temperature 1.0 --top-k 50

python -m scripts.diffusion_infer \
    --prompt "Write a poem" --max-tokens 256 --output-file poem.txt
```

---

## BPE Tokenizer

The tokenizer uses HuggingFace `tokenizers` library with ByteLevel BPE:

- **Vocab size**: 4096 (BOS=0, 4094 BPE tokens, UNK=4095)
- **Training**: Auto-trained from parquet files in `data/` on first run.
  Or explicitly via `python -m scripts.download_dataset`
- **Saved to**: `<project>/data/tokenizer_diffusion/tokenizer.json`
- **Roundtrip**: Lossless — `decode(encode(text)) == text`

---

## Checkpoints & Storage

All data is project-local:

```
nanochat-diffusion/
├── data/
│   ├── checkpoints/diffusion/train/
│   │   └── step_NNNNNNNNNN/
│   │       ├── model.pt        # Model weights
│   │       ├── optimizer.pt    # Optimizer state
│   │       ├── config.json     # Architecture config
│   │       └── metadata.json   # Training metadata
│   ├── tokenizer_diffusion/
│   │   └── tokenizer.json      # BPE tokenizer
│   └── train_*.parquet          # Training data (FineWeb)
├── cuda_kernels/                # Custom kernels
├── nanochat_diffusion/          # Core library
└── scripts/                     # Entry points
```

Override with `NANOCHAT_BASE_DIR`:
```bash
export NANOCHAT_BASE_DIR=/path/to/storage
```

Checkpoints are gitignored. Use `--save-every` to control save frequency.

---

## Noise Schedules

| Schedule | Behavior |
|----------|----------|
| `cosine` | Slow start/finish, steep middle — recommended |
| `linear` | Mask ratio increases linearly with t |
| `exponential` | Fast initial noise, gradual cleanup |
| `constant` | Fixed mask ratio throughout |

---

## Performance Tuning (DGX Spark / GB10)

```bash
TRITON_CUDA_ARCH=sm_120 python -m scripts.diffusion_train \
    --depth 8 --max-seq-len 512 --device-batch-size 16 \
    --num-iterations 2000 --lr 1e-3 --warmup-iters 100 --grad-clip 1.0 \
    --vocab-size 4096 --max-mask-ratio 0.15 --noise-schedule cosine \
    --compile --custom-rmsnorm
```

Speed findings (depth=8, n_embd=512, bs=16, seq=512):
- **216ms/iter** with `--compile --custom-rmsnorm` + `TRITON_CUDA_ARCH=sm_120`
- `torch.compile` gives ~25% speedup over eager
- `--custom-rmsnorm` gives ~7% over `F.rms_norm`

Training quality findings (2000 iters):
- Lower `--max-mask-ratio` = faster convergence (4.18 at 0.15 vs 5.32 at 0.8)
- `--noise-schedule cosine` > `linear`
- `--grad-clip 1.0` enables `--lr 1e-3` without divergence
- BPE tokenization is critical for learning (byte-level doesn't work)

---

## License

MIT (same as [nanochat](https://github.com/karpathy/nanochat))

# nanochat-diffusion

A diffusion-based language model (dLLM) framework. Instead of autoregressive next-token prediction, the model masks tokens at various noise levels and learns to denoise them in parallel, enabling faster generation and flexible conditioning.

Built on the GPT architecture from [karpathy/nanochat](https://github.com/karpathy/nanochat).

---

## How It Works

**Training:**
1. Take a sequence of tokens — `[BOS, THE, QUICK, BROWN, FOX, ...]`
2. Sample a noise level `t` (0–1000), mask a fraction of tokens with `UNK`
3. Feed the partially-masked sequence through the transformer
4. Predict the original tokens at masked positions only

**Inference (progressive denoising):**
1. Start with all tokens as `UNK`
2. Each step: forward pass predicts all masked positions in parallel
3. Replace some `UNK` tokens with the highest-confidence predictions
4. Repeat 10–20 steps until all tokens are determined

**Key advantage:** All positions are predicted simultaneously per step, not one-by-one. Generation takes O(steps) instead of O(sequence_length), typically 10–20 steps regardless of sequence length.

---

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended; CPU works but is slow)
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone https://github.com/stoykovstoyk/nanochat-diffusion.git
cd nanochat-diffusion

# CUDA GPU (tested on sm_80+, sm_120/121)
uv sync --extra gpu

# CPU only
uv sync --extra cpu

source .venv/bin/activate
```

### Train a Small Model

```bash
python -m scripts.diffusion_train \
    --depth 8 --max-seq-len 512 --device-batch-size 16 \
    --num-iterations 2000 --lr 4e-4 --compile
```

### Generate Text

```bash
python -m scripts.diffusion_infer \
    --prompt "The quick brown fox" \
    --max-tokens 128 --num-steps 20
```

---

## Installation Details

### GPU Setup

The project uses PyTorch 2.12+ compiled for CUDA 13.0 (`cu130` wheels). Supported GPUs:

| GPU | Compute Capability | Notes |
|-----|-------------------|-------|
| NVIDIA A100/H100/H200 | sm_80/sm_90 | Fully supported |
| NVIDIA RTX 4090/5090 | sm_89/sm_120 | Fully supported |
| NVIDIA GB10 (DGX Spark) | sm_121 | Supported via sm_120 binary compat |
| Older GPUs (RTX 3090, etc.) | sm_86 | Supported |

For Blackwell GPUs (sm_120/sm_121), set `TRITON_CUDA_ARCH=sm_120` for optimal triton codegen:

```bash
TRITON_CUDA_ARCH=sm_120 python -m scripts.diffusion_train --compile ...
```

### CPU Fallback

Omit `--extra gpu` for CPU-only torch. Training will be slow but functional for small models.

---

## Training

### Basic Usage

```bash
python -m scripts.diffusion_train \
    --depth 8                     # Transformer depth (layers)
    --max-seq-len 512             # Sequence length
    --device-batch-size 16        # Batch size per GPU
    --num-iterations 5000         # Training iterations
    --lr 4e-4                     # Learning rate
    --compile                     # Enable torch.compile (CUDA only)
```

### All Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth` | `8` | Number of transformer layers (`n_embd = depth × aspect_ratio`) |
| `--aspect-ratio` | `64` | Width multiplier |
| `--max-seq-len` | `1024` | Maximum sequence length |
| `--device-batch-size` | `16` | Per-device batch size |
| `--num-iterations` | `-1` | Training iterations (use target flops if `-1`) |
| `--lr` | `4e-4` | Learning rate |
| `--weight-decay` | `0.1` | AdamW weight decay |
| `--beta1` | `0.8` | AdamW beta1 |
| `--beta2` | `0.95` | AdamW beta2 |
| `--compile` | `False` | Enable `torch.compile` |
| `--compile-mode` | `reduce-overhead` | Compile mode: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs` |
| `--fullgraph` | `False` | Full CUDA graph capture |
| `--custom-rmsnorm` | `False` | Use custom Triton RMS norm (2.8x faster) |
| `--attention-backend` | `auto` | SDPA backend: `auto`, `math`, `flash`, `mem_efficient`, `cudnn` |
| `--cudnn-benchmark` | `False` | Enable `torch.backends.cudnn.benchmark` |
| `--num-diffusion-steps` | `1000` | Training timesteps |
| `--sampling-steps` | `20` | Inference denoising steps |
| `--max-mask-ratio` | `0.8` | Maximum mask fraction during training |
| `--noise-schedule` | `linear` | Schedule: `linear`, `cosine`, `exponential`, `constant` |
| `--vocab-size` | `32768` | Vocabulary size |
| `--unk-token-id` | `32767` | UNK token ID (must be ≥ vocab_size - 1) |
| `--warmup-iters` | `50` | LR warmup iterations |
| `--save-every` | `1000` | Save checkpoint every N steps |
| `--eval-iters` | `100` | Evaluate every N steps |
| `--resume` | `""` | Resume from checkpoint step number (`latest` for most recent) |
| `--device-type` | `auto` | Device: `cuda`, `cpu`, `mps` |
| `--num-cpus` | `all` | CPU threads for tokenization |
| `--run` | `diffusion_demo` | W&B run name |

### Training Examples

**Quick sanity check (CPU, depth=4):**
```bash
python -m scripts.diffusion_train \
    --depth 4 --max-seq-len 256 --device-batch-size 4 \
    --num-iterations 100
```

**GPU benchmark (DGX Spark/GB10 optimized):**
```bash
TRITON_CUDA_ARCH=sm_120 python -m scripts.diffusion_train \
    --depth 8 --max-seq-len 512 --device-batch-size 16 \
    --num-iterations 120 --compile --custom-rmsnorm
```

**Medium training run:**
```bash
python -m scripts.diffusion_train \
    --depth 12 --max-seq-len 1024 --device-batch-size 16 \
    --num-iterations 10000 --lr 3e-4 --warmup-iters 200 \
    --compile --save-every 1000 --eval-iters 500
```

**Resume from checkpoint:**
```bash
python -m scripts.diffusion_train --resume latest
python -m scripts.diffusion_train --resume 5000
```

---

## Inference

### Basic Generation

```bash
python -m scripts.diffusion_infer \
    --prompt "The future of AI is" \
    --max-tokens 256 --num-steps 20 --temperature 0.8
```

### All Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | `""` | Text prompt to continue |
| `--max-tokens` | `128` | Maximum tokens to generate (beyond prompt) |
| `--num-steps` | `20` | Denoising steps (more = better quality, slower) |
| `--temperature` | `0.8` | Sampling temperature (lower = more deterministic) |
| `--top-k` | `40` | Top-k filtering (0 = disabled) |
| `--checkpoint-step` | `latest` | Checkpoint step to load |
| `--checkpoint-dir` | `""` | Override checkpoint directory |
| `--output-file` | `""` | Save generated text to file |

### Examples

```bash
# Short generation
python -m scripts.diffusion_infer \
    --prompt "Hello world" --max-tokens 64 --num-steps 15

# Longer, more creative
python -m scripts.diffusion_infer \
    --prompt "Once upon a time" --max-tokens 512 \
    --num-steps 30 --temperature 1.0 --top-k 50

# Deterministic (temperature=0)
python -m scripts.diffusion_infer \
    --prompt "2 + 2 =" --max-tokens 16 --num-steps 10 --temperature 0

# Save output to file
python -m scripts.diffusion_infer \
    --prompt "Write a poem" --max-tokens 256 \
    --output-file poem.txt
```

---

## Evaluation

```bash
python -m scripts.diffusion_evaluate \
    --checkpoint-step latest --tasks gsm8k,arc

python -m scripts.diffusion_evaluate \
    --evaluate-perplexity --evaluate-generation
```

---

## Checkpoints & Storage

All data is stored inside the project directory:

```
nanochat-diffusion/
├── data/
│   ├── checkpoints/           # Model checkpoints
│   │   └── diffusion/
│   │       └── train/
│   │           ├── step_0000000500/
│   │           │   ├── model.pt        # Model weights
│   │           │   ├── optimizer.pt    # Optimizer state
│   │           │   ├── config.json     # Architecture config
│   │           │   └── metadata.json   # Training metadata
│   │           └── step_000001000/
│   │               └── ...
│   ├── tokenizer_diffusion/   # Tokenizer data
│   └── train_*.parquet        # Training data
├── cuda_kernels/              # Custom Triton/CUDA kernels
├── nanochat_diffusion/        # Core library
├── scripts/                   # Entry points
└── pyproject.toml             # Dependencies
```

Override the base directory with `NANOCHAT_BASE_DIR`:
```bash
export NANOCHAT_BASE_DIR=/path/to/storage
```

---

## Code Structure

```
nanochat_diffusion/
├── diffusion_model.py        # Core: DiffusionModel (GPT + timestep conditioning)
├── diffusion_scheduler.py    # Noise schedules (linear, cosine, exponential)
├── diffusion_sampler.py      # Progressive denoising for inference
├── gpt.py                    # GPT transformer (backbone for diffusion)
├── checkpoint_manager.py     # Save/load with metadata
├── common.py                 # DDP, device detection, logging
├── tokenizer.py              # BPE tokenizer
├── dataloader.py             # Distributed tokenized dataloader
├── flash_attention.py        # FA3 wrapper (falls back to PyTorch SDPA)
├── optim.py                  # AdamW optimizer
├── engine.py                 # KV-cache inference
└── core_eval.py              # DCLM CORE evaluation

scripts/
├── diffusion_train.py        # Training entry point
├── diffusion_infer.py        # Inference/generation
├── diffusion_evaluate.py     # Benchmark evaluation
└── download_dataset.py       # Download FineWeb dataset

cuda_kernels/
├── rms_norm_triton.py        # Custom Triton RMS norm (faster)
├── fp4_linear.py             # FP4 quantized linear layer (experimental)
├── fp4_linear_kernels.cu     # FP4 CUDA kernels
└── bench_gemm_dispatch.py    # GEMM backend benchmark
```

---

## Noise Schedules

| Schedule | Behavior | Best For |
|----------|----------|----------|
| `linear` | Mask ratio increases linearly with t | General purpose |
| `cosine` | Slow start/finish, steep middle | Better gradient signal at extremes |
| `exponential` | Fast initial noise, gradual cleanup | Short generation |
| `constant` | Fixed mask ratio throughout | Ablation studies |

---

## Performance Tuning

### Blackwell (GB10/DGX Spark, sm_121)

```bash
TRITON_CUDA_ARCH=sm_120 python -m scripts.diffusion_train \
    --depth 8 --max-seq-len 512 --device-batch-size 16 \
    --compile --compile-mode reduce-overhead --custom-rmsnorm \
    --num-iterations 120
```

Key findings (DGX Spark, depth=8, n_embd=512, bs=16, seq=512):
- **Best**: 216ms/iter — compile + custom RMS norm + `TRITON_CUDA_ARCH=sm_120`
- `compile` gives ~25% speedup over eager
- `--custom-rmsnorm` gives ~7% over `F.rms_norm`
- cuBLAS is already the default GEMM backend on sm_121

### Low VRAM
```bash
python -m scripts.diffusion_train \
    --depth 4 --max-seq-len 256 --device-batch-size 4 \
    --compile
```

---

## License

MIT (same as [nanochat](https://github.com/karpathy/nanochat))

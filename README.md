# nanochat-diffusion

A diffusion-based language model framework built on top of [karpathy/nanochat](https://github.com/karpathy/nanochat). Implements the Diffusion LLM (dLLM) paradigm: instead of autoregressive next-token prediction, the model masks tokens at various noise levels and learns to denoise them in parallel.

---

## What This Does

**Diffusion LLMs** work fundamentally differently from traditional LLMs:

| | Traditional LLMs (GPT) | Diffusion LLMs |
|---|---|---|
| **Generation** | Autoregressive — token by token | Iterative denoising — all tokens in parallel |
| **Training** | Predict next token | Predict original tokens from masked input |
| **Inference steps** | T tokens (sequential) | N steps (parallel per step) |
| **Flexibility** | Single continuation | Multiple continuations, partial conditioning |

### How It Works

**Training Phase:**
1. Take a sequence of tokens, e.g. `[BOS, THE, QUICK, BROWN, FOX, ...]`
2. Randomly sample a noise level (timestep `t` from 0 to 1000)
3. Mask tokens at that noise level (e.g., 80% masked with `UNK`)
4. Feed the partially-masked sequence through the transformer
5. Compute loss only on the masked positions (predict original tokens)

**Inference Phase:**
1. Initialize with all tokens as `UNK` (plus `BOS`)
2. Progressive denoising: each step predicts values for UNK positions and fills them in
3. After 10-20 steps, the sequence converges to a coherent output
4. All tokens are predicted in parallel per step, not one-by-one

### Key Advantages
- **Parallel denoising**: All positions predicted simultaneously (vs sequential for autoregressive)
- **Flexible generation**: Can condition on partial prompts, then complete the rest
- **Multiple continuations**: Generate different completions from the same prompt by varying the denoising path
- **Noise schedule control**: Adjust how quickly or gradually tokens are revealed

---

## Architecture

```
nanochat_diffusion/
├── __init__.py                    # Package init
├── common.py                      # Utilities, DDP setup, print0
├── gpt.py                        # GPT transformer architecture (from nanochat)
├── optim.py                      # AdamW optimizer (from nanochat)
├── tokenizer.py                  # BPE tokenizer (from nanochat)
├── dataloader.py                 # Distributed dataloaders
├── dataset.py                    # Data loading utilities
├── engine.py                     # Efficient inference with KV cache
├── core_eval.py                  # DCLM CORE score evaluation
├── checkpoint_manager.py         # Save/load checkpoints
├── flash_attention.py            # Flash Attention 3 wrapper
├── diffusion_model.py            # ** Core: Diffusion LLM model **
├── diffusion_scheduler.py        # Noise schedules (linear, cosine, etc.)
├── diffusion_sampler.py          # Progressive denoising sampler
├── tasks.py                      # Evaluation tasks (gsm8k, mmlu, arc, etc.)
├── runs/train_diffusion.sh       # Automated training script
├── scripts/
│   ├── diffusion_train.py        # Main training entry
│   ├── diffusion_infer.py        # Inference/generation entry
│   └── diffusion_evaluate.py     # Benchmark evaluation entry
├── ui.html                       # ChatGPT-like web UI (from nanochat)
├── logo.svg                      # Logo (from nanochat)
├── pyproject.toml                # Dependencies
├── LICENSE                       # MIT
├── README.md                     # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
# Clone nanochat (base)
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Install using uv
uv sync --extra gpu          # CUDA (A100/H100/etc.)
source .venv/bin/activate
```

### 2. Train a diffusion model

```bash
# Simple training on CPU
python -m scripts.diffusion_train

# Training with GPU (8x H100)
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
    -m scripts.diffusion_train \
    --depth=8 \
    --aspect-ratio=64 \
    --max-seq-len=1024 \
    --device-batch-size=16 \
    --lr=4e-4 \
    --warmup-iters=100 \
    --num-diffusion-steps=1000 \
    --max-mask-ratio=0.8 \
    --noise-schedule=linear \
    --unk-token-id=32767 \
    --vocab-size=32768
```

### 3. Generate text with diffusion sampling

```bash
# CLI generation
python -m scripts.diffusion_infer --prompt "Hello world" --model diffusion

# Web UI
python -m scripts.chat_web
```

### 4. Evaluate the model

```bash
python -m scripts.diffusion_evaluate --model diffusion --tasks gsm8k,arc
```

---

## Detailed Usage

### Training (`diffusion_train.py`)

**CLI Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--run` | `diffusion_demo` | W&B run name |
| `--model` | `diffusion` | Model type: `diffusion` or `gpt` |
| `--device` | `cuda` | Device to train on |
| `--depth` | `8` | Transformer depth (number of layers) |
| `--aspect-ratio` | `64` | Aspect ratio for computing model dimension (`n_embd = depth * aspect_ratio`) |
| `--max-seq-len` | `1024` | Maximum sequence length |
| `--window-pattern` | `SSSL` | Sliding window attention pattern |
| `--device-batch-size` | `16` | Per-device batch size |
| `--compile` | `False` | Enable `torch.compile` |
| `--warmup-iters` | `100` | Warmup iterations |
| `--lr` | `4e-4` | Base learning rate |
| `--weight-decay` | `0.1` | Weight decay |
| `--beta1` | `0.8` | AdamW beta1 |
| `--beta2` | `0.95` | AdamW beta2 |
| `--num-diffusion-steps` | `1000` | Total diffusion steps for training |
| `--sampling-steps` | `20` | Denoising steps for inference |
| `--max-mask-ratio` | `0.8` | Maximum token mask ratio (0.0-1.0) |
| `--noise-schedule` | `linear` | Noise schedule: `linear`, `cosine`, `exponential`, `constant` |
| `--unk-token-id` | `32767` | UNK token ID |
| `--vocab-size` | `32768` | Vocabulary size |
| `--num-iterations` | `-1` | Training iterations (`-1` = use target flops) |
| `--target-flops` | `-1.0` | Target FLOPs for training horizon |
| `--target-param-data-ratio` | `12` | Target data:param ratio |
| `--eval-iters` | `100` | Evaluate every N steps |
| `--save-every` | `1000` | Save checkpoint every N steps |
|| `--resume` | `""` | Resume from checkpoint step |
|| `--num-cpus` | `all` | Number of CPU cores to use for tokenization (integer or `all`) |

**Example — Quick experiment:**

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 \
    -m scripts.diffusion_train \
    --depth=4 \
    --max-seq-len=256 \
    --device-batch-size=8 \
    --num-diffusion-steps=500 \
    --max-mask-ratio=0.6 \
    --noise-schedule=coshine \
    --lr=1e-3 \
    --eval-iters=50 \
    --save-every=200 \
    --run="d4_test"
```

**Example — Full training (GPT-2 capability):**

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
    -m scripts.diffusion_train \
    --depth=26 \
    --aspect-ratio=64 \
    --max-seq-len=2048 \
    --device-batch-size=32 \
    --lr=3e-4 \
    --warmup-iters=200 \
    --num-diffusion-steps=1000 \
    --max-mask-ratio=0.85 \
    --noise-schedule=linear \
    --vocab-size=32768 \
    --run="d26_diffusion" \
    --save-every=500 \
    --eval-iters=100
```

**Resuming from checkpoint:**

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
    -m scripts.diffusion_train \
    --depth=26 \
    --resume=step_000000500  # or --resume=latest
```

### Inference (`diffusion_infer.py`)

**CLI Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `diffusion` | Model type |
| `--checkpoint-dir` | `""` | Checkpoint directory |
| `--checkpoint-step` | `latest` | Specific checkpoint step or `latest` |
| `--device` | `""` | Device (`""` = autodetect) |
| `--seq-len` | `256` | Output sequence length |
| `--max-tokens` | `128` | Max tokens to generate |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-k` | `0` | Top-k filtering (0 = disabled) |
| `--num-steps` | `20` | Denoising steps |
| `--prompt` | `""` | Text prompt to continue |
| `--mode` | `diffusion` | Inference mode: `diffusion`, `autoregressive`, `both` |
| `--output-file` | `""` | Save output to file |

**Examples:**

```bash
# Diffusion sampling from prompt
python -m scripts.diffusion_infer \
    --model diffusion \
    --prompt "The quick brown fox" \
    --max-tokens 64 \
    --temperature 0.8 \
    --num-steps 20 \
    --mode diffusion

# Autoregressive generation (token by token)
python -m scripts.diffusion_infer \
    --model diffusion \
    --prompt "Once upon a time" \
    --max-tokens 128 \
    --temperature 0.7 \
    --top-k 50 \
    --mode autoregressive

# Compare both modes
python -m scripts.diffusion_infer \
    --model diffusion \
    --prompt "Hello world" \
    --max-tokens 64 \
    --temperature 0.8 \
    --num-steps 15 \
    --mode both \
    --output-file "comparison.txt"
```

### Evaluation (`diffusion_evaluate.py`)

```bash
# Evaluate on all default tasks
python -m scripts.diffusion_evaluate \
    --model diffusion \
    --checkpoint-step latest \
    --tasks gsm8k,mmlu,arc,spellingbee

# Custom output
python -m scripts.diffusion_evaluate \
    --model diffusion \
    --output-file results.json \
    --seq-len 512

# Evaluate only specific metrics
python -m scripts.diffusion_evaluate \
    --model diffusion \
    --checkpoint-step latest \
    --evaluate-perplexity \
    --evaluate-generation \
    --evaluate-consistency
```

**Supported Tasks:**
- **GSM8K**: Grade school math word problems
- **MMLU**: Multiple choice across 57 subjects
- **ARC**: Alphabetical reasoning challenges
- **Spelling Bee**: Spell/count letters
- **HumanEval**: Simple Python coding tasks
- **Custom JSON**: Any JSONL format via `customjson`

---

## Hyperparameter Reference

### Diffusion-Specific Parameters

| Parameter | Range | Recommended | Description |
|---|---|---|---|
| `num-diffusion-steps` | `100-5000` | `1000` | Number of training timesteps. More steps = smoother schedule but slower training. |
| `sampling-steps` | `5-100` | `20` | Number of denoising steps during inference. Fewer = faster but lower quality. |
| `max-mask-ratio` | `0.1-1.0` | `0.8` | Maximum fraction of tokens masked during training. Higher = harder learning signal. |
| `noise-schedule` | string | `linear` | Noise schedule: `linear` (uniform), `cosine` (DDPM-style), `exponential`, `constant` |

### Model Parameters

| Parameter | Range | Description |
|---|---|---|
| `depth` | `2-64` | Number of transformer layers. Controls model capacity. |
| `aspect-ratio` | `32-128` | `n_embd = depth * aspect_ratio`. Controls width. |
| `max-seq-len` | `64-4096` | Maximum sequence length. |
| `vocab-size` | `32768` | Vocabulary size. |
| `unk-token-id` | `0-65535` | UNK token ID. Must be >= vocab_size-1. |

### Training Parameters

| Parameter | Range | Description |
|---|---|---|
| `lr` | `1e-5 - 1e-2` | Base learning rate. Lower for deeper models. |
| `weight-decay` | `0.01-0.5` | L2 regularization. |
| `warmup-iters` | `10-500` | Learning rate warmup steps. |
| `device-batch-size` | `1-64` | Batch size per GPU. |
| `num-iterations` | `-1 to N` | Training iterations (`-1` = auto). |

---

## Noise Schedules Explained

### Linear
Mask ratio increases linearly with timestep:
```
t=0:   0% masked
t=500: 40% masked
t=1000: 80% masked
```
Simple, predictable, works well.

### Cosine
Matches DDPM-style cosine schedule (more weight on early/late steps):
```
t=0:    0% masked
t=250:  15% masked (slow start)
t=500:  45% masked
t=750:  65% masked
t=1000: 80% masked (slow finish)
```
More nuanced — early denoising gets stronger gradient signal.

### Exponential
Exponential masking schedule:
```
t=0:    0% masked
t=100:  20% masked
t=500:  60% masked
t=1000: 80% masked
```
Fast initial noise, gradual cleanup.

### Constant
Fixed mask ratio throughout:
```
t=0-1000: always 50% masked
```
Good for ablation studies.

---

## Inference Modes Explained

### Diffusion Mode
1. Start with all tokens as `UNK` (e.g., `[UNK, UNK, UNK, ...]`)
2. Each step, forward pass predicts values for ALL `UNK` positions simultaneously
3. Replace `UNK` tokens with predicted values
4. Repeat for `num_steps` (typically 20)
5. After convergence, read the clean sequence

**Use case**: Full generation from scratch, or partial prompting.

### Autoregressive Mode
1. Start with prompt
2. Fill rest with `UNK`
3. Predict one position at a time
4. Append to prompt
5. Repeat

**Use case**: Conditional generation, familiar ChatGPT-like behavior.

### Both Mode
Run both modes side by side for comparison. Useful for benchmarking.

---

## Checkpoint Format

Checkpoints are saved at `runs/checkpoints/<model_name>/<phase>/`:

```
runs/checkpoints/diffusion/train/
├── step_000000001/
│   ├── model.pt           # Model weights
│   ├── config.json         # Architecture config
│   ├── optimizer.pt        # Optimizer state
│   └── metadata.json       # Training metadata (loss, step, timestamp)
├── step_000000500/
│   ├── ...
└── step_000001000/
    └── ...
```

**Resuming:**
```bash
--resume=latest          # Resume from most recent
--resume=step_000000500  # Resume from specific step
```

---

## Scaling Guide

| Scale | depth | seq_len | GPUs | VRAM | Use Case |
|---|---|---|---|---|---|
| **Tiny** | 2-4 | 64 | 1 | <8GB | Quick experiments |
| **Small** | 6-8 | 256 | 2-4 | <24GB | Prototype validation |
| **Medium** | 12-16 | 512-1024 | 4-8 | <80GB | Research experiments |
| **Large** | 24-32 | 1024-2048 | 8 | ~80GB each | Production training |
| **GPT-2** | ~26 | 2048 | 8xH100 | ~80GB each | GPT-2 capability |

**Single GPU note**: Omit `torchrun`, all code works on a single GPU but 8× slower.

**Low VRAM**: Reduce `--device-batch-size` from 32 → 16 → 8 → 4 → 2 → 1.

---

## Architecture Diagram

```
Input Sequence: [BOS, THE, QUICK, BROWN, FOX, ...]
                  │
          ┌──────┴──────┐
          │  mask_tokens │  ← Randomly mask at noise level
          └──────┴──────┘
                  │
          Partially Masked:
          [BOS, UNK, QUICK, UNK, UNK, ...]
                  │
          ┌──────┴──────┐
          │ Diffusion     │
          │ Transformer   │
          │ (GPT + Timestep │
          │  Embedding +  │
          │  UNK Type)    │
          └──────┴──────┘
                  │
          Predicted Tokens:
          [BOS, THE, QUICK, BROWN, FOX, ...]
                  │
          ┌──────┴──────┐
          │ Loss on      │  ← Cross-entropy on masked only
          │ masked only  │
          └──────┴──────┘
```

---

## File Structure

### Core Models
- **`diffusion_model.py`** — The main DiffusionModel: wraps GPT with timestep embedding, UNK type embedding, iterative denoising
- **`diffusion_scheduler.py`** — Noise schedules: linear, cosine, exponential, constant
- **`diffusion_sampler.py`** — Progressive denoising sampler for inference
- **`gpt.py`** — Standard GPT architecture (from nanochat)
- **`optim.py`** — AdamW + Muon optimizer (from nanochat)
- **`engine.py`** — KV cache for efficient inference
- **`flash_attention.py`** — Flash Attention 3 wrapper

### Utilities
- **`common.py`** — Distributed training setup, print0, dtype handling
- **`tokenizer.py`** — BPE tokenizer (GPT-4 style)
- **`dataloader.py`** — Distributed data loaders (best-fit cropping)
- **`dataset.py`** — Data loading utilities
- **`checkpoint_manager.py`** — Save/load checkpoints with metadata
- **`core_eval.py`** — DCLM CORE score evaluation

### Scripts
- **`scripts/diffusion_train.py`** — Main training loop
- **`scripts/diffusion_infer.py`** — Inference/generation
- **`scripts/diffusion_evaluate.py`** — Benchmark evaluation
- **`runs/train_diffusion.sh`** — Automated training script

---

## Training Loop (Step by Step)

```python
# 1. Load batch: shape (B, T)
batch = next(dataloader)          # e.g., (16, 1024)

# 2. Sample random timesteps for the batch
t = torch.randint(0, 1000, (B,))  # (16,)  ← one timestep per sequence

# 3. Mask tokens at the sampled noise levels
masked = model.mask_tokens(batch, t)  # Some tokens → UNK_ID

# 4. Forward pass through the diffusion model
logits = model(masked, t=t, targets=batch)  # (B, T, vocab_size)

# 5. Compute loss on masked positions only
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    batch.view(-1),
    ignore_index=0  # Ignore BOS
)

# 6. Backward pass
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**Key difference from autoregressive:** The loss is computed on ALL masked positions simultaneously, not sequentially. Each position gets a gradient signal for its correct token.

---

## Inference Loop (Step by Step)

```python
# 1. Initialize with all UNK
current_tokens = [UNK, UNK, UNK, ..., UNK]  # (seq_len,)

# 2. Progressive denoising
for step in range(num_steps):
    # Forward pass
    logits = model(current_tokens)  # (1, seq_len, vocab_size)
    
    # Find which positions are still UNK
    unk_positions = [i for i, t in enumerate(current_tokens) if t == UNK_ID]
    
    # Sample predicted values for UNK positions
    for pos in unk_positions:
        probs = logits[0, pos]
        predicted_token = sample(probs)
        current_tokens[pos] = predicted_token
    
    # Progress tracking
    if all_determined(): break
    
# 3. Read clean sequence
sequence = current_tokens  # All positions filled
```

---

## Troubleshooting

### Out of Memory
- Reduce `--device-batch-size`
- Reduce `--max-seq-len`
- Reduce `--depth`
- Use `--noise-schedule=linear` (simplest memory footprint)

### Loss Not Decreasing
- Check `--lr` is appropriate (try `1e-3` for small models, `3e-4` for large)
- Increase `--max-mask-ratio` (e.g., `0.9` for harder training signal)
- Ensure `--num-diffusion-steps` is reasonable (e.g., `1000`)
- Check `--warmup-iters` is sufficient for the batch size

### Poor Generation Quality
- Increase `--num-steps` during inference (e.g., `40` instead of `20`)
- Adjust `--temperature` (lower = more deterministic)
- Try different `--noise-schedule` (cosine often works better than linear)
- Increase training iterations

### Vocab Mismatch
- Ensure `--unk-token-id >= vocab_size - 1` (e.g., `32767` for vocab `32768`)
- The UNK token must be outside the normal vocabulary range

---

## Research Questions

1. **Scaling Laws**: How does diffusion LLM performance scale with model size and depth?
2. **Noise Schedules**: Does linear/cosine/exponential matter for final quality?
3. **Parallel vs Autoregressive**: What's the actual inference speedup?
4. **Conditioning**: How well does it handle partial prompts?
5. **Quality**: Can diffusion LLMs match autoregressive LLMs at same compute budget?
6. **Diversity**: Can one model produce multiple different continuations?

---

## Running the Speedrun Script

Automated training run:

```bash
bash runs/train_diffusion.sh
```

This runs the full training pipeline with default hyperparameters for GPT-2 capability. Monitor with W&B.

---

## License

MIT License (same as nanochat)

---

**Built on top of:** [karpathy/nanochat](https://github.com/karpathy/nanochat) and [karpathy/nanogpt](https://github.com/karpathy/nanogpt)

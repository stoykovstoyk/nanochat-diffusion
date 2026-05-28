# Autoresearch: Optimize Diffusion Training Speed on DGX Spark (GB10)

## Objective
Optimize the diffusion LLM training speed on the DGX Spark (NVIDIA GB10 Blackwell GPU).
Benchmark: depth=8, n_embd=512, batch=8, seq_len=256, 100 iterations.

## Metrics
- **Primary**: ms_per_iter (lower is better, measured at steady-state after compile warmup)
- **Secondary**: final_loss (should stay below ~4.0 for meaningful training)

## How to Run
```bash
TRITON_CUDA_ARCH=sm_120 uv run python -m scripts.diffusion_train \
  --depth 8 --aspect-ratio 64 --device-batch-size 16 --max-seq-len 512 \
  --compile --custom-rmsnorm \
  --num-iterations 120 --warmup-iters 100 --eval-iters 1000
```
Note: `TRITON_CUDA_ARCH=sm_120` required for correct triton codegen on sm_121.

## Files in Scope
- `scripts/diffusion_train.py` — Training loop, data loading, optim config
- `nanochat_diffusion/diffusion_model.py` — Diffusion model forward pass
- `nanochat_diffusion/gpt.py` — Base GPT model (Linear, attention, etc.)
- `nanochat_diffusion/common.py` — Dtype detection, compute init
- `nanochat_diffusion/dataloader.py` — Data loading pipeline
- `cuda_kernels/rms_norm_triton.py` — Custom Triton RMS norm (used via --custom-rmsnorm)
- `cuda_kernels/rms_norm_kernels.cu`, `fused_rms_norm.py` — Deprecated ctypes versions
- `pyproject.toml` — Dependency management, torch version

## Off Limits
- `nanochat_diffusion/tokenizer.py`, `flash_attention.py`, `engine.py`, `optim.py`
- `data/` files and sample generation scripts

## Constraints
- Training must converge (final loss < 4.0)
- No new external dependencies
- Must work with `uv sync --extra gpu` (cu130 torch)

## What's Been Tried (this session)

### Current Best: 217.4ms/iter (−43% vs baseline 384.4ms)
- depth=8, n_embd=512, bs=16, seq=512, 120 iters (100 warmup)
- Config: `--compile --custom-rmsnorm` + `TRITON_CUDA_ARCH=sm_120`

### Custom Triton RMS Norm (replaces ctypes version)
- **Problem**: ctypes-based CUDA kernel breaks `torch.compile` (Dynamo can't trace ctypes → empty CUDAGraph)
- **Solution**: Triton kernel `cuda_kernels/rms_norm_triton.py` — forward+backward autograd Function, handles any dtype
- **Speedup**: 2.82x vs F.rms_norm standalone (0.266ms vs 0.750ms fwd+bwd)
- **Key fix**: Original ctypes kernel failed because model activations are float32, not bf16. Triton kernel auto-adapts.

### triton 3.7.0 Regression Fix
- **Problem**: triton 3.7.0 generates wrong-arch kernels for sm_121 by default
- **Fix**: `TRITON_CUDA_ARCH=sm_120` env var forces correct sm_120 codegen
- **Impact**: Compile faster than eager (257ms vs 357ms) instead of slower (357ms vs 347ms)

### FP4 on SM121 CONFIRMED (HW supports it!)
- `__nv_fp4_e2m1` (E2M1: 1s/2e/1m) data type works on GB10
- `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32` instruction works
- **Critical**: Must compile for `compute_121a` / `sm_121a` (NOT sm_120a)
- CUDA 13.0.88 ptxas DOES support FP4 MMA with compute_120a target
- CUTLASS 4.4.2 builds for sm_120a but sm_120a binaries DON'T run on sm_121
- FP4 block-scaled GEMM would give 2x FP8 / 4x BF16 throughput

## Key Insights
- **GB10 is sm_121**: binary-compatible with sm_120 CUDA kernels, but NOT sm_120a (FP4 variant)
- **Triton 3.7.0 needs TRITON_CUDA_ARCH=sm_120** for correct sm_121 codegen
- **Model activations are float32**, not bf16 — embedding outputs float32, timestep addition upcasts
- **FP4 MMA on SM121 has non-standard routing** — D[0]=D[1], D[2]=D[3]; A values warp-level broadcast within thr_v groups; m-values not distinguished. CuTe SM80 INT8 layout incompatible. **ABANDONED**.
- **GEMM dispatch verified**: cuBLAS is default on sm_121; cuBLAS/cuBLASLt/Default all identical at ~60 TFLOPS
- **Triton < 3.7 blocked**: needs `python3.12-dev` (no sudo) to compile CUDA utils; triton 3.7 ships precompiled
- **Larger model (depth=12)**: 557ms/iter (better per-param utilization but slower absolute)
- **217.7ms/iter** is current best — GEMM (43%), element-wise (25%), attention (12%), optimizer (9%)

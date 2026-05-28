import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import time
import math
import argparse
import multiprocessing
import sys

import wandb
import torch
import torch.distributed as dist

torch.set_float32_matmul_precision('high')

from nanochat_diffusion.diffusion_model import DiffusionModel, DiffusionConfig
from nanochat_diffusion.diffusion_scheduler import create_noise_schedule, mask_tokens_simple
from nanochat_diffusion.tokenizer import Tokenizer, UNK_TOKEN_ID
from nanochat_diffusion.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat_diffusion.common import (
    compute_init, compute_cleanup, print0, DummyWandb, print_banner,
    get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON, is_ddp_initialized
)
from nanochat_diffusion.checkpoint_manager import save_checkpoint, load_checkpoint

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Train Diffusion LLM")

# Logging
parser.add_argument("--run", type=str, default="diffusion_demo", help="wandb run name")

# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--num-cpus", type=str, default="all", help="number of CPU cores to use (integer or 'all')")

# Model architecture
parser.add_argument("--depth", type=int, default=8, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=1024, help="max context length")

# Diffusion-specific config
parser.add_argument("--num-diffusion-steps", type=int, default=1000, help="total diffusion steps for training")
parser.add_argument("--sampling-steps", type=int, default=20, help="denoising steps for inference")
parser.add_argument("--max-mask-ratio", type=float, default=0.8, help="maximum token mask ratio")
parser.add_argument("--noise-schedule", type=str, default="linear", help="noise schedule type")
parser.add_argument("--unk-token-id", type=int, default=4095, help="UNK token ID (outside BPE vocab, within padded range)")

# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate iterations to reach target flops")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="target data:param ratio")

# Optimization
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size")
parser.add_argument("--attention-backend", type=str, default="auto",
                    choices=["auto", "math", "flash", "mem_efficient", "cudnn"],
                    help="SDPA attention backend (default: auto)")
parser.add_argument("--compile", action="store_true", help="torch.compile the model")
parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    help="torch.compile mode (default: reduce-overhead)")
parser.add_argument("--fullgraph", action="store_true", help="use fullgraph=True with torch.compile (more aggressive fusion)")
parser.add_argument("--cudnn-benchmark", action="store_true", help="enable torch.backends.cudnn.benchmark")
parser.add_argument("--custom-rmsnorm", action="store_true", help="use custom fused CUDA RMS norm kernel")
parser.add_argument("--warmup-iters", type=int, default=50, help="warmup steps")
parser.add_argument("--grad-clip", type=float, default=0.0, help="gradient clipping (0=disabled)")
parser.add_argument("--lr", type=float, default=4e-4, help="base learning rate")
parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay")
parser.add_argument("--beta1", type=float, default=0.8, help="AdamW beta1")
parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")

# Eval & save
parser.add_argument("--eval-iters", type=int, default=100, help="eval every N steps")
parser.add_argument("--eval-only", action="store_true", help="just evaluate and exit")
parser.add_argument("--eval-init-only", action="store_true", help="initial eval only, no training")
parser.add_argument("--eval-batches", type=int, default=50, help="number of batches to evaluate per eval pass")
parser.add_argument("--save-every", type=int, default=1000, help="save checkpoint every N steps")
parser.add_argument("--resume", type=str, default="", help="resume from checkpoint step")

# Data
parser.add_argument("--vocab-size", type=int, default=4096, help="vocab size (4096 = BOS + 4094 BPE tokens + UNK)")
parser.add_argument("--tokenizer-batch-size", type=int, default=128, help="batch size for tokenization")

args = parser.parse_args()

# Apply cudnn benchmark
if args.cudnn_benchmark:
    torch.backends.cudnn.benchmark = True
    print0("Enabled cudnn benchmark")

# -----------------------------------------------------------------------------
# Parse CPU core count
if args.num_cpus == "all":
    num_cpus = multiprocessing.cpu_count()
else:
    try:
        num_cpus = int(args.num_cpus)
        if num_cpus < 1:
            raise ValueError("num_cpus must be >= 1")
    except ValueError as e:
        raise argparse.ArgumentError(None, f"Invalid --num_cpus value: {args.num_cpus} (use an integer >= 1 or 'all')") from e
print0(f"CPU cores: {num_cpus}")

# -----------------------------------------------------------------------------
# Compute initialization
device_type = args.device_type or autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# Diffusion LLM setup
print0(f"Setting up Diffusion LLM (depth={args.depth}, seq_len={args.max_seq_len})")

n_embd = args.depth * args.aspect_ratio
print0(f"Model dimension: {n_embd}")

n_head = n_embd // 128
n_kv_head = n_head
print0(f"n_head: {n_head}, n_kv_head: {n_kv_head}")

diffusion_config = DiffusionConfig(
    sequence_len=args.max_seq_len,
    vocab_size=args.vocab_size,
    n_layer=args.depth,
    n_head=n_head,
    n_kv_head=n_kv_head,
    n_embd=n_embd,
    window_pattern="SSSL",
    num_diffusion_steps=args.num_diffusion_steps,
    unk_token_id=args.unk_token_id,
    max_mask_ratio=args.max_mask_ratio,
    sampling_steps=args.sampling_steps,
)

model = DiffusionModel(diffusion_config)
model.to(device)

base_dir = get_base_dir()
tokenizer_path = os.path.join(base_dir, "tokenizer_diffusion")
os.makedirs(tokenizer_path, exist_ok=True)

# Check if BPE is already trained, otherwise train from data
tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
if not os.path.exists(tokenizer_json):
    import glob
    parquet_files = sorted(glob.glob(os.path.join(base_dir, "train_*.parquet")))
    if not parquet_files:
        print0("WARNING: No training data found for BPE tokenizer. Use download_dataset.py first.")
        tokenizer = Tokenizer(tokenizer_path, verbose=True)
    else:
        print0("Training BPE tokenizer from parquet data...")
        import pyarrow.parquet as pq
        def text_iter():
            for path in parquet_files:
                table = pq.read_table(path, columns=["text"])
                for batch in table.to_batches():
                    for t in batch.column("text").to_pylist():
                        if t:
                            yield t
        tokenizer = Tokenizer(data_dir="", verbose=False)
        tokenizer.train(text_iter(), vocab_size=32768)
        tokenizer.save(tokenizer_json)
        print0(f"BPE tokenizer trained and saved to {tokenizer_json}")
else:
    tokenizer = Tokenizer(tokenizer_path, verbose=True)

# -----------------------------------------------------------------------------
# Setup tokenizer data loader
print0("Setting up data loader...")
print0(f"Parallelism: {num_cpus} tokenizer threads, buffer_size={num_cpus}")
dataloader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer=tokenizer,
    B=args.device_batch_size,
    T=args.max_seq_len,
    split="train",
    tokenizer_threads=num_cpus,
    tokenizer_batch_size=args.tokenizer_batch_size,
    device=str(device),
    resume_state_dict=None,
    buffer_size=num_cpus,
)

def get_dataloader(split="train"):
    if split == "train":
        return dataloader
    return tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer=tokenizer,
        B=args.device_batch_size,
        T=args.max_seq_len,
        split="val",
        tokenizer_threads=num_cpus,
        tokenizer_batch_size=args.tokenizer_batch_size,
        device=str(device),
        resume_state_dict=None,
        buffer_size=num_cpus,
    )

# -----------------------------------------------------------------------------
# Set up optimizer
print0("Setting up optimizer...")
diffusion_params = list(model.parameters())
optimizer = torch.optim.AdamW(
    diffusion_params,
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay,
    fused=True
)
print0(f"Optimizer: AdamW with lr={args.lr}, weight_decay={args.weight_decay}")

# LR scheduler: linear warmup + cosine decay (or constant after warmup for infinite)
total_iters = args.num_iterations
warmup_iters = args.warmup_iters
if total_iters > 0 and warmup_iters > 0:
    def lr_lambda(step):
        if step < warmup_iters:
            return step / max(1, warmup_iters)
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print0(f"LR scheduler: linear warmup {warmup_iters} -> cosine decay {total_iters}")
elif total_iters <= 0 and warmup_iters > 0:
    # Infinite training: warmup then constant LR
    def lr_lambda(step):
        return min(1.0, step / max(1, warmup_iters))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print0(f"LR scheduler: linear warmup {warmup_iters} -> constant (infinite training)")
else:
    scheduler = None

# -----------------------------------------------------------------------------
# Setup wandb
try:
    wandb_run = wandb.init(
        project="nanochat_diffusion",
        name=args.run,
        config=vars(args),
        tags=["diffusion", "llm"],
    )
except Exception:
    print0("wandb not configured, using DummyWandb")
    import wandb as _wandb
    wandb_run = _wandb.init(
        project="nanochat_diffusion",
        name=args.run,
        config=vars(args),
        tags=["diffusion", "llm"],
        mode="disabled",
    )

# -----------------------------------------------------------------------------
# Training loop
print0("=" * 80)
print0("Starting training...")
print0("=" * 80)

train_start_time = time.time()
total_steps = 0
best_loss = float('inf')

peak_flops = get_peak_flops("cuda" if torch.cuda.is_available() else "unknown")

if args.resume:
    result = load_checkpoint(model, optimizer, step=args.resume, model_name="diffusion", phase="train")
    if result[0] is not None:
        model, metadata = result
        total_steps = metadata.get("step", 0) if metadata else 0
        print0(f"Resumed from step {total_steps}")
    else:
        print0(f"Could not resume from '{args.resume}', starting from scratch")

if args.eval_only:
    print0("Running evaluation only...")
    model.eval()
    sys.exit(0)

if args.attention_backend != "auto":
    import torch.backends.cuda as bc
    bc.enable_flash_sdp(args.attention_backend == "flash")
    bc.enable_mem_efficient_sdp(args.attention_backend == "mem_efficient")
    bc.enable_math_sdp(args.attention_backend == "math")
    bc.enable_cudnn_sdp(args.attention_backend == "cudnn")
    print0(f"Attention backend set to: {args.attention_backend}")

if args.custom_rmsnorm:
    from nanochat_diffusion.gpt import use_custom_rmsnorm
    use_custom_rmsnorm(True)
    print0("Using custom fused CUDA RMS norm kernel")

if args.compile:
    fg = "fullgraph" if args.fullgraph else "partial"
    print0(f"Compiling model with torch.compile (mode={args.compile_mode}, {fg})...")
    model = torch.compile(model, mode=args.compile_mode, fullgraph=args.fullgraph)

# Training loop
model.train()
losses = []
best_loss = float('inf')

dataloader = get_dataloader("train")

for step_idx, (input_tokens, target_tokens) in enumerate(dataloader):
    total_steps += 1
    if total_steps == 1 or total_steps == 6:
        print0(f"Step {total_steps}: batch shape={input_tokens.shape}")

    inputs, targets = input_tokens, target_tokens
    B, T = inputs.shape

    # Generate timestep on the fly (supports infinite training with num_iterations=-1)
    t = torch.randint(0, args.num_diffusion_steps, (B,), device=device)

    masked_tokens = model.mask_tokens(inputs, t)

    loss = model(masked_tokens, t=t, targets=targets)

    loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if scheduler:
        scheduler.step()

    losses.append(loss.detach().clone())
    current_loss = losses[-1].item()

    avg_loss = sum(losses[-10:]) / min(10, len(losses))

    if total_steps % 50 == 0:
        print0(f"Step {total_steps}: loss = {avg_loss.item():.4f}, "
               f"lr = {optimizer.param_groups[0]['lr']:.6f}")

    # Save every save_every steps if loss improved
    if args.save_every > 0 and total_steps % args.save_every == 0 and current_loss < best_loss:
        best_loss = current_loss
        save_checkpoint(
            model, optimizer, total_steps, current_loss,
            {"final_loss": current_loss, "best_loss": best_loss},
            model_name="diffusion",
            phase="train"
        )

    if args.num_iterations > 0 and total_steps >= args.num_iterations:
        break

train_elapsed = time.time() - train_start_time
print0("=" * 80)
print0("Training complete!")
print0(f"Total steps: {total_steps}")
print0(f"Total time: {train_elapsed:.2f}s")
print0(f"Avg time/iter: {train_elapsed/total_steps*1000:.1f}ms")
print0(f"Final loss: {losses[-1].item():.4f}")
print0("=" * 80)

# Always save final checkpoint
final_loss_val = losses[-1].item() if hasattr(losses[-1], 'item') else float(losses[-1])
save_checkpoint(
    model, optimizer, total_steps, final_loss_val,
    {"final_loss": final_loss_val},
    model_name="diffusion",
    phase="train"
)

try:
    wandb_run.finish()
except Exception:
    pass
compute_cleanup()

print0("Diffusion LLM training pipeline complete!")

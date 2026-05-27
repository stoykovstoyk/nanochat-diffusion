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

from nanochat_diffusion.gpt import GPT, GPTConfig, Linear
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
parser.add_argument("--model", type=str, default="diffusion", help="model type: 'diffusion' or 'gpt'")

# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--num-cpus", type=str, default="all", help="number of CPU cores to use (integer or 'all')")

# Model architecture
parser.add_argument("--depth", type=int, default=8, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=1024, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")

# Diffusion-specific config
parser.add_argument("--num-diffusion-steps", type=int, default=1000, help="total diffusion steps for training")
parser.add_argument("--sampling-steps", type=int, default=20, help="denoising steps for inference")
parser.add_argument("--max-mask-ratio", type=float, default=0.8, help="maximum token mask ratio")
parser.add_argument("--noise-schedule", type=str, default="linear", help="noise schedule type")
parser.add_argument("--unk-token-id", type=int, default=32767, help="UNK token ID")

# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate iterations to reach target flops")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="target data:param ratio")

# Optimization
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size")
parser.add_argument("--compile", action="store_true", help="torch.compile the model")
parser.add_argument("--warmup-iters", type=int, default=50, help="warmup steps")
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
parser.add_argument("--vocab-size", type=int, default=32768, help="vocab size")
parser.add_argument("--tokenizer-batch-size", type=int, default=128, help="batch size for tokenization")

args = parser.parse_args()

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

if args.model == "diffusion":
    # Diffusion LLM setup
    print0(f"Setting up Diffusion LLM (depth={args.depth}, seq_len={args.max_seq_len})")
    
    # Calculate model dimension
    n_embd = args.depth * args.aspect_ratio  # e.g., 8 * 64 = 512
    print0(f"Model dimension: {n_embd}")
    
    # Compute n_head dynamically (same as original nanochat: n_embd // 128)
    n_head = n_embd // 128
    n_kv_head = n_head  # GQA: n_kv_head == n_head
    print0(f"n_head: {n_head}, n_kv_head: {n_kv_head}")
    
    # Create diffusion config
    diffusion_config = DiffusionConfig(
        sequence_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.depth,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        window_pattern=args.window_pattern,
        num_diffusion_steps=args.num_diffusion_steps,
        unk_token_id=args.unk_token_id,
        max_mask_ratio=args.max_mask_ratio,
        sampling_steps=args.sampling_steps,
    )
    
    # Create model
    model = DiffusionModel(diffusion_config)
    model.to(device)
    
    # Setup tokenizer
    base_dir = get_base_dir()
    tokenizer_path = os.path.join(base_dir, "tokenizer_diffusion")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer = Tokenizer(tokenizer_path, verbose=True)
    
elif args.model == "gpt":
    # Standard GPT setup for comparison
    print0(f"Setting up GPT (depth={args.depth}, seq_len={args.max_seq_len})")
    
    n_embd = args.depth * args.aspect_ratio
    gpt_config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.depth,
        n_head=6,
        n_kv_head=6,
        n_embd=n_embd,
        window_pattern=args.window_pattern,
    )
    
    model = GPT(gpt_config)
    model.to(device)
    
    base_dir = get_base_dir()
    tokenizer_path = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer = Tokenizer(tokenizer_path, verbose=True)
else:
    raise ValueError(f"Unknown model type: {args.model}")

# -----------------------------------------------------------------------------
# Setup tokenizer data loader
print0("Setting up data loader...")
# Create sample data for demonstration
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world of artificial intelligence.",
    "Deep learning models have achieved remarkable results in NLP tasks.",
    "The diffusion process allows for iterative refinement of generated sequences.",
    "Attention mechanisms enable models to focus on relevant parts of input.",
] * 100  # Repeat for more data

# In production, this would load from parquet files
# For now, create synthetic data
class SyntheticDataset:
    def __init__(self, texts, tokenizer, max_seq_len=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            return torch.stack([self.__getitem__(int(i)) for i in idx])
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, prepend=True)
        # Pad or truncate to max_seq_len
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        else:
            tokens = tokens[:self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)

# For demo, use simple dataset
dataset = SyntheticDataset(sample_texts, tokenizer, args.max_seq_len)

# Create proper distributed dataloader driven by --num_cpus
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

def get_dataloader(split="train", num_workers=None):
    """Return the distributed dataloader for the requested split."""
    if split == "train":
        return dataloader
    # For eval, create a fresh iterator from the same config
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
if args.model == "diffusion":
    # For diffusion LLM, use AdamW with special learning rates
    # The diffusion-specific params go into AdamW
    diffusion_params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        diffusion_params,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=True
    )
else:
    # Standard GPT optimizer (simplified from nanochat)
    model_dim = n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    
    param_groups = [
        dict(params=list(model.transformer.wte.parameters()), lr=0.2 * dmodel_lr_scale),
        dict(params=list(model.lm_head.parameters()), lr=0.004 * dmodel_lr_scale),
        dict(params=list(model.transformer.h.parameters()), lr=0.02 * dmodel_lr_scale),
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

print0(f"Optimizer: AdamW with lr={args.lr}, weight_decay={args.weight_decay}")

# -----------------------------------------------------------------------------
# Setup wandb (optional, falls back to DummyWandb if no API key)
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

total_steps = 0
best_loss = float('inf')

# Get model to compute FLOPs
peak_flops = get_peak_flops("cuda" if torch.cuda.is_available() else "unknown")

# Check if we should just evaluate
if args.eval_only:
    print0("Running evaluation only...")
    model.eval()
    # Evaluation code would go here
    sys.exit(0)

# Training loop
model.train()
num_epochs = 3  # For demo, use fixed epochs
losses = []
eval_losses = []

for epoch in range(num_epochs):
    print0(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Create dataloader for this epoch
    dataloader = get_dataloader("train")
    
    for step_idx, (input_tokens, target_tokens) in enumerate(dataloader):
        total_steps += 1
        if total_steps == 1 or total_steps == 6:
            print0(f"Step {total_steps}: receiving batch, shape={input_tokens.shape}")
        
        # Prepare batch (dataloader yields (inputs, targets) tuples)
        if args.model == "diffusion":
            # For diffusion LLM:
            # - Sample random timestep
            # - Mask tokens at that noise level
            # - Forward pass with masked input
            # - Compute loss on unmasked positions
            
            inputs, targets = input_tokens, target_tokens
            B, T = inputs.shape
            
            # Sample random timestep for this batch
            t = torch.randint(0, args.num_diffusion_steps, (B,), device=device)
            
            # Mask tokens at this noise level
            masked_tokens = model.mask_tokens(inputs, t)
            
            # Forward pass
            loss = model(masked_tokens, t=t, targets=targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Store loss
            losses.append(loss.item())
            
        else:
            # Standard GPT forward pass
            targets = inputs[:, 1:]  # Next token
            inputs = inputs[:, :-1]
            
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0  # Ignore BOS
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(loss.item())
        
        # Print progress
        if total_steps % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            print0(f"Step {total_steps}: loss = {avg_loss:.4f}, "
                   f"lr = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if total_steps % args.save_every == 0:
            print0(f"Saving checkpoint at step {total_steps}")
            save_checkpoint(
                model, optimizer, total_steps, loss.item(),
                {"loss": loss.item(), "avg_loss": avg_loss},
                model_name=args.model,
                phase="train"
            )
        
        # Evaluate
        if total_steps % args.eval_iters == 0:
            model.eval()
            eval_losses_epoch = []
            
            with torch.no_grad():
                eval_dataloader = get_dataloader("eval")
                # Eval loop: cap at --eval-batches so it doesn't run forever
                for eval_step in range(args.eval_batches):
                    try:
                        eval_inputs, eval_targets = next(iter(eval_dataloader))
                    except StopIteration:
                        break
                    
                    if args.model == "diffusion":
                        t_eval = torch.zeros(eval_inputs.shape[0], device=device)
                        masked_eval = model.mask_tokens(eval_inputs, t_eval)
                        eval_loss = model(masked_eval, t=t_eval, targets=eval_targets)
                    else:
                        eval_logits = model(eval_inputs)
                        eval_loss = torch.nn.functional.cross_entropy(
                            eval_logits.view(-1, eval_logits.size(-1)),
                            eval_targets.view(-1),
                            ignore_index=0
                        )
                    
                    eval_losses_epoch.append(eval_loss.item())
            
            eval_loss = sum(eval_losses_epoch) / len(eval_losses_epoch)
            print0(f"Eval step {total_steps}: loss = {eval_loss:.4f}")
            
            eval_losses.append({
                'step': total_steps,
                'loss': eval_loss,
            })
            
            # Log to wandb
            if isinstance(wandb_run, wandb.wandb_run.Run):
                wandb_run.log({
                    'eval_loss': eval_loss,
                    'train_loss': avg_loss,
                    'step': total_steps,
                })
            
            model.train()
        
        # Check if we've reached target iterations
        if args.num_iterations > 0 and total_steps >= args.num_iterations:
            break
    
    if args.num_iterations > 0 and total_steps >= args.num_iterations:
        break

print0("=" * 80)
print0("Training complete!")
print0(f"Total steps: {total_steps}")
print0(f"Final loss: {losses[-1]:.4f}")
print0("=" * 80)

# Save final checkpoint
save_checkpoint(
    model, optimizer, total_steps, losses[-1],
    {"final_loss": losses[-1]},
    model_name=args.model,
    phase="train"
)

# Cleanup
try:
    wandb_run.finish()
except Exception:
    pass
compute_cleanup()

print0("Diffusion LLM training pipeline complete!")

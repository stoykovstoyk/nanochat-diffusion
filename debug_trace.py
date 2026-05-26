#!/usr/bin/env python3
"""Debug: trace where the training script hangs."""
import sys
sys.path.insert(0, '/tmp/nanochat_diffusion')

import os, json, time, math, argparse, multiprocessing
import torch, torch.distributed as dist

from nanochat_diffusion.common import (
    compute_init, compute_cleanup, print0, DummyWandb, print_banner,
    get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON, is_ddp_initialized
)
from nanochat_diffusion.gpt import GPT, GPTConfig, Linear
from nanochat_diffusion.diffusion_model import DiffusionModel, DiffusionConfig
from nanochat_diffusion.diffusion_scheduler import create_noise_schedule, mask_tokens_simple
from nanochat_diffusion.tokenizer import Tokenizer, UNK_TOKEN_ID
from nanochat_diffusion.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat_diffusion.dataset import list_parquet_files
from nanochat_diffusion.checkpoint_manager import save_checkpoint, load_checkpoint

print_banner()
print0("Step 1: compute_init")
device_type = "cpu"
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
print0(f"Step 2: device={device}, ddp={ddp}")

print0("Step 3: create diffusion model")
diffusion_config = DiffusionConfig(
    sequence_len=256,
    vocab_size=32768,
    n_layer=4,
    n_head=2,
    n_kv_head=2,
    n_embd=256,
    window_pattern='SSSL',
    num_diffusion_steps=1000,
    unk_token_id=32767,
    max_mask_ratio=0.8,
    sampling_steps=20,
)
model = DiffusionModel(diffusion_config)
model.to(device)
print0("Step 4: create tokenizer")
base_dir = get_base_dir()
tokenizer_path = os.path.join(base_dir, "tokenizer_diffusion")
os.makedirs(tokenizer_path, exist_ok=True)
tokenizer = Tokenizer(tokenizer_path, verbose=True)

print0("Step 5: create dataloader")
dataloader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer=tokenizer,
    B=4,
    T=256,
    split="train",
    tokenizer_threads=2,
    tokenizer_batch_size=128,
    device="cpu",
    buffer_size=2,
)
print0("Step 6: get first batch")
batch = next(iter(dataloader))
print0(f"Step 7: got batch - type={type(batch)}, len={len(batch)}")
print("DONE!")

"""
Distributed dataloaders for diffusion LLM pretraining.

Adapted from karpathy/nanochat/nanochat/dataloader.py
"""

import torch
import pyarrow.parquet as pq
from nanochat_diffusion.common import get_dist_info
from nanochat_diffusion.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches from parquet files.
    
    Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    
    resume_pq_idx = resume_state_dict.get("pq_idx", 0) if resume_state_dict else 0
    resume_rg_idx = resume_state_dict.get("rg_idx") if resume_state_dict else None
    first_pass = True
    
    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            rg_idx = ddp_rank if first_pass else (resume_rg_idx if resume_rg_idx is not None else ddp_rank)
            
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False

def tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", buffer_size=1000
):
    """Simplified dataloader without resume state."""
    state = {"resume_pq_idx": 0, "resume_rg_idx": None}
    return tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, B, T, split,
        tokenizer_threads=tokenizer_threads,
        tokenizer_batch_size=tokenizer_batch_size,
        device=device,
        resume_state_dict=state,
        buffer_size=buffer_size
    )


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None, buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.
    
    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding)
    - ~35% tokens discarded due to cropping
    """
    assert split in ["train", "val"]
    
    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    
    # Pre-allocate buffers
    use_cuda = device == "cuda"
    # row_buffer holds one row of T+1 tokens at a time (B rows, T+1 columns each)
    row_buffer = torch.empty(B, row_capacity, dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    
    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    doc_batch, _ = next(batches)
                    token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
                    for tokens in token_lists:
                        doc_buffer.append(tokens)
                
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        
        # Copy to GPU
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets

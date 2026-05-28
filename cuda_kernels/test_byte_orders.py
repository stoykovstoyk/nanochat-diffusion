"""Systematically test byte order conventions for A and B registers."""
import torch
import ctypes
import subprocess

M, N, K = 16, 8, 32
device = "cuda"

# Read original source
with open("/home/stoyko/Desktop/nanochat-diffusion/cuda_kernels/fp4_mma_gemm.cu") as f:
    src_base = f.read()

# Test data: A[m][k=0]=1 for ALL m, B all 1.0
A_full = torch.zeros(M, K, dtype=torch.uint8, device=device)
for row in range(M):
    A_full[row, 0] = 2
A_packed = A_full.reshape(-1)[::2] | (A_full.reshape(-1)[1::2] << 4)
B_nk = torch.full((N, K), 2, dtype=torch.uint8, device=device)
B_packed = B_nk.reshape(-1)[::2] | (B_nk.reshape(-1)[1::2] << 4)

# Test different byte order combinations
# A byte_off formulas:
# col_major: v + k0*4 + k1*8
# row_major: v*4 + k0*2 + k1
# 
# B byte_off formulas:
# col_major: v + k_val*4
# row_major: v*2 + k_val

def make_kernel(a_off_type, b_off_type):
    """Generate kernel source with given byte offset formulas."""
    src = src_base
    if a_off_type == "col":
        src = src.replace("v * 4 + k0 * 2 + k1", "v + k0 * 4 + k1 * 8")
    else:
        src = src.replace("v * 4 + k0 * 2 + k1", "v * 4 + k0 * 2 + k1")
    if b_off_type == "col":
        src = src.replace("v * 2 + k_val", "v + k_val * 4")
    else:
        src = src.replace("v * 2 + k_val", "v * 2 + k_val")
    # Fix comment too - keep it simple
    return src

# The current source already uses col_major for A (v + k0*4 + k1*8)
# Wait, let me check what the current source actually has
print("Current A byte_off in source:", src_base.count("v * 4 + k0 * 2 + k1"), "occurrences of row")
print("Current A byte_off in source:", src_base.count("v + k0 * 4 + k1 * 8"), "occurrences of col")
print("Current B byte_off in source:", src_base.count("v * 2 + k_val"), "occurrences of row")
print("Current B byte_off in source:", src_base.count("v + k_val * 4"), "occurrences of col")

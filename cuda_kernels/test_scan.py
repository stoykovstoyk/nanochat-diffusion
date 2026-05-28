"""Diagnose layout: scan all one-hot positions."""
import torch
import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary("/home/stoyko/Desktop/nanochat-diffusion/cuda_kernels/libfp4_mma.so")
lib.launch_fp4_mma_gemm.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.launch_fp4_mma_gemm.restype = None

M, N, K = 16, 8, 32
device = "cuda"

# Scan: for each (m_k, k_idx) in A and (k_idx, n_idx) in B,
# check where output peak goes
# Use A[m=0,k=0..31] * B[k=0..31,n=0] -> should all map to C[0][0]

# Set B to all-ones so any A non-zero at matching k produces output
B_nk = torch.zeros(N, K, dtype=torch.uint8, device=device)
for n in range(N):
    for k in range(K):
        B_nk[n, k] = 2  # 1.0 for all elements
B_flat_nk = B_nk.reshape(-1)
B_packed = B_flat_nk[::2] | (B_flat_nk[1::2] << 4)

print("Test: A[m=0,k=*] x B[all], check output at C[m=0,*]")
print("For each k, expected output at C[0][n] = sum over k: A[0][k] * B[k][n]")
print("With B all-ones, C[0][n] should equal sum_k A[0][k] * 1.0")
print()

for k_test in range(32):
    A_e2m1 = torch.zeros(M, K, dtype=torch.uint8, device=device)
    A_e2m1[0, k_test] = 2  # 1.0 at A[0, k_test]
    A_flat = A_e2m1.reshape(-1)
    A_packed = A_flat[::2] | (A_flat[1::2] << 4)

    C_out = torch.zeros(M, N, dtype=torch.float32, device=device)
    lib.launch_fp4_mma_gemm(
        ctypes.c_void_p(A_packed.data_ptr()),
        ctypes.c_void_p(B_packed.data_ptr()),
        ctypes.c_void_p(C_out.data_ptr()),
        M, N, K,
    )
    torch.cuda.synchronize()

    # Find non-zero outputs
    nz = (C_out.abs() > 0.1).nonzero(as_tuple=False)
    if len(nz) > 0:
        print(f"A[0,{k_test:2d}]: non-zero outputs at {nz.tolist()} (values={C_out[C_out.abs()>0.1].tolist()})")
    else:
        print(f"A[0,{k_test:2d}]: NO non-zero output!")

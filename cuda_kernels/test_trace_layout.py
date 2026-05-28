"""Empirically determine SM120 FP4 A layout:
For each byte_off within each thread, set it to 1.0 and see which C[m][n] gets output."""
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

# B all 1.0
B_nk = torch.full((N, K), 2, dtype=torch.uint8, device=device)
B_packed = B_nk.reshape(-1)[::2] | (B_nk.reshape(-1)[1::2] << 4)

# For each (m,k) position, set it to 1.0 and observe output
print("Tracing A layout (m,k) -> output C[m][n]:")
print("=" * 80)

for m_test in range(16):
    for k_test in range(0, 32, 8):  # Only 0, 8, 16, 24
        A = torch.zeros(M, K, dtype=torch.uint8, device=device)
        A[m_test, k_test] = 2  # 1.0
        A_packed = A.reshape(-1)[::2] | (A.reshape(-1)[1::2] << 4)

        C = torch.zeros(M, N, dtype=torch.float32, device=device)
        lib.launch_fp4_mma_gemm(
            ctypes.c_void_p(A_packed.data_ptr()),
            ctypes.c_void_p(B_packed.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            M, N, K,
        )
        torch.cuda.synchronize()

        nz = (C.abs() > 0.1).nonzero(as_tuple=False)
        if len(nz) > 0:
            rows = sorted(set(nz[:, 0].tolist()))
            cols = sorted(set(nz[:, 1].tolist()))
            val = C[nz[0][0], nz[0][1]].item()
            print(f"A[{m_test:2d},{k_test:2d}=1.0] -> C rows={rows} cols={cols} val={val:.2f}")

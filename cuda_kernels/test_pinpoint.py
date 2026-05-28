"""Pinpoint test: set A[0][0] and A[2][0] to different values, see output."""
import torch
import ctypes

lib = ctypes.cdll.LoadLibrary("/home/stoyko/Desktop/nanochat-diffusion/cuda_kernels/libfp4_mma.so")
lib.launch_fp4_mma_gemm.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.launch_fp4_mma_gemm.restype = None

M, N, K = 16, 8, 32
device = "cuda"

# A[0][0] = 1.0 (E2M1=2), A[2][0] = 0.5 (E2M1=1)
# B all 1.0 (E2M1=2)
# Expected: C[0][n] = 1.0, C[2][n] = 0.5 for all n

A_e2m1 = torch.zeros(M, K, dtype=torch.uint8, device=device)
A_e2m1[0, 0] = 2   # 1.0
A_e2m1[2, 0] = 1   # 0.5

A_flat = A_e2m1.reshape(-1)
A_packed = A_flat[::2] | (A_flat[1::2] << 4)

B_nk = torch.full((N, K), 2, dtype=torch.uint8, device=device)  # all 1.0
B_flat_nk = B_nk.reshape(-1)
B_packed = B_flat_nk[::2] | (B_flat_nk[1::2] << 4)

C_out = torch.zeros(M, N, dtype=torch.float32, device=device)
lib.launch_fp4_mma_gemm(
    ctypes.c_void_p(A_packed.data_ptr()),
    ctypes.c_void_p(B_packed.data_ptr()),
    ctypes.c_void_p(C_out.data_ptr()),
    M, N, K,
)
torch.cuda.synchronize()

print("A[0][0]=1.0, A[2][0]=0.5, B all 1.0")
print("Expected: C[0][*]=1.0, C[2][*]=0.5, all others 0")
print()
for m in range(M):
    for n in range(N):
        v = C_out[m,n].item()
        if abs(v) > 0.01:
            print(f"  C[{m}][{n}] = {v:.2f}")
print(f"\nC_out row 0: {C_out[0].tolist()}")
print(f"C_out row 2: {C_out[2].tolist()}")
print(f"C_out row 1: {C_out[1].tolist()}")
print(f"C_out row 3: {C_out[3].tolist()}")

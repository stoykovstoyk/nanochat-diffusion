"""Test: A[0][0]=0, A[2][0]=0.5 to isolate broadcast effect."""
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

# Test: A[0][0]=0, A[2][0]=0.5, B all 1.0
A_e2m1 = torch.zeros(M, K, dtype=torch.uint8, device=device)
A_e2m1[0, 0] = 0   # 0.0
A_e2m1[2, 0] = 1   # 0.5
A_packed = A_e2m1.reshape(-1)[::2] | (A_e2m1.reshape(-1)[1::2] << 4)

B_nk = torch.full((N, K), 2, dtype=torch.uint8, device=device)
B_packed = B_nk.reshape(-1)[::2] | (B_nk.reshape(-1)[1::2] << 4)

C_out = torch.zeros(M, N, dtype=torch.float32, device=device)
lib.launch_fp4_mma_gemm(
    ctypes.c_void_p(A_packed.data_ptr()),
    ctypes.c_void_p(B_packed.data_ptr()),
    ctypes.c_void_p(C_out.data_ptr()),
    M, N, K,
)
torch.cuda.synchronize()

print("A[0][0]=0, A[2][0]=0.5, B all 1.0")
print("Expected: C[0][*]=0, C[2][*]=0.5 (if routing correct)")
print()
for m in range(M):
    for n in range(N):
        v = C_out[m,n].item()
        if abs(v) > 0.01:
            print(f"  C[{m}][{n}] = {v:.2f}")

# Also test: all A rows with k=0 set to different values
print("\n--- Row-by-row test ---")
A_full = torch.zeros(M, K, dtype=torch.uint8, device=device)
for row in range(M):
    A_full[row, 0] = 2  # all 1.0 at k=0

A_full_packed = A_full.reshape(-1)[::2] | (A_full.reshape(-1)[1::2] << 4)
C_out2 = torch.zeros(M, N, dtype=torch.float32, device=device)
lib.launch_fp4_mma_gemm(
    ctypes.c_void_p(A_full_packed.data_ptr()),
    ctypes.c_void_p(B_packed.data_ptr()),
    ctypes.c_void_p(C_out2.data_ptr()),
    M, N, K,
)
torch.cuda.synchronize()

print("A[m][0]=1.0 for all m=0..15, B all 1.0")
print("Expected: C[m][n] = 1.0 for all m,n")
for m in range(M):
    print(f"  Row {m:2d}: {C_out2[m].tolist()}")

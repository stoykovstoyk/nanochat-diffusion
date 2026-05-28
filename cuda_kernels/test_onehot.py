"""One-hot test to debug FP4 MMA layout."""
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

# E2M1 encoding: 0->0.0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0

# Test: A has one non-zero value at position (m=0, k=0)
# B has one non-zero value at position (k=0, n=0)
# Expected: C[0][0] = 1.0 * 1.0 = 1.0, all others = 0

# A: MxK row-major, element at (0,0) = 1.0 (E2M1=2), rest = 0 (E2M1=0)
A_e2m1 = torch.zeros(M, K, dtype=torch.uint8, device=device)
A_e2m1[0, 0] = 2  # 1.0

# Pack A row-major: low nibble = even K index
A_flat = A_e2m1.reshape(-1)
A_packed = A_flat[::2] | (A_flat[1::2] << 4)

# B: column-major KxN (packed as NxK row-major)
# B[k=0, n=0] = 1.0 (E2M1=2)
B_nk = torch.zeros(N, K, dtype=torch.uint8, device=device)  # NxK row-major
# Element B_kxn[k=0, n=0] = B_nk[n=0, k=0] = 2
B_nk[0, 0] = 2

# Pack as NxK row-major
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

print("One-hot test: A[0,0]=1.0, B[0,0]=1.0, rest=0")
print("Expected: C[0,0] = 1.0, all others = 0")
print(f"C_out:\n{C_out}")
peak_val = C_out.max().item()
peak_pos = (C_out == peak_val).nonzero(as_tuple=True)
print(f"Peak value: {peak_val:.4f} at position {peak_pos}")
print(f"Sum of all elements: {C_out.sum().item():.4f}")
print(f"Sum should be: 1.0")

# Also check: A[2,5]=1.0, B[5,3]=1.0 -> expected C[2,3] = 1.0
A_e2m1_2 = torch.zeros(M, K, dtype=torch.uint8, device=device)
A_e2m1_2[2, 5] = 2  # 1.0
A_flat_2 = A_e2m1_2.reshape(-1)
A_packed_2 = A_flat_2[::2] | (A_flat_2[1::2] << 4)

# B: column-major. B[k=5, n=3] = 1.0 -> B_nk[n=3, k=5] = 2
B_nk_2 = torch.zeros(N, K, dtype=torch.uint8, device=device)
B_nk_2[3, 5] = 2  # B[5][3] = 1.0
B_flat_2 = B_nk_2.reshape(-1)
B_packed_2 = B_flat_2[::2] | (B_flat_2[1::2] << 4)

C_out_2 = torch.zeros(M, N, dtype=torch.float32, device=device)
lib.launch_fp4_mma_gemm(
    ctypes.c_void_p(A_packed_2.data_ptr()),
    ctypes.c_void_p(B_packed_2.data_ptr()),
    ctypes.c_void_p(C_out_2.data_ptr()),
    M, N, K,
)
torch.cuda.synchronize()

print("\nOne-hot test: A[2,5]=1.0, B[5,3]=1.0")
print("Expected: C[2,3] = 1.0, all others = 0")
peak_val2 = C_out_2.max().item()
peak_pos2 = (C_out_2 == peak_val2).nonzero(as_tuple=True)
print(f"Peak value: {peak_val2:.4f} at position {peak_pos2}")
print(f"Sum of all elements: {C_out_2.sum().item():.4f} (should be 1.0)")
if peak_pos2[0].numel() > 0:
    print(f"  Expected (2,3), got ({peak_pos2[0][0].item()}, {peak_pos2[1][0].item()})")

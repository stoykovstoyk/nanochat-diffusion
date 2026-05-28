"""Minimal FP4 MMA test with constant values."""
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

# E2M1 encoding: 0->0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
# We want A=m*1.0 + n*0.0, B=k*0.0 + n*1.0 (identities)
# Let's make A fill with 1.0 (E2M1=2) and B fill with 1.0 (E2M1=2)
# Expected: C[m][n] = K = 32.0

device = "cuda"

# A: MxK row-major, all 1.0
A_e2m1 = torch.full((M, K), 2, dtype=torch.uint8, device=device)  # 2 = E2M1(1.0)
# Pack: low nibble first, high nibble second
A_flat = A_e2m1.reshape(-1)
A_packed = A_flat[::2] | (A_flat[1::2] << 4)

# B: packed as column-major KxN. Create NxK row-major, all 1.0
B_nk = torch.full((N, K), 2, dtype=torch.uint8, device=device)
B_flat_nk = B_nk.reshape(-1)
B_packed_nk = B_flat_nk[::2] | (B_flat_nk[1::2] << 4)

C_out = torch.zeros(M, N, dtype=torch.float32, device=device)

print("Launching kernel...")
lib.launch_fp4_mma_gemm(
    ctypes.c_void_p(A_packed.data_ptr()),
    ctypes.c_void_p(B_packed_nk.data_ptr()),
    ctypes.c_void_p(C_out.data_ptr()),
    M, N, K,
)

err = torch.cuda.synchronize()
print(f"CUDA error: {err}")

print(f"C_out:\n{C_out}")
print(f"Expected: {K} * 1.0 * 1.0 = {K:.1f}")
print(f"Max diff from expected: {(C_out - K).abs().max().item():.4f}")

# Now try with A=1.0, B=2.0 => expected C = K * 1.0 * 2.0 = 64.0
A_e2m1_2 = torch.full((M, K), 2, dtype=torch.uint8, device=device)
A_packed_2 = A_e2m1_2.reshape(-1)
A_packed_2 = A_packed_2[::2] | (A_packed_2[1::2] << 4)

B_e2m1_4 = torch.full((N, K), 4, dtype=torch.uint8, device=device)  # 4 = E2M1(2.0)
B_packed_4 = B_e2m1_4.reshape(-1)
B_packed_4 = B_packed_4[::2] | (B_packed_4[1::2] << 4)

C_out2 = torch.zeros(M, N, dtype=torch.float32, device=device)
lib.launch_fp4_mma_gemm(
    ctypes.c_void_p(A_packed_2.data_ptr()),
    ctypes.c_void_p(B_packed_4.data_ptr()),
    ctypes.c_void_p(C_out2.data_ptr()),
    M, N, K,
)
torch.cuda.synchronize()
print(f"\nA=1.0, B=2.0")
print(f"C_out[0,:4]: {C_out2[0,:4].tolist()}")
print(f"Expected: {K * 1.0 * 2.0:.1f}")

# Now test with random values but proper quantization
torch.manual_seed(42)
A_rand = torch.randn(M, K, device=device)
B_rand = torch.randn(K, N, device=device)

# Convert to nearest FP4 and pack
fp4_levels = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.], device=device)

def to_e2m1(t):
    """Convert float to E2M1 uint8 encoding."""
    sign = (t < 0).to(torch.uint8)
    pos = t.abs()
    # Find nearest FP4 level
    idx = torch.argmin((pos.unsqueeze(-1) - fp4_levels).abs(), dim=-1).to(torch.uint8)
    return (sign << 3) | idx

A_enc = to_e2m1(A_rand)
A_packed_r = A_enc.reshape(-1)
A_packed_r = A_packed_r[::2] | (A_packed_r[1::2] << 4)

B_t = B_rand.t().contiguous()
B_enc = to_e2m1(B_t)
B_packed_r = B_enc.reshape(-1)
B_packed_r = B_packed_r[::2] | (B_packed_r[1::2] << 4)

# Reference
def decode_fp4(enc):
    """Decode E2M1 to float."""
    sign = (enc >> 3) & 1
    idx = enc & 0x7
    return fp4_levels[idx.long()] * (1 - 2 * sign.float())

A_ref = decode_fp4(A_enc)
B_ref = decode_fp4(B_enc).t()  # was NxK, transpose to KxN
C_ref = A_ref @ B_ref

C_out_r = torch.zeros(M, N, dtype=torch.float32, device=device)
lib.launch_fp4_mma_gemm(
    ctypes.c_void_p(A_packed_r.data_ptr()),
    ctypes.c_void_p(B_packed_r.data_ptr()),
    ctypes.c_void_p(C_out_r.data_ptr()),
    M, N, K,
)
torch.cuda.synchronize()

diff = (C_out_r - C_ref).abs().max().item()
print(f"\nRandom values, max diff vs reference: {diff:.6f}")
print(f"C_out[0,:4]: {C_out_r[0,:4].tolist()}")
print(f"C_ref[0,:4]: {C_ref[0,:4].tolist()}")

"""Test all possible C layouts to find the correct one."""
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

# Recompile kernel with different c_m_off
# Build options for the c_m_off permutations
import subprocess
import os

permutations = [
    [0, 1, 2, 3],  # sequential
    [0, 1, 3, 2],
    [0, 2, 1, 3],  # CuTe SM80
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
]

# Source template
with open("/home/stoyko/Desktop/nanochat-diffusion/cuda_kernels/fp4_mma_gemm.cu") as f:
    src = f.read()

# Read the current CU file and replace c_m_off with a define
src_test = src.replace(
    "static const int c_m_off[4] = {0, 2, 1, 3};",
    "static const int c_m_off[4] = {C0, C1, C2, C3};"
)

# Test data
A_full = torch.zeros(M, K, dtype=torch.uint8, device=device)
for row in range(M):
    A_full[row, 0] = 2  # 1.0 at k=0 for ALL rows
A_packed = A_full.reshape(-1)[::2] | (A_full.reshape(-1)[1::2] << 4)

B_nk = torch.full((N, K), 2, dtype=torch.uint8, device=device)
B_packed = B_nk.reshape(-1)[::2] | (B_nk.reshape(-1)[1::2] << 4)

best = None
best_err = float('inf')

for perm in permutations[:4]:  # Try first 4
    c_src = src_test.replace("{C0, C1, C2, C3}", str(perm).replace('[', '{').replace(']', '}'))
    
    with open("/tmp/test_perm.cu", "w") as f:
        f.write(c_src)
    
    ret = subprocess.run([
        "nvcc", "-gencode=arch=compute_121a,code=sm_121a",
        "-I/tmp/cutlass/include",
        "-O3", "--shared", "-Xcompiler", "-fPIC",
        "-o", "/tmp/libtest_perm.so",
        "/tmp/test_perm.cu"
    ], capture_output=True, text=True)
    
    if ret.returncode != 0:
        print(f"perm {perm}: BUILD FAILED")
        continue
    
    lib2 = ctypes.cdll.LoadLibrary("/tmp/libtest_perm.so")
    lib2.launch_fp4_mma_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib2.launch_fp4_mma_gemm.restype = None
    
    C_out = torch.zeros(M, N, dtype=torch.float32, device=device)
    lib2.launch_fp4_mma_gemm(
        ctypes.c_void_p(A_packed.data_ptr()),
        ctypes.c_void_p(B_packed.data_ptr()),
        ctypes.c_void_p(C_out.data_ptr()),
        M, N, K,
    )
    torch.cuda.synchronize()
    
    # Expected: all 1.0
    expected = torch.ones(M, N, device=device)
    err = (C_out - expected).abs().max().item()
    print(f"perm {perm}: max_err={err:.4f}, rows 0-3={C_out[0:4, 0].tolist()}")
    
    if err < best_err:
        best_err = err
        best = perm

print(f"\nBest: {best} with err={best_err:.4f}")

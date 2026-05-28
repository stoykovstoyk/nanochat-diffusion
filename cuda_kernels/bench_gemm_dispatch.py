import torch
import time
import warnings

device = "cuda"
dtype = torch.bfloat16

# Model shapes from depth=8, n_embd=512, n_head=4, seq=512, bs=16
B, T, C = 16, 512, 512
shapes = [
    ("QKV proj",      B*T, C, 3*C),    # (8192, 512) x (512, 1536)
    ("Attn out",       B*T, C, C),      # (8192, 512) x (512, 512)
    ("MLP gate+up",    B*T, C, 4*C),    # (8192, 512) x (512, 2048)
    ("MLP down",       B*T, 4*C, C),    # (8192, 2048) x (2048, 512)
    ("E2V / lm_head", B*T, C, C),      # (8192, 512) x (512, 32768)
]

import torch.backends.cuda as bc
CUBLAS = 'cublas'
CUBLASLT = 'cublaslt'
DEFAULT = 'default'

def bench_matmul(M, N, K, desc, num_iters=200, warmup=50):
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms = elapsed / num_iters * 1000
    tflops = 2.0 * M * N * K / (elapsed / num_iters) / 1e12
    return ms, tflops

print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name()}")
print(f"Capability: {torch.cuda.get_device_capability()}")
print()

# --- Test 1: Default (cuBLAS on sm_121) ---
print("="*70)
print("Test 1: DEFAULT (cuBLAS on sm_121)")
print("="*70)
bc.preferred_blas_library(DEFAULT)
for _ in range(3):
    torch.mm(torch.randn(1,1,device=device,dtype=dtype), torch.randn(1,1,device=device,dtype=dtype))
for desc, M, K, N in shapes:
    ms, tflops = bench_matmul(M, N, K, desc)
    print(f"  {desc:20s} M={M:5d} N={N:5d} K={K:5d}  {ms:7.3f}ms  {tflops:.1f} TFLOPS")

# --- Test 2: cuBLASLt ---
print()
print("="*70)
print("Test 2: cuBLASLt")
print("="*70)
bc.preferred_blas_library(CUBLASLT)
for _ in range(3):
    torch.mm(torch.randn(1,1,device=device,dtype=dtype), torch.randn(1,1,device=device,dtype=dtype))
for desc, M, K, N in shapes:
    ms, tflops = bench_matmul(M, N, K, desc)
    print(f"  {desc:20s} M={M:5d} N={N:5d} K={K:5d}  {ms:7.3f}ms  {tflops:.1f} TFLOPS")

# --- Test 3: cuBLAS ---
print()
print("="*70)
print("Test 3: cuBLAS")
print("="*70)
bc.preferred_blas_library(CUBLAS)
for _ in range(3):
    torch.mm(torch.randn(1,1,device=device,dtype=dtype), torch.randn(1,1,device=device,dtype=dtype))
for desc, M, K, N in shapes:
    ms, tflops = bench_matmul(M, N, K, desc)
    print(f"  {desc:20s} M={M:5d} N={N:5d} K={K:5d}  {ms:7.3f}ms  {tflops:.1f} TFLOPS")

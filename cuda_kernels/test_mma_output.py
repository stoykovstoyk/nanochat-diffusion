"""Test MMA output with known C values."""
import torch
import ctypes

lib = ctypes.cdll.LoadLibrary("/home/stoyko/Desktop/nanochat-diffusion/cuda_kernels/libtest_mma_output.so")
lib.run_test_mma.argtypes = [ctypes.c_void_p]
lib.run_test_mma.restype = None

out = torch.zeros(32 * 5, dtype=torch.float32, device="cuda")
lib.run_test_mma(ctypes.c_void_p(out.data_ptr()))
torch.cuda.synchronize()

# Parse output
for tid in range(32):
    d = [out[tid * 5 + i].item() for i in range(4)]
    print(f"lane {tid:2d}: D=[{d[0]:8.1f} {d[1]:8.1f} {d[2]:8.1f} {d[3]:8.1f}]", end="")
    # Show how much each D changed from C
    c_base = 100 + tid * 10
    changes = [f"{d[i] - (c_base + i):+.1f}" for i in range(4)]
    print(f"  delta=[{changes[0]} {changes[1]} {changes[2]} {changes[3]}]")

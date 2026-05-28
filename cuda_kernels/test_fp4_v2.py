"""Test FP4 MMA v2 with random values against reference."""
import torch
import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary("/home/stoyko/Desktop/nanochat-diffusion/cuda_kernels/libfp4_mma_v2.so")
lib.launch_fp4_mma_v2.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.launch_fp4_mma_v2.restype = None

def pack_fp4(x):
    """Pack float32 values into FP4 (e2m1) format, row-major MxK."""
    x = x.contiguous().cpu()
    # Round to nearest FP4 value
    fp4_bits = torch.zeros(x.numel(), dtype=torch.uint8)
    for i, val in enumerate(x.flatten()):
        # Find closest FP4 value
        fp4_codes = [0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111,
                     0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111]
        fp4_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 128.0]
        best = min(range(16), key=lambda j: abs(fp4_vals[j] - val))
        fp4_bits[i] = best
    fp4_bits = fp4_bits.reshape(x.shape)
    # Pack: even indices get low nibble, odd get high nibble
    packed = torch.zeros(x.numel() // 2, dtype=torch.uint8)
    packed = (fp4_bits.flatten()[1::2] << 4) | fp4_bits.flatten()[::2]
    return packed.reshape(x.shape[0], x.shape[1] // 2)

def unpack_fp4(packed, shape):
    """Unpack FP4 to float32."""
    flat = torch.zeros(np.prod(shape), dtype=torch.float32)
    pflat = packed.flatten()
    fp4_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 128.0]
    for i in range(len(pflat)):
        byte = pflat[i].item()
        flat[2*i] = fp4_vals[byte & 0xF]
        flat[2*i+1] = fp4_vals[(byte >> 4) & 0xF]
    return flat.reshape(shape)

def ref_gemm(A_f32, B_f32):
    """Reference matmul: C = A @ B.T (since B is KxN column-major)."""
    # A: MxK, B: KxN (column-major => B stored as KxN, so C = A @ B)
    # Actually, B is stored column-major KxN. The matmul C_mn = Σ A_mk * B_kn
    # where B_kn is the k-th row, n-th column of B (B is KxN in memory, and we access B[k][n]).
    # B[k][n] = B[k * N + n] = B_stored[k][n]
    # So C = A @ B (A in MxK, B in KxN)
    return A_f32 @ B_f32

M, N, K = 16, 8, 32
device = "cuda"

torch.manual_seed(42)
A_f32 = (torch.rand(M, K, dtype=torch.float32) * 4).to(device)
B_f32 = (torch.rand(K, N, dtype=torch.float32) * 4).to(device)

C_ref = ref_gemm(A_f32.cpu(), B_f32.cpu())

# Pack to FP4
A_packed = pack_fp4(A_f32.cpu()).to(device)
B_packed = pack_fp4(B_f32.cpu()).to(device)

# Also compute the FP4 reference (dequantized matmul)
A_fp4_f32 = unpack_fp4(A_packed.cpu(), (M, K))
B_fp4_f32 = unpack_fp4(B_packed.cpu(), (K, N))
C_fp4_ref = ref_gemm(A_fp4_f32, B_fp4_f32)

# Run kernel
C = torch.zeros(M, N, dtype=torch.float32, device=device)
lib.launch_fp4_mma_v2(
    ctypes.c_void_p(A_packed.data_ptr()),
    ctypes.c_void_p(B_packed.data_ptr()),
    ctypes.c_void_p(C.data_ptr()),
    M, N, K,
)
torch.cuda.synchronize()

C_kernel = C.cpu()

diff_fp4 = (C_kernel - C_fp4_ref).abs().max().item()
diff_f32 = (C_kernel - C_ref).abs().max().item()

print(f"v2 kernel vs FP4 reference: max diff = {diff_fp4:.6f}")
print(f"v2 kernel vs f32 reference: max diff = {diff_f32:.6f}")
print(f"\nv2 kernel output:\n{C_kernel.numpy().round(4)}")
print(f"\nFP4 reference:\n{C_fp4_ref.numpy().round(4)}")

if diff_fp4 < 0.1:
    print("\n✓ SUCCESS: kernel matches FP4 reference!")
else:
    print(f"\n✗ FAILED: diff = {diff_fp4}")
    # Show positions of largest errors
    err = (C_kernel - C_fp4_ref).abs()
    max_pos = err.argmax().item()
    max_m = max_pos // N
    max_n = max_pos % N
    print(f"  Largest error at C[{max_m}][{max_n}]: "
          f"kernel={C_kernel[max_m,max_n]:.4f}, ref={C_fp4_ref[max_m,max_n]:.4f}")

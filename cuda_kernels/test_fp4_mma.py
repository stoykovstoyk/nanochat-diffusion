"""Test FP4 MMA GEMM kernel correctness vs reference dequantized matmul."""
import torch
import numpy as np
import ctypes
import time

lib = ctypes.cdll.LoadLibrary("/home/stoyko/Desktop/nanochat-diffusion/cuda_kernels/libfp4_mma.so")
lib.launch_fp4_mma_gemm.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K
]
lib.launch_fp4_mma_gemm.restype = None

def pack_fp4(tensor):
    """Convert float32 tensor to packed FP4 (2 values per byte).
    A: row-major MxK, B: column-major KxN
    Returns (packed_tensor, scale).
    """
    # Quantize to FP4 E2M1 (4-bit): range [-6, 6] with 0, 0.5, 1, 1.5, 2, 3, 4, 6
    scale = tensor.abs().max().item() / 6.0
    if scale == 0:
        scale = 1.0
    scaled = tensor / scale
    # Clamp to FP4 representable range
    clamped = scaled.clamp(-6, 6)
    # Convert to uint8: pack as unsigned 4-bit values (E2M1)
    # E2M1 values: 0=0, 1=0.5, 2=1, 3=1.5, 4=2, 5=3, 6=4, 7=6
    # We'll just use round-to-nearest mapping
    fp4_levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=tensor.device)
    # For negative values, use symmetric mapping
    pos = clamped.abs()
    idx = torch.bucketize(pos, fp4_levels) - 1
    idx = idx.clamp(0, 7)
    sign = (clamped < 0).to(torch.uint8)
    # Encode: bit 3 = sign, bits 0-2 = magnitude index
    encoded = (sign << 3) | idx.to(torch.uint8)
    # Pack: low nibble = first element, high nibble = second
    flat = encoded.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, flat[:1]])  # pad
    packed = flat[::2] | (flat[1::2] << 4)
    return packed.contiguous(), scale

def dequantize_fp4(packed, scale, shape):
    """Unpack FP4 back to float32."""
    flat = torch.zeros(shape[0] * shape[1], dtype=torch.uint8, device=packed.device)
    flat[::2] = packed & 0xF
    flat[1::2] = (packed >> 4) & 0xF
    encoded = flat[:shape[0] * shape[1]].reshape(shape)
    sign = (encoded >> 3) & 1
    mag_idx = encoded & 0x7
    fp4_levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    value = fp4_levels[mag_idx.long()]
    value = value * (1 - 2 * sign.float())
    return value * scale

def nearest_fp4(t):
    """Convert float to FP4 and back."""
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=t.device)
    neg = t < 0
    pos = t.abs()
    idx = torch.bucketize(pos, levels) - 1
    idx = idx.clamp(0, 7)
    result = levels[idx] * (1 - 2 * neg.float())
    return result

def test():
    device = torch.device("cuda")
    
    M, N, K = 16, 8, 32  # Single tile
    
    # Reference: standard bf16 matmul
    A_f32 = torch.randn(M, K, device=device)
    B_f32 = torch.randn(K, N, device=device)
    
    # FP4 quantize
    A_packed, scale_A = pack_fp4(A_f32)
    A_q = dequantize_fp4(A_packed, scale_A, (M, K))
    
    # For B: physically stored as column-major KxN
    # Equivalent to transposing to NxK row-major and packing that
    B_f32_t = B_f32.t().contiguous()  # shape (N, K), row-major = column-major KxN
    B_packed, scale_B = pack_fp4(B_f32_t)
    # Dequantize back to KxN column-major view
    B_q = dequantize_fp4(B_packed, scale_B, (N, K)).t()
    
    # Reference: FP4 dequant matmul
    C_ref = A_q @ B_q
    
    # Our kernel
    C_out = torch.zeros(M, N, dtype=torch.float32, device=device)
    
    lib.launch_fp4_mma_gemm(
        ctypes.c_void_p(A_packed.data_ptr()),
        ctypes.c_void_p(B_packed.data_ptr()),
        ctypes.c_void_p(C_out.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    
    # Compare
    diff = (C_out - C_ref).abs().max().item()
    print(f"M={M} N={N} K={K}")
    print(f"  Max diff vs FP4 dequant reference: {diff:.6f}")
    print(f"  C_out[0,:4]: {C_out[0,:4].tolist()}")
    print(f"  C_ref[0,:4]: {C_ref[0,:4].tolist()}")
    print(f"  C_out norm: {C_out.norm().item():.4f}")
    print(f"  C_ref norm: {C_ref.norm().item():.4f}")
    
    # Test multiple tiles
    M2, N2, K2 = 32, 16, 64
    A2_f32 = torch.randn(M2, K2, device=device)
    B2_f32 = torch.randn(K2, N2, device=device)
    
    A2_packed, sA2 = pack_fp4(A2_f32)
    A2_q = dequantize_fp4(A2_packed, sA2, (M2, K2))
    B2_f32_t = B2_f32.t().contiguous()
    B2_packed, sB2 = pack_fp4(B2_f32_t)
    B2_q = dequantize_fp4(B2_packed, sB2, (N2, K2)).t()
    
    C2_ref = A2_q @ B2_q
    C2_out = torch.zeros(M2, N2, dtype=torch.float32, device=device)
    
    lib.launch_fp4_mma_gemm(
        ctypes.c_void_p(A2_packed.data_ptr()),
        ctypes.c_void_p(B2_packed.data_ptr()),
        ctypes.c_void_p(C2_out.data_ptr()),
        ctypes.c_int(M2),
        ctypes.c_int(N2),
        ctypes.c_int(K2),
    )
    
    diff2 = (C2_out - C2_ref).abs().max().item()
    print(f"\nM={M2} N={N2} K={K2}")
    print(f"  Max diff vs FP4 dequant reference: {diff2:.6f}")
    print(f"  C2_out[0,:4]: {C2_out[0,:4].tolist()}")
    print(f"  C2_ref[0,:4]: {C2_ref[0,:4].tolist()}")
    
    if diff < 0.1 and diff2 < 0.1:
        print("\nSUCCESS: FP4 MMA GEMM matches reference!")
    else:
        print(f"\nWARNING: Large diff detected. diff={diff:.6f} diff2={diff2:.6f}")
    
    # Benchmark using the already-packed data
    C_bm = torch.zeros(M2, N2, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(10):
        lib.launch_fp4_mma_gemm(
            ctypes.c_void_p(A2_packed.data_ptr()),
            ctypes.c_void_p(B2_packed.data_ptr()),
            ctypes.c_void_p(C_bm.data_ptr()),
            ctypes.c_int(M2),
            ctypes.c_int(N2),
            ctypes.c_int(K2),
        )
    torch.cuda.synchronize()
    
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        lib.launch_fp4_mma_gemm(
            ctypes.c_void_p(A2_packed.data_ptr()),
            ctypes.c_void_p(B2_packed.data_ptr()),
            ctypes.c_void_p(C_bm.data_ptr()),
            ctypes.c_int(M2),
            ctypes.c_int(N2),
            ctypes.c_int(K2),
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    print(f"\nBenchmark ({n_iters} iters): {elapsed:.2f} ms per call ({M2}x{N2}x{K2})")

if __name__ == "__main__":
    test()

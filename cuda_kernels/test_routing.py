"""Brute-force determine SM120 FP4 MMA routing: 
For each A byte_off and each B byte_off, which D register gets A*B?"""
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
FP4_VAL = 2  # 1.0 in FP4

# Pre-encode the CuTe ALayout's byte_off -> (thr_v, thr_m, byte_within_thread, k1, k0, v)
# Based on: flat = thr_v*64 + thr_m + v*16 + k0*8 + k1*256
A_byte_table = []
for thr_v in range(4):
    for thr_m in range(8):
        for k1 in range(2):
            for k0 in range(2):
                for v in range(4):
                    flat = thr_v * 64 + thr_m + v * 16 + k0 * 8 + k1 * 256
                    m = flat // 32
                    k = flat % 32
                    byte_off = v + k0 * 4 + k1 * 8
                    A_byte_table.append((thr_v, thr_m, byte_off, m, k))

B_byte_table = []
for thr_v in range(4):
    for thr_m in range(8):
        for k_val in range(2):
            for n_val in range(4):
                flat = thr_v * 32 + thr_m + n_val * 8 + k_val * 128
                n = flat // 32
                k = flat % 32
                byte_off = n_val + k_val * 4
                B_byte_table.append((thr_v, thr_m, byte_off, k, n))

# For a single thread (thr_v=0, thr_m=0), build routing table
# For each pair (A_byte_off, B_byte_off), find which D output has A*B contribution
print("Testing byte routing for thr_v=0, thr_m=0")
print("Testing all 16 A byte positions × all 8 B byte positions...")
print()

# We'll test by setting ONE byte in A and ONE byte in B, and checking which D[i] is non-zero
# D outputs are C[thr_v*4 + c_off][thr_m] where c_off = {0,2,1,3}
c_m_off = [0, 2, 1, 3]

# Build routing as: routing[5][thr_m] = dict mapping (a_byte_off, b_byte_off, k_match) -> d_idx
# Actually simpler: just test empirically

results = {}
for a_bo in range(16):
    for b_bo in range(8):
        # Build A with only a_bo set
        A = torch.zeros(M, K, dtype=torch.uint8, device=device)
        # Find which (m,k) this byte_off corresponds to for thr_v=0, thr_m=0
        a_match = [(tv,tm,bo,m,k) for tv,tm,bo,m,k in A_byte_table if tv==0 and tm==0 and bo==a_bo]
        if not a_match: continue
        m_a, k_a = a_match[0][3], a_match[0][4]
        A[m_a, k_a] = FP4_VAL
        A_packed = A.reshape(-1)[::2] | (A.reshape(-1)[1::2] << 4)
        
        # Build B with only b_bo set
        B = torch.zeros(K, N, dtype=torch.uint8, device=device)
        b_match = [(tv,tm,bo,k,n) for tv,tm,bo,k,n in B_byte_table if tv==0 and tm==0 and bo==b_bo]
        if not b_match: continue
        k_b, n_b = b_match[0][3], b_match[0][4]
        B[k_b, n_b] = FP4_VAL
        B_packed = B.reshape(-1)[::2] | (B.reshape(-1)[1::2] << 4)
        
        C = torch.zeros(M, N, dtype=torch.float32, device=device)
        lib.launch_fp4_mma_gemm(
            ctypes.c_void_p(A_packed.data_ptr()),
            ctypes.c_void_p(B_packed.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            M, N, K,
        )
        torch.cuda.synchronize()
        
        # Check which D registers have the contribution for thr_v=0
        for d_idx, c_off in enumerate(c_m_off):
            for tm in range(8):
                val = C[thr_v*4 + c_off, tm].item()
                if val > 0.5:  # We set 1.0, so product = 1.0
                    key = (a_bo, b_bo, tm, d_idx)
                    results[key] = results.get(key, 0) + 1

# Print routing map
print(f"Found {len(results)} A*B -> D routing entries")
print("\nFor each A byte_off (0..15) and B byte_off (0..7), which D[i] gets output?")
print("Format: A_bo B_bo -> D[d_idx] at C[thr_v*4+c_off][tm=col]")
print()

# Summarize: for each A_bo, which D indices does it contribute to (across all B_bo)?
print("=== A byte_off -> D output mapping ===")
for a_bo in range(16):
    d_set = set()
    for (a_bo2, b_bo, tm, d_idx), cnt in results.items():
        if a_bo2 == a_bo:
            d_set.add(d_idx)
    if d_set:
        a_info = [(tv,tm,bo,m,k) for tv,tm,bo,m,k in A_byte_table if tv==0 and tm==0 and bo==a_bo]
        m,k = a_info[0][3], a_info[0][4]
        print(f"A_bo={a_bo:2d} (m={m:2d},k={k:2d}) -> D indices {sorted(d_set)} [C m_offs: {[c_m_off[d] for d in sorted(d_set)]}]")

print("\n=== B byte_off -> D output mapping ===")
for b_bo in range(8):
    d_set = set()
    for (a_bo, b_bo2, tm, d_idx), cnt in results.items():
        if b_bo2 == b_bo:
            d_set.add(d_idx)
    if d_set:
        b_info = [(tv,tm,bo,k,n) for tv,tm,bo,k,n in B_byte_table if tv==0 and tm==0 and bo==b_bo]
        k,n = b_info[0][3], b_info[0][4]
        print(f"B_bo={b_bo:2d} (k={k:2d},n={n}) -> D indices {sorted(d_set)}")

# Check: do A bytes with k0=0 (0,1,2,3) map to D[0],D[1] and k0=1 (4,5,6,7) to D[2],D[3]?
print("\n=== Pattern analysis ===")
for k0_test in [0, 1]:
    d_set = set()
    a_bos = [a_bo for a_bo in range(16) 
             if any(tv==0 and tm==0 and a_bo==bo and k0==k0_test 
                    for tv,tm,bo,m,k in A_byte_table 
                    for k1 in range(2) for v in range(4)
                    if (bo:=(v + k0_test*4 + k1*8)) == a_bo)]
    # Simplier: just enumerate
    for a_bo in range(16):
        for (a_bo2, b_bo, tm, d_idx), cnt in results.items():
            if a_bo2 == a_bo:
                a_info = [x for x in A_byte_table if x[0]==0 and x[1]==0 and x[2]==a_bo]
                if a_info and a_info[0][1] == k0_test:  # k0 = 
                    pass
                
print("\nD[0] routes:")
print("A bytes:", [a_bo for a_bo in range(16) if any(ab==a_bo for ab,_,_,d in results if d==0)])

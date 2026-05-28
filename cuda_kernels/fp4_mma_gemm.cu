#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm120.hpp>

using namespace cute;

// FP4 MMA GEMM for SM121a
// C[M,N] += A[M,K] * B[K,N] where A,B are FP4 E2M1, C is float32
// Dimensions must be multiples of the tile size (16/8/32).
// A: row-major MxK, FP4 packed 2 per byte
// B: column-major KxN, FP4 packed 2 per byte

__global__ void fp4_mma_gemm_kernel(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K,
    int stride_A,
    int stride_B
) {
    int thr_v = threadIdx.x / 8;
    int thr_m = threadIdx.x % 8;
    int base_m = blockIdx.y * 16;
    int base_n = blockIdx.x * 8;

    // ---- Load A: ALayout Shape<4,2,2> (col-major register) ----
    // Value layout: shape <_4,_2,_2> col-major
    //   linear value index = v + k0*4 + k1*8
    // Logical flat = thr_v*64 + thr_m + v*16 + k0*8 + k1*256
    // For M=16,K=32 row-major: flat = m*32 + k
    //   m = flat / 32, k = flat % 32

    unsigned char frag_A[16];
    for (int k1 = 0; k1 < 2; k1++) {
        for (int k0 = 0; k0 < 2; k0++) {
            for (int v = 0; v < 4; v++) {
                int flat = thr_v * 64 + thr_m + v * 16 + k0 * 8 + k1 * 256;
                int m = flat / 32;
                int k = flat % 32;

                int global_m = base_m + m;
                int pk = (global_m * stride_A + k) / 2;
                int nibble = (global_m * stride_A + k) & 1;
                unsigned char byte = A_packed[pk];
                unsigned char fp4_bits = nibble ? (byte >> 4) : (byte & 0xF);

                // Register byte: column-major order for shape <4,2,2>
                int byte_off = v + k0 * 4 + k1 * 8;
                frag_A[byte_off] = fp4_bits << 2;
            }
        }
    }

    uint32_t reg_A[4];
    memcpy(reg_A, frag_A, 16);

    // ---- Load B: BLayout Shape<4,2> (col-major register) ----
    // Value layout: shape <_4,_2> col-major
    //   linear value index = v + k_val*4
    // Logical flat = thr_v*32 + thr_m + v*8 + k_val*128
    // For column-major KxN: flat = k + n*32
    //   n = flat / 32, k = flat % 32

    unsigned char frag_B[8];
    for (int k_val = 0; k_val < 2; k_val++) {
        for (int v = 0; v < 4; v++) {
            int n = thr_v + k_val * 4;
            int k = thr_m + v * 8;

            int global_n = base_n + n;
            int pk = (k + global_n * stride_B) / 2;
            int nibble = (k + global_n * stride_B) & 1;
            unsigned char byte = B_packed[pk];
            unsigned char fp4_bits = nibble ? (byte >> 4) : (byte & 0xF);

            int byte_off = v + k_val * 4;
            frag_B[byte_off] = fp4_bits << 2;
        }
    }

    uint32_t reg_B[2];
    memcpy(reg_B, frag_B, 8);

    // ---- Load C: CLayout SM80_16x8_Row ----
    // Value shape <_2,_2> col-major: linear index = m_frag + n_frag*2
    // Logical: m = thr_v*4 + m_frag*2 + n_frag, n = thr_m
    //   i | m_frag | n_frag | m_offset
    //   0 | 0      | 0      | 0
    //   1 | 1      | 0      | 2
    //   2 | 0      | 1      | 1
    //   3 | 1      | 1      | 3
    static const int c_m_off[4] = {0, 2, 1, 3};

    float reg_C[4];
    for (int i = 0; i < 4; i++) {
        int m_idx = base_m + thr_v * 4 + c_m_off[i];
        int n_idx = base_n + thr_m;
        reg_C[i] = C[m_idx * N + n_idx];
    }

    // ---- Execute MMA ----
    float reg_D[4];

    using MmaOp = SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>;

    MmaOp::fma(
        reg_D[0], reg_D[1], reg_D[2], reg_D[3],
        reg_A[0], reg_A[1], reg_A[2], reg_A[3],
        reg_B[0], reg_B[1],
        reg_C[0], reg_C[1], reg_C[2], reg_C[3]
    );

    // ---- Store D (same register order as C) ----
    for (int i = 0; i < 4; i++) {
        int m_idx = base_m + thr_v * 4 + c_m_off[i];
        int n_idx = base_n + thr_m;
        C[m_idx * N + n_idx] = reg_D[i];
    }
}

extern "C" {

void launch_fp4_mma_gemm(
    const unsigned char* A,
    const unsigned char* B,
    float* C,
    int M, int N, int K
) {
    dim3 block(32);
    dim3 grid((N + 7) / 8, (M + 15) / 16);

    fp4_mma_gemm_kernel<<<grid, block>>>(
        A, B, C, M, N, K, K, K
    );
}

}

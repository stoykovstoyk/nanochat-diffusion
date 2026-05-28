// CuTe-based FP4 MMA GEMM using CuTe's tensor operations
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm120.hpp>

using namespace cute;

__global__ void fp4_mma_cute(
    uint8_t const* __restrict__ A,  // MxK row-major, FP4 packed
    uint8_t const* __restrict__ B,  // KxN column-major, FP4 packed
    float* __restrict__ C,           // MxN output
    int M, int N, int K
) {
    int tid = threadIdx.x;
    int thr_v = tid / 8;
    int thr_m = tid % 8;
    int base_m = blockIdx.y * 16;
    int base_n = blockIdx.x * 8;

    // Use CuTe's MMA traits
    using MMA = SM120_16x8x32_TN<__nv_fp4_e2m1>;
    using ValTypeA = typename MMA::ValTypeA;  // uint8_t
    using ValTypeB = typename MMA::ValTypeB;  // uint8_t

    using ALayout = typename MMA::ALayout;
    using BLayout = typename MMA::BLayout;
    using CLayout = typename MMA::CLayout;

    // A fragment
    uint32_t reg_A[4] = {0, 0, 0, 0};

    // Load A using CuTE layout
    int logical_flat_base = thr_v * 64 + thr_m;

    // Iterate over tiles of K
    for (int k_tile = 0; k_tile < K; k_tile += 32) {
        // Load A fragment
        for (int k1 = 0; k1 < 2; ++k1) {
            for (int k0 = 0; k0 < 2; ++k0) {
                for (int v = 0; v < 4; ++v) {
                    int logical_flat = logical_flat_base + v * 16 + k0 * 8 + k1 * 256;
                    int m = logical_flat / 32;
                    int k = logical_flat % 32;
                    int global_m = base_m + m;
                    int global_k = k_tile + k;

                    // Read packed FP4 byte
                    int linear = global_m * (K/2) + global_k / 2;
                    int nibble = global_k & 1;
                    uint8_t byte = A[linear];
                    uint8_t fp4_val = nibble ? (byte >> 4) : (byte & 0xF);

                    int byte_off = v + k0 * 4 + k1 * 8;
                    ((uint8_t*)reg_A)[byte_off] = fp4_val << 2;
                }
            }
        }

        // Load B fragment
        uint32_t reg_B[2] = {0, 0};
        for (int k_val = 0; k_val < 2; ++k_val) {
            for (int v = 0; v < 4; ++v) {
                int n = thr_v + k_val * 4;
                int k = thr_m + v * 8;
                int global_n = base_n + n;
                int global_k = k_tile + k;

                int linear = global_k * (N/2) + global_n / 2;
                int nibble = global_n & 1;
                uint8_t byte = B[linear];
                uint8_t fp4_val = nibble ? (byte >> 4) : (byte & 0xF);

                int byte_off = v + k_val * 4;
                ((uint8_t*)reg_B)[byte_off] = fp4_val << 2;
            }
        }

        // Load C
        float reg_C[4], reg_D[4];
        for (int i = 0; i < 4; ++i) {
            int m_idx = base_m + thr_v * 4 + i;
            int n_idx = base_n + thr_m;
            // Handle C layout for FP4
            // SM80 CLayout: D[0]=C[thr_v*4+0][n], D[1]=C[thr_v*4+2][n],
            //              D[2]=C[thr_v*4+1][n], D[3]=C[thr_v*4+3][n]
            static const int c_off[4] = {0, 2, 1, 3};
            reg_C[i] = C[m_idx * N + n_idx];
        }

        // MMA call
        asm volatile(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(reg_D[0]), "=f"(reg_D[1]), "=f"(reg_D[2]), "=f"(reg_D[3])
            :  "r"(reg_A[0]),  "r"(reg_A[1]),  "r"(reg_A[2]),  "r"(reg_A[3]),
               "r"(reg_B[0]),  "r"(reg_B[1]),
               "f"(reg_C[0]),  "f"(reg_C[1]),  "f"(reg_C[2]),  "f"(reg_C[3]));

        // Write back C
        for (int i = 0; i < 4; ++i) {
            int m_idx = base_m + thr_v * 4 + i;
            int n_idx = base_n + thr_m;
            static const int c_off[4] = {0, 2, 1, 3};
            C[m_idx * N + n_idx] = reg_D[c_off[i]];
        }
    }
}

extern "C" {
void launch_fp4_mma_cute(
    const uint8_t* A, const uint8_t* B,
    float* C, int M, int N, int K
) {
    dim3 block(32);
    dim3 grid((N + 7) / 8, (M + 15) / 16);
    fp4_mma_cute<<<grid, block>>>(A, B, C, M, N, K);
}
}

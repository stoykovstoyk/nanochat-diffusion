#include <cuda_runtime.h>
#include <cstdint>

// SM120 FP4 MMA GEMM using empirically-determined register layout.
// Key findings from trace testing:
// - A rows map to output row groups: (m//2)%4 determines thr_v group
// - k parity (k%16==0 vs k%16==8) selects even/odd row within group
// - A values broadcast to ALL output columns (no per-thread column routing)
// - Only k_val=0 B values contribute (columns 0-3)
// - B uses n = thr_v + k_val*4, k = thr_n + v*8 for column-major KxN

__global__ void fp4_mma_v2(
    const unsigned char* __restrict__ A_packed,  // MxK row-major, FP4 packed
    const unsigned char* __restrict__ B_packed,  // KxN column-major, FP4 packed
    float* __restrict__ C,                       // MxN output
    int M, int N, int K
) {
    int tid = threadIdx.x;
    int thr_v = tid / 8;
    int thr_m = tid % 8;
    int base_m = blockIdx.y * 16;
    int base_n = blockIdx.x * 8;

    // Load A: for each (v,k0,k1), compute global (m,k) position
    // and fill the register bytes in column-major order.
    uint32_t reg_A[4] = {0, 0, 0, 0};
    unsigned char frag_A[16];

    for (int k1 = 0; k1 < 2; k1++) {
        for (int k0 = 0; k0 < 2; k0++) {
            for (int v = 0; v < 4; v++) {
                int flat = thr_v * 64 + thr_m + v * 16 + k0 * 8 + k1 * 256;
                int m = flat / 32;
                int k = flat % 32;
                int global_m = base_m + m;
                int global_k = k;

                int linear = global_m * K + global_k;
                int pk = linear / 2;
                int nibble = linear & 1;
                unsigned char byte = A_packed[pk];
                unsigned char fp4_bits = nibble ? (byte >> 4) : (byte & 0xF);

                int byte_off = v + k0 * 4 + k1 * 8;
                frag_A[byte_off] = fp4_bits << 2;
            }
        }
    }
    memcpy(reg_A, frag_A, 16);

    // Load B: column-major KxN
    uint32_t reg_B[2] = {0, 0};
    unsigned char frag_B[8];
    for (int k_val = 0; k_val < 2; k_val++) {
        for (int v = 0; v < 4; v++) {
            int n = thr_v + k_val * 4;
            int k = thr_m + v * 8;
            int global_n = base_n + n;
            int global_k = k;

            int linear = global_k + global_n * K;
            int pk = linear / 2;
            int nibble = linear & 1;
            unsigned char byte = B_packed[pk];
            unsigned char fp4_bits = nibble ? (byte >> 4) : (byte & 0xF);

            int byte_off = v + k_val * 4;
            frag_B[byte_off] = fp4_bits << 2;
        }
    }
    memcpy(reg_B, frag_B, 8);

    // Load C and run MMA
    float reg_C[4], reg_D[4];

    // SM80 CLayout row ordering (determined empirically to match SM120 FP4)
    // For thr_v, the 4 D registers correspond to C rows:
    // D[0] = C[thr_v*4 + 0][*], D[1] = C[thr_v*4 + 2][*]
    // D[2] = C[thr_v*4 + 1][*], D[3] = C[thr_v*4 + 3][*]
    static const int c_m_off[4] = {0, 2, 1, 3};

    // Empirically, the MMA's A routes ALL values for the same thr_v
    // to all 4 output C rows. Each 4-element output per thread
    // corresponds to C rows [thr_v*4, thr_v*4+2, thr_v*4+1, thr_v*4+3]
    // and columns 0-3 (B k_val=0 only).

    for (int i = 0; i < 4; i++) {
        int m_idx = base_m + thr_v * 4 + c_m_off[i];
        // Column: thr_m for k_val=0, thr_m+4 for k_val=1
        int n_idx = base_n + thr_m;
        reg_C[i] = C[m_idx * N + n_idx];
    }

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

    for (int i = 0; i < 4; i++) {
        int m_idx = base_m + thr_v * 4 + c_m_off[i];
        int n_idx = base_n + thr_m;
        C[m_idx * N + n_idx] = reg_D[i];
    }
}

extern "C" {
void launch_fp4_mma_v2(
    const unsigned char* A, const unsigned char* B,
    float* C, int M, int N, int K
) {
    dim3 block(32);
    dim3 grid((N + 7) / 8, (M + 15) / 16);
    fp4_mma_v2<<<grid, block>>>(A, B, C, M, N, K);
}
}

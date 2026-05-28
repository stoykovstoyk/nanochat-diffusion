#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

// Direct PTX MMA test with manually controlled register values
__global__ void direct_mma_test(float* out, int test_case) {
    int tid = threadIdx.x;
    int thr_v = tid / 8;
    int thr_m = tid % 8;

    // A regs: 4 x uint32_t (128 bits = 16 FP4 values)
    uint32_t reg_A[4] = {0, 0, 0, 0};
    // B regs: 2 x uint32_t (64 bits = 8 FP4 values)
    uint32_t reg_B[2] = {0, 0};

    // Set specific values based on test_case
    if (test_case == 0) {
        // Test 0: All 1.0 (E2M1 encoding 2, left-shifted by 2 = 8)
        // Each byte = 0x88 (both nibbles = 2, << 2 = 8)
        // A and B all 1.0: expected output = 32.0 for each thread
        for (int i = 0; i < 4; i++) reg_A[i] = 0x08080808;
        for (int i = 0; i < 2; i++) reg_B[i] = 0x08080808;
    }
    else if (test_case == 1) {
        // Test 1: Only lane 0, byte 0 of A = 1.0, B all 1.0
        if (tid == 0) {
            reg_A[0] = 0x08000000;  // byte 0 = 8, rest = 0
        }
        if (tid < 4) {
            // thr_v=0, thr_m=tid
            // For these threads, byte 0 of B = 1.0, so B[thr_m][n] = 1.0
            // Wait, B is consumed as column-major KxN.
            // Actually for the MMA, B's layout determines which thread has which B[k][n].
            // For the MMA TN, B's "col" means column-major.
        }
        // Set B all 1.0
        for (int i = 0; i < 2; i++) reg_B[i] = 0x08080808;
    }
    else if (test_case == 2) {
        // Test 2: A and B filling with specific patterns
        // Lane 0 gets A[0][0..3] and B[0][0..7] (well, something like that)
        // Let's just set everything to 1.0 and see
        for (int i = 0; i < 4; i++) reg_A[i] = 0x08080808;
        for (int i = 0; i < 2; i++) reg_B[i] = 0x08080808;
    }

    float reg_C[4] = {0, 0, 0, 0};
    float reg_D[4];

    // MMA via inline PTX
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

    // Store output
    out[tid * 4 + 0] = reg_D[0];
    out[tid * 4 + 1] = reg_D[1];
    out[tid * 4 + 2] = reg_D[2];
    out[tid * 4 + 3] = reg_D[3];
}

extern "C" {
void run_direct_test(float* out, int test_case) {
    direct_mma_test<<<1, 32>>>(out, test_case);
}
}

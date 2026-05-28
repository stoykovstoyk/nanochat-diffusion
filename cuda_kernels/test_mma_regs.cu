#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

// Test MMA with known register values per thread
// Each thread sets its own regs and checks output
__global__ void test_mma_regs(float* out, uint32_t a_val, uint32_t b_val) {
    uint32_t reg_A[4] = {a_val, a_val, a_val, a_val};
    uint32_t reg_B[2] = {b_val, b_val};
    float reg_C[4] = {0, 0, 0, 0};
    float reg_D[4];

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

    out[threadIdx.x * 5 + 0] = reg_D[0];
    out[threadIdx.x * 5 + 1] = reg_D[1];
    out[threadIdx.x * 5 + 2] = reg_D[2];
    out[threadIdx.x * 5 + 3] = reg_D[3];
    out[threadIdx.x * 5 + 4] = threadIdx.x;
}

extern "C" {
void run_mma_regs(float* out, uint32_t a_val, uint32_t b_val) {
    test_mma_regs<<<1, 32>>>(out, a_val, b_val);
}
}

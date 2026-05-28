#include <cuda_runtime.h>
#include <cstdint>

// Test SM120 FP4 MMA: set C values and see which D outputs get them
__global__ void test_mma_output(float* d_out) {
    uint32_t reg_A[4] = {0x08080808, 0x08080808, 0x08080808, 0x08080808};
    uint32_t reg_B[2] = {0x08080808, 0x08080808};
    float reg_C[4];
    float reg_D[4];

    // Set different C initial values for each thread
    reg_C[0] = 100.0f + threadIdx.x * 10.0f;
    reg_C[1] = 101.0f + threadIdx.x * 10.0f;
    reg_C[2] = 102.0f + threadIdx.x * 10.0f;
    reg_C[3] = 103.0f + threadIdx.x * 10.0f;

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

    d_out[threadIdx.x * 5 + 0] = reg_D[0];
    d_out[threadIdx.x * 5 + 1] = reg_D[1];
    d_out[threadIdx.x * 5 + 2] = reg_D[2];
    d_out[threadIdx.x * 5 + 3] = reg_D[3];
    d_out[threadIdx.x * 5 + 4] = threadIdx.x;
}

extern "C" {
void run_test_mma(float* d_out) {
    test_mma_output<<<1, 32>>>(d_out);
}
}

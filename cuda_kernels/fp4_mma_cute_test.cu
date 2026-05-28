// Minimal CuTe-based test: use CuTe's copy+mma to see if SM120 FP4 works correctly
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cstdio>

using namespace cute;

// Direct CuTe test: load A and B using tensor views and CuTe copy, then MMA
__global__ void test_cute_fp4(
    float* C_out,  // output C
    int test_case
) {
    using MMA = SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>;
    
    // Fragment tensors in registers (using CuTe's native types)
    Tensor mA = make_tensor_like<uint8_t>(make_shape(Int<16>{}, Int<32>{})); // won't work
  
    // Manual approach: load data using the CuTe ALayout mapping
    // First, let's just set values directly to test
    int tid = threadIdx.x;
    int thr_v = tid / 8;
    int thr_m = tid % 8;
    
    if (threadIdx.x == 0) {
        printf("Testing SM120 FP4 MMA with direct register setup\n");
        printf("SM architecture: %d\n", cuda_arch());
    }
}

// Second attempt: load from shared memory using CuTe Copy
__global__ void test_cute_fp4_smem(
    uint8_t const* A_packed,  // M x K/2 row-major packed FP4
    uint8_t const* B_packed,  // K x N/2 col-major packed FP4  
    float* C_out,              // M x N output
    int M, int N, int K
) {
    int tid = threadIdx.x;
    int thr_v = tid / 8;
    int thr_m = tid % 8;
    int base_m = blockIdx.y * 16;
    int base_n = blockIdx.x * 8;
    
    // Use CuTe's tensor operations to load A and B
    // First, create gmem tensors
    auto gA = make_tensor(make_gmem_ptr(A_packed), make_layout(make_shape(M, K/2)));
    auto gB = make_tensor(make_gmem_ptr(B_packed), make_layout(make_shape(K, N/2)));
    
    // Create an A fragment using the CuTe ALayout
    using ALayout = typename MMA_Traits<SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>>::ALayout;
    using BLayout = typename MMA_Traits<SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>>::BLayout;
    using CLayout = typename MMA_Traits<SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>>::CLayout;
    
    // Construct thread-level tensors using the layouts
    // For A: shape is (16, 32) logical, but we pack FP4
    // CuTe's A fragment: 16 uint8_t values = the FP4 values before packing
    
    // Actually, CuTe expects raw FP4 values (4-bit shifted to bits 2-5)
    // The packing in memory is different - each byte in memory holds 2 FP4 values
    
    // Let me just manually implement this using the known layout
    // and check if values are correct
    
    // A fragment (per thread): 16 x uint8_t = 4 x uint32_t
    uint32_t reg_A[4] = {0};
    uint32_t reg_B[2] = {0};
    
    // Load A using the ALayout mapping (same as before)
    for (int k1 = 0; k1 < 2; ++k1) {
        for (int k0 = 0; k0 < 2; ++k0) {
            for (int v = 0; v < 4; ++v) {
                int flat = thr_v * 64 + thr_m + v * 16 + k0 * 8 + k1 * 256;
                int m = flat / 32;
                int k = flat % 32;
                int global_m = base_m + m;
                int global_k = k;
                
                // Read from packed FP4 memory
                int linear = global_m * (K/2) + global_k / 2;
                int nibble = global_k & 1;
                uint8_t byte = A_packed[linear];
                uint8_t fp4_bits = nibble ? (byte >> 4) : (byte & 0xF);
                
                int byte_off = v + k0 * 4 + k1 * 8;
                ((uint8_t*)reg_A)[byte_off] = fp4_bits << 2;
            }
        }
    }
    
    // Load B using BLayout mapping
    for (int k_val = 0; k_val < 2; ++k_val) {
        for (int n_val = 0; n_val < 4; ++n_val) {
            int n = thr_v + k_val * 4;
            int k = thr_m + n_val * 8;
            int global_n = base_n + n;
            int global_k = k;
            
            // B_packed is K x N/2 column-major
            // linear = k * (N/2) + n/2
            int linear = global_k * (N/2) + global_n / 2;
            int nibble = global_n & 1;
            uint8_t byte = B_packed[linear];
            uint8_t fp4_bits = nibble ? (byte >> 4) : (byte & 0xF);
            
            int byte_off = n_val + k_val * 4;
            ((uint8_t*)reg_B)[byte_off] = fp4_bits << 2;
        }
    }
    
    // Load C
    float reg_C[4], reg_D[4];
    for (int i = 0; i < 4; ++i) {
        int m_idx = base_m + thr_v * 4 + i;
        int n_idx = base_n + thr_m;
        reg_C[i] = C_out[m_idx * N + n_idx];
    }
    
    // MMA
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
    
    // Write back via CLayout 
    // SM80_16x8_Row: D[i] = C[thr_v*4 + i][thr_m]
    // But hardware outputs in order: %0->m0, %1->m2, %2->m1, %3->m3
    static const int c_off_swap[4] = {0, 2, 1, 3};
    for (int i = 0; i < 4; ++i) {
        int m_idx = base_m + thr_v * 4 + c_off_swap[i];
        int n_idx = base_n + thr_m;
        C_out[m_idx * N + n_idx] = reg_D[i];
    }
}

extern "C"
void launch_cute_fp4_test(
    const uint8_t* A, const uint8_t* B,
    float* C, int M, int N, int K
) {
    dim3 block(32);
    dim3 grid((N + 7) / 8, (M + 15) / 16);
    test_cute_fp4_smem<<<grid, block>>>(A, B, C, M, N, K);
}

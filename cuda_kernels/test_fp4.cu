#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cstdio>

// Test FP4 conversion
__global__ void test_fp4_convert(const float* in, unsigned char* out, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        __nv_fp4_storage_t fp4_val = __nv_cvt_float_to_fp4(in[idx], __NV_E2M1, cudaRoundNearest);
        out[idx] = (unsigned char)fp4_val;
    }
}

int main() {
    printf("FP4 test compiled successfully!\n");
    printf("sizeof __nv_fp4_storage_t: %zu\n", sizeof(__nv_fp4_storage_t));
    printf("sizeof __nv_fp4x2_storage_t: %zu\n", sizeof(__nv_fp4x2_storage_t));
    
    // Test conversion (host-side)
    float val = 1.0f;
    __nv_fp4_storage_t fp4 = __nv_cvt_float_to_fp4(val, __NV_E2M1, cudaRoundNearest);
    printf("FP4(1.0) = 0x%x\n", (unsigned char)fp4);
    
    // Test conv float -> fp4 with GPU kernel
    float* d_in;
    unsigned char* d_out;
    cudaMalloc(&d_in, 1024 * sizeof(float));
    cudaMalloc(&d_out, 1024);
    test_fp4_convert<<<1, 1024>>>(d_in, d_out, 1024);
    cudaDeviceSynchronize();
    printf("Kernel launched OK\n");
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}

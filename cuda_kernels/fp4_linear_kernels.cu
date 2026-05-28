#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Compute per-block (16 elements) max absolute value
__global__ void fp4_block_max_kernel(
    const __nv_bfloat16* __restrict__ inp,
    float* __restrict__ out_scales,
    int N
) {
    int block_idx = blockIdx.x;
    int base = block_idx * 16;
    if (base >= N) return;

    float max_val = 0.0f;
    int end = min(base + 16, N);
    for (int i = base; i < end; i++) {
        float f = __bfloat162float(inp[i]);
        max_val = fmaxf(max_val, fabsf(f));
    }

    float scale = (max_val > 1e-10f) ? (max_val / 6.0f) : 1.0f;
    out_scales[block_idx] = scale;
}

// Quantize bf16 → FP4 using pre-computed scales
__global__ void fp4_quantize_vals_kernel(
    const __nv_bfloat16* __restrict__ inp,
    const float* __restrict__ scales,
    unsigned char* __restrict__ out_packed,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float f = __bfloat162float(inp[i]);
    float scale = scales[i / 16];
    float scaled = f / scale;

    __nv_fp4_e2m1 fp4_val(scaled);
    unsigned char bits = (unsigned char)fp4_val.__x;

    if ((i & 1) == 0) {
        out_packed[i / 2] = bits;
    } else {
        out_packed[i / 2] |= (bits << 4);
    }
}

// FP4 E2M1 → bf16 dequantization
__global__ void fp4_dequantize_kernel(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned char byte = packed[i / 2];
    unsigned char fp4_bits = (i & 1) ? (byte >> 4) : (byte & 0xF);

    __nv_fp4_e2m1 fp4_val;
    fp4_val.__x = (__nv_fp4_storage_t)fp4_bits;
    float f = static_cast<float>(fp4_val);

    float scale = scales[i / 16];
    f = f * scale;

    out[i] = __float2bfloat16(f);
}

extern "C" {

void launch_fp4_dequantize(
    const unsigned char* packed,
    const float* scales,
    __nv_bfloat16* out,
    int N
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fp4_dequantize_kernel<<<blocks, threads>>>(packed, scales, out, N);
}

void launch_fp4_quantize(
    const __nv_bfloat16* inp,
    unsigned char* out_packed,
    float* out_scales,
    int N
) {
    int num_blocks = (N + 15) / 16;
    fp4_block_max_kernel<<<num_blocks, 1>>>(inp, out_scales, N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fp4_quantize_vals_kernel<<<blocks, threads>>>(inp, out_scales, out_packed, N);
}

} // extern "C"

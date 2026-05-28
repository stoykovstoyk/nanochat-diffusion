#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Fused RMS Norm forward kernel
template <int BLOCK_SIZE>
__global__ void fused_rms_norm_forward_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ rstd,
    int B, int T, int N, float eps
) {
    int row = blockIdx.x;
    if (row >= B * T) return;

    __shared__ float s_sum[BLOCK_SIZE];
    float local_sum = 0.0f;
    int tid = threadIdx.x;

    for (int i = tid; i < N; i += blockDim.x) {
        float val = __bfloat162float(x[row * (size_t)N + i]);
        local_sum += val * val;
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / (float)N;
        rstd[row] = rsqrtf(mean + eps);
    }
    __syncthreads();

    float rstd_val = rstd[row];
    for (int i = tid; i < N; i += blockDim.x) {
        float val = __bfloat162float(x[row * (size_t)N + i]);
        float normalized = val * rstd_val;
        float scaled = normalized * w[i];
        out[row * (size_t)N + i] = __float2bfloat16(scaled);
    }
}

// RMS Norm forward (no weight)
template <int BLOCK_SIZE>
__global__ void rms_norm_forward_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ rstd,
    int B, int T, int N, float eps
) {
    int row = blockIdx.x;
    if (row >= B * T) return;

    __shared__ float s_sum[BLOCK_SIZE];
    float local_sum = 0.0f;
    int tid = threadIdx.x;

    for (int i = tid; i < N; i += blockDim.x) {
        float val = __bfloat162float(x[row * (size_t)N + i]);
        local_sum += val * val;
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / (float)N;
        rstd[row] = rsqrtf(mean + eps);
    }
    __syncthreads();

    float rstd_val = rstd[row];
    for (int i = tid; i < N; i += blockDim.x) {
        float val = __bfloat162float(x[row * (size_t)N + i]);
        out[row * (size_t)N + i] = __float2bfloat16(val * rstd_val);
    }
}

// Fused RMS Norm backward kernel
__global__ void fused_rms_norm_backward_kernel(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ dx,
    float* __restrict__ dw,
    int B, int T, int N
) {
    int row = blockIdx.x;
    if (row >= B * T) return;

    int tid = threadIdx.x;
    extern __shared__ float s_buf[];
    float* s_sum = s_buf;

    float rstd_val = rstd[row];
    float row_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float dout_val = __bfloat162float(dout[row * (size_t)N + i]);
        float x_val = __bfloat162float(x[row * (size_t)N + i]);
        float w_val = w[i];
        row_sum += dout_val * w_val * x_val * rstd_val;
    }

    s_sum[tid] = row_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    float row_dot = s_sum[0];
    float dx_coeff = rstd_val / (float)N;

    for (int i = tid; i < N; i += blockDim.x) {
        float dout_val = __bfloat162float(dout[row * (size_t)N + i]);
        float x_val = __bfloat162float(x[row * (size_t)N + i]);
        float w_val = w[i];
        float x_norm = x_val * rstd_val;

        float dx_val = (dout_val * w_val - x_norm * row_dot * dx_coeff) * rstd_val;
        dx[row * (size_t)N + i] = __float2bfloat16(dx_val);
    }

    for (int i = tid; i < N; i += blockDim.x) {
        float dout_val = __bfloat162float(dout[row * (size_t)N + i]);
        float x_val = __bfloat162float(x[row * (size_t)N + i]);
        atomicAdd(&dw[i], dout_val * x_val * rstd_val);
    }
}

// RMS Norm backward (no weight)
__global__ void rms_norm_backward_kernel(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ dx,
    int B, int T, int N
) {
    int row = blockIdx.x;
    if (row >= B * T) return;

    int tid = threadIdx.x;
    extern __shared__ float s_buf[];
    float* s_sum = s_buf;

    float rstd_val = rstd[row];
    float row_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float dout_val = __bfloat162float(dout[row * (size_t)N + i]);
        float x_val = __bfloat162float(x[row * (size_t)N + i]);
        row_sum += dout_val * x_val;
    }

    s_sum[tid] = row_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    float row_dot = s_sum[0];
    float dx_coeff = rstd_val / (float)N;

    for (int i = tid; i < N; i += blockDim.x) {
        float dout_val = __bfloat162float(dout[row * (size_t)N + i]);
        float x_val = __bfloat162float(x[row * (size_t)N + i]);
        float x_norm = x_val * rstd_val;

        float dx_val = (dout_val - x_norm * row_dot * dx_coeff) * rstd_val;
        dx[row * (size_t)N + i] = __float2bfloat16(dx_val);
    }
}

extern "C" {

void fused_rms_norm_forward(
    const void* x, const void* w, void* out, void* rstd,
    int B, int T, int N, float eps
) {
    int rows = B * T;
    int threads = min(256, N);
    fused_rms_norm_forward_kernel<256><<<rows, threads>>>(
        (const __nv_bfloat16*)x, (const float*)w,
        (__nv_bfloat16*)out, (float*)rstd,
        B, T, N, eps
    );
}

void fused_rms_norm_backward(
    const void* dout, const void* x, const void* w, const void* rstd,
    void* dx, void* dw,
    int B, int T, int N
) {
    int rows = B * T;
    int threads = min(256, N);
    int shared_mem = threads * sizeof(float);
    fused_rms_norm_backward_kernel<<<rows, threads, shared_mem>>>(
        (const __nv_bfloat16*)dout, (const __nv_bfloat16*)x,
        (const float*)w, (const float*)rstd,
        (__nv_bfloat16*)dx, (float*)dw,
        B, T, N
    );
}

void rms_norm_forward(
    const void* x, void* out, void* rstd,
    int B, int T, int N, float eps
) {
    int rows = B * T;
    int threads = min(256, N);
    rms_norm_forward_kernel<256><<<rows, threads>>>(
        (const __nv_bfloat16*)x,
        (__nv_bfloat16*)out, (float*)rstd,
        B, T, N, eps
    );
}

void rms_norm_backward(
    const void* dout, const void* x, const void* rstd,
    void* dx,
    int B, int T, int N
) {
    int rows = B * T;
    int threads = min(256, N);
    int shared_mem = threads * sizeof(float);
    rms_norm_backward_kernel<<<rows, threads, shared_mem>>>(
        (const __nv_bfloat16*)dout, (const __nv_bfloat16*)x,
        (const float*)rstd,
        (__nv_bfloat16*)dx,
        B, T, N
    );
}

} // extern "C"

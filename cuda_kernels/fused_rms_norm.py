import torch
import ctypes
from torch.utils.cpp_extension import load_inline

cuda_source = open('/home/stoyko/Desktop/nanochat-diffusion/cuda_kernels/rms_norm_kernels.cu').read()

cpp_decls = '''
extern "C" {
    void fused_rms_norm_forward(const void*, const void*, void*, void*, int, int, int, float);
    void fused_rms_norm_backward(const void*, const void*, const void*, const void*, void*, void*, int, int, int);
    void rms_norm_forward(const void*, void*, void*, int, int, int, float);
    void rms_norm_backward(const void*, const void*, const void*, void*, int, int, int);
}
'''

rms_norm_module = load_inline(
    name='fused_rms_norm',
    cpp_sources=cpp_decls,
    cuda_sources=cuda_source,
    functions=['fused_rms_norm_forward', 'fused_rms_norm_backward', 'rms_norm_forward', 'rms_norm_backward'],
    extra_include_paths=[
        '/home/stoyko/Desktop/nanochat-diffusion/.venv/include/python3.12',
        '/home/stoyko/Desktop/nanochat-diffusion/.venv/include',
    ],
    extra_cuda_cflags=['-arch=sm_120', '-O3', '--use_fast_math'],
    verbose=False,
)

so_path = rms_norm_module.__file__
_lib = ctypes.CDLL(so_path)

# Set argtypes for weighted versions
_lib.fused_rms_norm_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
_lib.fused_rms_norm_forward.restype = None
_lib.fused_rms_norm_backward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_lib.fused_rms_norm_backward.restype = None

# Set argtypes for unweighted versions
_lib.rms_norm_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
_lib.rms_norm_forward.restype = None
_lib.rms_norm_backward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_lib.rms_norm_backward.restype = None


class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        B, T, N = x.shape
        out = torch.empty_like(x, dtype=torch.bfloat16)
        rstd = torch.empty(B * T, dtype=torch.float32, device=x.device)
        _lib.fused_rms_norm_forward(
            ctypes.c_void_p(x.contiguous().data_ptr()),
            ctypes.c_void_p(weight.contiguous().data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(rstd.data_ptr()),
            B, T, N, eps,
        )
        ctx.save_for_backward(x, weight, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        B, T, N = x.shape
        grad_x = torch.empty_like(x, dtype=torch.bfloat16)
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        _lib.fused_rms_norm_backward(
            ctypes.c_void_p(grad_output.contiguous().data_ptr()),
            ctypes.c_void_p(x.contiguous().data_ptr()),
            ctypes.c_void_p(weight.contiguous().data_ptr()),
            ctypes.c_void_p(rstd.contiguous().data_ptr()),
            ctypes.c_void_p(grad_x.data_ptr()),
            ctypes.c_void_p(grad_w.data_ptr()),
            B, T, N,
        )
        return grad_x, grad_w, None


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=1e-5):
        x_flat = x.contiguous().view(-1, x.size(-1))
        M, N = x_flat.shape
        out = torch.empty_like(x_flat, dtype=torch.bfloat16)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        _lib.rms_norm_forward(
            ctypes.c_void_p(x_flat.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(rstd.data_ptr()),
            M, 1, N, eps,
        )
        ctx.shape = x.shape
        ctx.save_for_backward(x_flat, rstd)
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, rstd = ctx.saved_tensors
        x_shape = ctx.shape
        M, N = x_flat.shape
        grad_flat = grad_output.contiguous().view(-1, N)
        grad_x_flat = torch.empty_like(x_flat, dtype=torch.bfloat16)
        _lib.rms_norm_backward(
            ctypes.c_void_p(grad_flat.data_ptr()),
            ctypes.c_void_p(x_flat.data_ptr()),
            ctypes.c_void_p(rstd.contiguous().data_ptr()),
            ctypes.c_void_p(grad_x_flat.data_ptr()),
            M, 1, N,
        )
        return grad_x_flat.view(x_shape), None


def fused_rms_norm(x, weight, eps=1e-5):
    return FusedRMSNormFunction.apply(x, weight, eps)


def rms_norm(x, eps=1e-5):
    return RMSNormFunction.apply(x, eps)


if __name__ == '__main__':
    torch.manual_seed(42)
    B, T, N = 16, 512, 512
    x = torch.randn(B, T, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    # No-weight version: compare with F.rms_norm
    ref_out = torch.nn.functional.rms_norm(x.float(), (N,), None, 1e-5).bfloat16()
    ref_out.sum().backward()
    ref_gx = x.grad.clone()
    x.grad = None

    out = rms_norm(x)
    out.sum().backward()
    fused_gx = x.grad.clone()

    print('=== No-weight RMS Norm ===')
    print(f'Output max diff: {(out - ref_out).abs().max().item():.6f}')
    print(f'dx max diff:     {(fused_gx - ref_gx).abs().max().item():.6f}')

    for _ in range(10):
        out = rms_norm(x)
        out.sum().backward()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    t0.record()
    for _ in range(100):
        out = rms_norm(x)
        out.sum().backward()
    t1.record()
    torch.cuda.synchronize()
    fused_ms = t0.elapsed_time(t1) / 100

    x.grad = None
    t0.record()
    for _ in range(100):
        ref_out = torch.nn.functional.rms_norm(x.float(), (N,), None, 1e-5).bfloat16()
        ref_out.sum().backward()
    t1.record()
    torch.cuda.synchronize()
    ref_ms = t0.elapsed_time(t1) / 100

    print(f'Fused fwd+bwd:  {fused_ms:.3f}ms')
    print(f'Torch fwd+bwd:  {ref_ms:.3f}ms')
    print(f'Speedup: {ref_ms/fused_ms:.2f}x')

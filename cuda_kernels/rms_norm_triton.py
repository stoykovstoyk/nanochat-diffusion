import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_kernel(
    x_ptr, rstd_ptr, out_ptr,
    x_row_stride, out_row_stride,
    N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptrs = x_ptr + row * x_row_stride + tl.arange(0, BLOCK_SIZE)
    out_ptrs = out_ptr + row * out_row_stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = x * x
    sum_x2 = tl.sum(x2, axis=0)
    rstd = 1.0 / tl.sqrt(sum_x2 / N + eps)
    tl.store(rstd_ptr + row, rstd)

    out = x * rstd
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    dout_ptr, x_ptr, rstd_ptr, dx_ptr,
    x_row_stride, dx_row_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptrs = x_ptr + row * x_row_stride + tl.arange(0, BLOCK_SIZE)
    dout_ptrs = dout_ptr + row * x_row_stride + tl.arange(0, BLOCK_SIZE)
    dx_ptrs = dx_ptr + row * dx_row_stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(dout_ptrs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    x_norm = x * rstd
    row_dot = tl.sum(dout * x_norm, axis=0)
    dx_coeff = rstd / N
    dx = (dout - x_norm * row_dot * dx_coeff) * rstd

    tl.store(dx_ptrs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=1e-5):
        x_flat = x.contiguous().view(-1, x.size(-1))
        M, N = x_flat.shape
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        out = torch.empty_like(x)
        out_flat = out.view(-1, N)

        BLOCK_SIZE = triton.next_power_of_2(N)
        grid = (M,)
        _rms_norm_fwd_kernel[grid](
            x_flat, rstd, out_flat,
            x_flat.stride(0), out_flat.stride(0),
            N, eps, BLOCK_SIZE,
        )
        ctx.shape = x.shape
        ctx.save_for_backward(x_flat, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, rstd = ctx.saved_tensors
        x_shape = ctx.shape
        M, N = x_flat.shape
        grad_flat = grad_output.contiguous().view(-1, N)
        grad_x = torch.empty_like(x_flat)

        BLOCK_SIZE = triton.next_power_of_2(N)
        grid = (M,)
        _rms_norm_bwd_kernel[grid](
            grad_flat, x_flat, rstd, grad_x,
            x_flat.stride(0), grad_x.stride(0),
            N, BLOCK_SIZE,
        )
        return grad_x.view(x_shape), None


def rms_norm(x, eps=1e-5):
    return RMSNormFn.apply(x, eps)

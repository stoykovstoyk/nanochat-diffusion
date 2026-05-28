import torch
import ctypes
import os

_kernel_lib = None

def _get_lib():
    global _kernel_lib
    if _kernel_lib is not None:
        return _kernel_lib
    lib_path = os.path.join(os.path.dirname(__file__), "libfp4_kernels.so")
    _kernel_lib = ctypes.cdll.LoadLibrary(lib_path)
    _kernel_lib.launch_fp4_dequantize.argtypes = [
        ctypes.c_void_p,  # packed
        ctypes.c_void_p,  # scales
        ctypes.c_void_p,  # out
        ctypes.c_int,     # N
    ]
    _kernel_lib.launch_fp4_quantize.argtypes = [
        ctypes.c_void_p,  # inp
        ctypes.c_void_p,  # out_packed
        ctypes.c_void_p,  # out_scales
        ctypes.c_int,     # N
    ]
    return _kernel_lib


class FP4Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.master_weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )
        n_weights = out_features * in_features
        self.register_buffer("packed_weight", torch.zeros(
            (n_weights + 1) // 2, dtype=torch.uint8
        ))
        self.register_buffer("scale_factors", torch.zeros(
            (n_weights + 15) // 16, dtype=torch.float32
        ))

        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16)
            )
        else:
            self.register_buffer("bias", None)

        self._quantized = False

    def quantize_weights(self):
        if self._quantized:
            return
        w = self.master_weight.data.view(-1)
        N = w.numel()

        lib = _get_lib()
        cuda_stream = torch.cuda.current_stream()

        lib.launch_fp4_quantize(
            ctypes.c_void_p(w.data_ptr()),
            ctypes.c_void_p(self.packed_weight.data_ptr()),
            ctypes.c_void_p(self.scale_factors.data_ptr()),
            ctypes.c_int(N),
        )
        self._quantized = True

    def dequantize_weights(self) -> torch.Tensor:
        N = self.in_features * self.out_features
        out = torch.empty_like(self.master_weight)

        lib = _get_lib()
        lib.launch_fp4_dequantize(
            ctypes.c_void_p(self.packed_weight.data_ptr()),
            ctypes.c_void_p(self.scale_factors.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(N),
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._quantized:
            w = self.master_weight
        else:
            w = self.dequantize_weights()

        x_dtype = x.dtype
        if x_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        out = torch.nn.functional.linear(x, w, self.bias)
        if x_dtype != torch.bfloat16:
            out = out.to(x_dtype)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, quantized={self._quantized}"

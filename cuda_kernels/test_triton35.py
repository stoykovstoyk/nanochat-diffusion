import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x, y):
    output = torch.zeros_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

x = torch.randn(4, device='cuda')
y = torch.randn(4, device='cuda')
z = add(x, y)
print('Basic kernel works:', torch.allclose(x + y, z))

# Test torch.compile with simple model
import torch.nn as nn
m = nn.Linear(512, 512).cuda().to(torch.bfloat16)
mc = torch.compile(m, mode='default')
for i in range(5):
    inp = torch.randn(16, 512, dtype=torch.bfloat16, device='cuda')
    out = mc(inp)
print('torch.compile OK')

# Test torch.compile with mode='reduce-overhead'
mc2 = torch.compile(m, mode='reduce-overhead')
for i in range(5):
    inp = torch.randn(16, 512, dtype=torch.bfloat16, device='cuda')
    out = mc2(inp)
print('torch.compile reduce-overhead OK')

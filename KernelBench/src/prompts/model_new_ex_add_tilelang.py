import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):

    @T.prim_func
    def main(A: T.Tensor((M, N), in_dtype), B: T.Tensor((M, N), in_dtype), C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                C[y, x] = A[y, x] + B[y, x]

    return main


class ModelNew(nn.Module):
    """
    Simple model that performs element-wise addition of two matrices
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        func = elementwise_add(M, N, 1, 128, "float16", "float16", 128)
        self.compiled_kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs element-wise addition of A and B.

        Args:
            A: Input tensor of shape (M, N)
            B: Input tensor of shape (M, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return self.compiled_kernel(A, B)


M = 1
N = 128

def get_inputs(device):
    A = torch.randn(M, N, device=device, dtype=torch.float16)
    B = torch.randn(M, N, device=device, dtype=torch.float16)
    return [A, B]

def get_init_inputs():
    return []
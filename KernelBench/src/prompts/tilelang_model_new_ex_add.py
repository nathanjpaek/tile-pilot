import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def elementwise_add(M, N, block_M, block_N, in_dtype="float16", out_dtype="float", threads=256):
    @tilelang.jit(
        out_idx=-1, # create the output tensor during runtime
    )
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
    def __init__(self):
        super(ModelNew, self).__init__()
        self.block_M_cfg = 128
        self.block_N_cfg = 128
        self.threads_cfg = 256 

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

        # TileLang only supports float16 on CUDA
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)

        M, N = A.shape

        sum_kernel = elementwise_add(
            M, N,
            self.block_M_cfg, self.block_N_cfg,
            self.input_dtype, self.output_dtype, self.threads_cfg
        )

        sum_kernel = elementwise_add(M, N)
        return sum_kernel(A, B).to(torch.float32)

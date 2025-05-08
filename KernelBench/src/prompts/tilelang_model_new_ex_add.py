import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def elementwise_add(M, N, block_M, block_N, in_dtype="float32", out_dtype="float32", threads=256):
    @tilelang.jit(
        out_idx=-1,
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

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)

        M, K = A.shape
        N = B.shape[1]

        matmul_kernel = elementwise_add(M, N, K)
        return matmul_kernel(A, B)



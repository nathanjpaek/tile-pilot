import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

def transpose(M, N, block_M=128, block_N=128, dtype="float16"):
    @tilelang.jit(
        out_idx=-1,  # create the output tensor during runtime
    )
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        A_T: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_local = T.alloc_fragment((block_M, block_N), dtype)

            for m in T.Parallel(block_M):
                for n in T.Parallel(block_N):
                    A_local[m, n] = A[by * block_M + m, bx * block_N + n]

            T.copy(A_local, A_T[bx * block_N, by * block_M])

    return main

def matmul(M, N, K, block_M=128, block_N=128, block_K=32, dtype="float16", accum_dtype="float"):
    @tilelang.jit(
        out_idx=-1,  # create the output tensor during runtime
    )
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.transpose_kernel = transpose(K, M)
        self.matmul_kernel = matmul(M, N, K)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)

        A_T = torch.empty((M, K), device="cuda", dtype=torch.float16)
        self.transpose_kernel(A, A_T)

        C = torch.empty((M, N), device="cuda", dtype=torch.float16)
        self.matmul_kernel(A_T, B, C)

        return C

M = 1024
K = 4096
N = 2048

model = ModelNew()

A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

C = model(A, B)

C_ref = A.T @ B

torch.testing.assert_close(C, C_ref, rtol=1e-2, atol=1e-2)

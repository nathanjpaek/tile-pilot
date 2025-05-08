import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


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
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Allocate shared and local fragments
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear the local accumulation buffer
            T.clear(C_local)

            # Pipelined iteration over K dimension

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                # This is a sugar syntax for parallelized copy
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Parallelized copy of tiles from global to shared for B
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(A_shared, B_shared, C_local)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    """
    Optimized model using TileLang for matrix multiplication (C = A * B) with a large K dimension
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        # TileLang only supports float16 on CUDA
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)

        M, K = A.shape
        N = B.shape[1]

        matmul_kernel = matmul(M, N, K)
        return matmul_kernel(A, B)


if __name__ == "__main__":
    model = ModelNew()
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    C = model(A, B)
    print(C)

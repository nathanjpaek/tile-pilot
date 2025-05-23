import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def matmul(M, N, K, block_M=128, block_N=128, block_K=32, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Define a grid with enough blocks to cover M×N
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Allocate shared memory for the current tile of A and B
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # Allocate a local (register) fragment for partial accumulations
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Initialize the local accumulation buffer to zero
            T.clear(C_local)

            # Loop over the K dimension in block_K chunks, using a 3-stage pipeline
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Parallelized copy from global memory to shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # Copy the accumulated result from local memory (C_local) to global memory (C)
                T.gemm(A_shared, B_shared, C_local)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def tilelang_matmul(A, B):
    # TileLang inputs must be in float16 precision
    A = A.cuda().half()
    B = B.cuda().half()
    M, K = A.shape
    N = B.shape[1]
    
    # Compile the TileLang kernel with out_idx=-1 to create the output tensor during runtime
    func = matmul(M, N, K)
    kernel = tilelang.compile(func, out_idx=-1)
    return kernel(A, B)


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
        return tilelang_matmul(A, B)
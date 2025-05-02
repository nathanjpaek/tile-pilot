import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

class ModelNew(nn.Module):
    """
    Optimized model using TileLang for matrix multiplication (C = A * B)
    """
    
    def __init__(self, M=256, N=256, K=131072):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K

        # Block sizes for tiling (can be tuned for performance)
        block_M = 16
        block_N = 16
        threads = block_M * block_N

        in_dtype = "float16"
        out_dtype = "float16"

        @T.prim_func
        def matmul_kernel(
            A: T.Tensor((self.M, self.K), in_dtype),
            B: T.Tensor((self.K, self.N), in_dtype),
            C: T.Tensor((self.M, self.N), out_dtype)
        ):
            with T.Kernel(
                T.ceildiv(self.M, block_M),
                T.ceildiv(self.N, block_N),
                threads=threads
            ) as (block_y, block_x):
                start_m = block_y * block_M
                start_n = block_x * block_N
                for (local_m, local_n) in T.Parallel(block_M, block_N):
                    m = start_m + local_m
                    n = start_n + local_n
                    acc = T.float32(0.0)
                    for k in range(self.K):
                        acc += A[m, k] * B[k, n]
                    C[m, n] = acc

        self.compiled_kernel = tilelang.compile(matmul_kernel, out_idx=[2], target="cuda")

    def forward(self, A, B):
        return self.compiled_kernel(A, B)
    
if __name__ == "__main__":
    model = ModelNew()
    A = torch.randn(256, 131072)
    B = torch.randn(131072, 256)
    C = model(A, B)
    print(C.shape)
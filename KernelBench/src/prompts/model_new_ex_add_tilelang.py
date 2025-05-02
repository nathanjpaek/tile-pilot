import torch
import torch.nn as nn
import tilelang 
import tilelang.language as T

class ModelNew(nn.Module):
    def init(self, M=1, N=128) -> None: 
        super().init()
        self.M = M
        self.N = N

        block_M, block_N, threads = 1, 128, 128 
        in_dtype, out_dtype = "float32", "float32"

        @T.prim_func
        def elementwise_add_kernel(A: T.Tensor((self.M, self.N), in_dtype),
                                B: T.Tensor((self.M, self.N), in_dtype),
                                C: T.Tensor((self.M, self.N), out_dtype)):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                start_x = bx * block_N
                start_y = by * block_M
                for (local_y, local_x) in T.Parallel(block_M, block_N):
                    y = start_y + local_y
                    x = start_x + local_x
                    C[y, x] = A[y, x] + B[y, x]

        self.compiled_kernel = tilelang.compile(elementwise_add_kernel, out_idx=-1, target="cuda")

    def forward(self, a, b):
        return self.compiled_kernel(a, b)
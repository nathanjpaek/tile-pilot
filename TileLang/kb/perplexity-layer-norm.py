import torch
import tilelang
import tilelang.language as T

def layernorm_kernel(features, dim1, dim2, threads=256, dtype="float32"):
    @T.prim_func
    def main(
        X: T.Tensor(("B", features, dim1, dim2), dtype),
        gamma: T.Tensor((features, dim1, dim2), dtype),
        beta: T.Tensor((features, dim1, dim2), dtype),
        Y: T.Tensor(("B", features, dim1, dim2), dtype),
        eps: T.Constant(float) = 1e-5
    ):
        M = features * dim1 * dim2
        with T.Kernel(T.symbolic("B"), threads=threads) as batch_idx:
            # Shared memory for block reduction
            shmem_sum = T.allocate((threads,), dtype, "shared")
            shmem_sqsum = T.allocate((threads,), dtype, "shared")
            
            # First pass: Compute mean and variance
            thread_sum = T.allocate((1,), dtype, "local")
            thread_sqsum = T.allocate((1,), dtype, "local")
            thread_sum[0] = 0.0
            thread_sqsum[0] = 0.0
            
            # Grid-stride loop for coalesced memory access
            for i in T.ParallelRange(0, M, step=threads):
                linear_idx = i + T.thread_id()
                if linear_idx < M:
                    # Convert linear index to 3D coordinates
                    f = linear_idx // (dim1 * dim2)
                    d1 = (linear_idx // dim2) % dim1
                    d2 = linear_idx % dim2
                    val = X[batch_idx, f, d1, d2]
                    thread_sum[0] += val
                    thread_sqsum[0] += val * val
            
            # Parallel reduction across threads
            shmem_sum[T.thread_id()] = thread_sum[0]
            shmem_sqsum[T.thread_id()] = thread_sqsum[0]
            T.sync()
            
            # Tree reduction
            stride = threads // 2
            while stride > 0:
                if T.thread_id() < stride:
                    shmem_sum[T.thread_id()] += shmem_sum[T.thread_id() + stride]
                    shmem_sqsum[T.thread_id()] += shmem_sqsum[T.thread_id() + stride]
                T.sync()
                stride >>= 1
            
            # Compute statistics
            if T.thread_id() == 0:
                mean = shmem_sum[0] / M
                variance = (shmem_sqsum[0] / M) - (mean * mean)
                inv_std = T.rsqrt(variance + eps)
                # Broadcast through shared memory
                shmem_sum[0] = mean
                shmem_sqsum[0] = inv_std
            T.sync()
            
            mean = shmem_sum[0]
            inv_std = shmem_sqsum[0]
            
            # Second pass: Normalize and transform
            for i in T.ParallelRange(0, M, step=threads):
                linear_idx = i + T.thread_id()
                if linear_idx < M:
                    f = linear_idx // (dim1 * dim2)
                    d1 = (linear_idx // dim2) % dim1
                    d2 = linear_idx % dim2
                    val = X[batch_idx, f, d1, d2]
                    normalized = (val - mean) * inv_std
                    Y[batch_idx, f, d1, d2] = normalized * gamma[f, d1, d2] + beta[f, d1, d2]
        
        return Y
    return main

# Compilation with shape specialization
features, dim1, dim2 = 64, 256, 256
program = layernorm_kernel(features, dim1, dim2)
kernel = tilelang.compile(program, target="cuda", opt_level=3)

# Modified model using TileLang kernel
class OptimizedModel(torch.nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        return kernel(x, self.gamma, self.beta)

# Usage remains identical
model = OptimizedModel((features, dim1, dim2)).cuda()
x = torch.randn(16, 64, 256, 256, device="cuda")
output = model(x)

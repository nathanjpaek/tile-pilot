import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def rms_norm(M, N, blk_m=128, blk_n=256, dtype="float16", threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            # Allocate shared memory for the block
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            # Copy input to shared memory
            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            T.copy(A_shared, A_local)

            # Compute square of elements
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]

            # Compute mean along feature dimension
            T.reduce_sum(A_pow_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-5)

            # Normalize input
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]

            # Copy result back to global memory
            T.copy(A_local, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main


def tilelang_rms_norm(x, eps=1e-5):
    x = x.cuda()
    batch_size, features, dim1, dim2 = x.shape
    
    # Reshape input to 2D for the kernel
    x_reshaped = x.reshape(batch_size * dim1 * dim2, features)
    
    # Create and compile kernel
    func = rms_norm(batch_size * dim1 * dim2, features)
    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    
    # Apply kernel
    output = kernel(x_reshaped)
    
    # Reshape back to original dimensions
    return output.reshape(batch_size, features, dim1, dim2)


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using TileLang kernels.
    """
    def __init__(self, eps: float = 1e-5):
        """
        Initializes the optimized RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using TileLang kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return tilelang_rms_norm(x, self.eps)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features]


if __name__ == "__main__":
    device = torch.device("cuda")
    model = ModelNew().to(device)
    inputs = get_inputs()
    c = model(*inputs)
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def elementwise_add(M, N, block_M=128, block_N=256, in_dtype="float16", out_dtype="float16", threads=128):

    @T.prim_func
    def main(
        A: T.Tensor((M, N), in_dtype),  # First input matrix
        B: T.Tensor((M, N), in_dtype),  # Second input matrix
        C: T.Tensor((M, N), out_dtype),  # Output matrix
    ):
        # Launch grid of blocks to cover the entire matrix
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            # Calculate starting indices for this block
            start_x = bx * block_N
            start_y = by * block_M

            # Parallel iteration over elements within the block
            for local_y, local_x in T.Parallel(block_M, block_N):
                # Convert local indices to global matrix indices
                y = start_y + local_y
                x = start_x + local_x

                # Perform element-wise addition
                C[y, x] = A[y, x] + B[y, x]

    return main


def tilelang_elem_add(A, B):
    """
    Performs element-wise addition of two tensors using TileLang.

    Args:
        A: First input tensor
        B: Second input tensor

    Returns:
        Result of element-wise addition A + B
    """
    # Move inputs to GPU
    A = A.cuda()
    B = B.cuda()

    # Get matrix dimensions
    M, N = A.shape

    # Create and compile the TileLang kernel
    func = elementwise_add(M, N)
    kernel = tilelang.compile(func, out_idx=[2], target="cuda")

    # Execute the kernel
    return kernel(A, B)


class ModelNew(nn.Module):
    """
    Simple model that performs element-wise addition of two matrices
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs element-wise addition of A and B.

        Args:
            A: Input tensor of shape (M, N)
            B: Input tensor of shape (M, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return tilelang_elem_add(A, B)


M = 1
N = 128


def get_inputs(device):
    A = torch.randn(M, N, device=device, dtype=torch.float16)
    B = torch.randn(M, N, device=device, dtype=torch.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


if __name__ == "__main__":
    device = torch.device("cuda")
    model = ModelNew().to(device)
    inputs = get_inputs(device)
    c = model(*inputs)

    c_ref = inputs[0] + inputs[1]

    torch.testing.assert_close(c, c_ref)
    print("Success!")

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def sftmx(batch_size: int, dim: int, dtype: str = "float", n_threads: int = 1024):
   scale = 1.44269504 # log2(e)
   
   @T.prim_func
   def wrapper(
       src: T.Tensor((batch_size, dim), dtype=dtype),  # type: ignore
       des: T.Tensor((batch_size, dim), dtype=dtype)   # type: ignore
   ):
       with T.Kernel(1, 1, threads=n_threads) as (bx, by):
           buf = T.alloc_fragment((batch_size, dim), dtype=dtype)
           maxes = T.alloc_fragment((batch_size, ), dtype=dtype)
           norms = T.alloc_fragment((batch_size, ), dtype=dtype)
           
           T.fill(maxes, -T.infinity(dtype))
           T.copy(src, buf)
           T.reduce_max(buf, maxes, dim=1, clear=False)
           for i, j in T.Parallel(batch_size, dim):
               buf[i, j] = T.exp2(buf[i, j] * scale - maxes[i] * scale)
               
           T.reduce_sum(buf, norms, dim=1)
           for i, j in T.Parallel(batch_size, dim):
               des[i, j] = buf[i, j] / norms[i]
               
   return wrapper


def tilelang_softmax(x):
    M, N = x.shape
    func = sftmx(M, N)
    kernel = tilelang.compile(func, out_idx=-1)
    return kernel(x)


class ModelNew(nn.Module):
    """
    Optimized model using TileLang for Softmax activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Define and compile TileLang kernel
        self.kernel = tilelang_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return self.kernel(x)


batch_size = 1024
dim = 1024

def get_inputs(device):
    x = torch.randn(batch_size, dim, device=device, dtype=torch.float16)
    return [x]

x = get_inputs(torch.device("cuda"))
model = ModelNew()
y = model(*x)
print(y)
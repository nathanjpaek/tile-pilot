import tilelang
import tilelang.language as T

def elementwise_add(N, threads=256, dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype), 
        C: T.Tensor((N,), dtype)
    ):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as block_idx:
            for i in T.Parallel(threads):
                idx = block_idx * threads + i
                if idx < N:  # Boundary check
                    C[idx] = A[idx] + B[idx]
    return main

# Compilation (dynamic shape example)
program = elementwise_add(T.symbolic("N"), threads=256, dtype="float32")
kernel = tilelang.compile(program, out_idx=-1, target="cuda")

# Usage with PyTorch tensors
import torch
a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
c = kernel(a, b)  # Equivalent to torch.add(a, b)

c_ref = a + b

torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")

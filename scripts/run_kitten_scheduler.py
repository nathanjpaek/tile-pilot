import modal
import os
import KernelBench

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang" 
                )
    .pip_install(  
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
    )
    #.run_commands(
    #    "apt-get update && apt-get install -y git gcc-12 g++-12 clang",
    #    "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100",
    #    "update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100",
    #)
    .copy_local_dir("./thunderkittens", "/ThunderKittens")
    .copy_local_dir("./kernelbench", "/kernelbench")
)

app = modal.App("tk-kernelbench", image=image)

@app.function(gpu="H100", timeout=1800)
def run_evaluation(problem_id, tk_kernel_code):
    import sys
    import os
    import torch
    import tempfile
    
    os.environ["THUNDERKITTENS_ROOT"] = "/ThunderKittens"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "Hopper"  # For H100 GPUs
    
    sys.path.append("/KernelBench")
    sys.path.append("/ThunderKittens")
    
    # Import kernelbench evaluation function
    from kernelbench.eval import eval_kernel_against_ref
    from kernelbench.dataset import construct_kernelbench_dataset
    
    # Get the dataset and reference architecture
    dataset = construct_kernelbench_dataset(level=1)  # Adjust level as needed
    
    # Find the reference architecture for this problem
    problem_idx = problem_id - 1  # Convert to 0-indexed
    ref_arch_path = dataset[problem_idx]
    
    # Read reference architecture
    with open(ref_arch_path, "r") as f:
        ref_arch_src = f.read()
    
    # Write kernel code to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        kernel_path = f.name
        f.write(tk_kernel_code)
    
    # Run evaluation
    result = eval_kernel_against_ref(
        ref_arch_src, 
        tk_kernel_code,
        num_correct_trials=5,
        num_perf_trials=100,
        measure_performance=True,
        device=torch.device("cuda:0")
    )
    
    return result
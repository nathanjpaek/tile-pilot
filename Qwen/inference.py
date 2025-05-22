"""
This module performs inference using vLLM.

I used this to generate the results on the test set for the original model (Qwen2.5-Coder-7B-Instruct) and
Qwen2.5-Coder-14B-Instruct.

The results on the test set of the model trained with Unsloth are not generated with this script, due to Unsloth
bugs. See unsloth_inference.py for more details.

The results saved by this script can be evaluated with the eval.py script.
"""

MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"

KB_LEVEL = 1
KB_PROBLEM = 6

# ! pip install vllm datasets

import os
import datasets
from vllm import LLM, SamplingParams


# Load the model
# max_model_len: Maximum sequence length the model can handle (context window size)
#   For coding tasks, a larger context length (4096-8192) is recommended to fit more code
# gpu_memory_utilization: Fraction of GPU memory to use (0.9 = 90% of available memory)
#   This controls the trade-off between memory usage and performance
llm = LLM(model=MODEL_NAME, max_model_len=4096, gpu_memory_utilization=0.9)

SOURCE = open("../KernelBench/KernelBench/level1/6_Matmul_with_large_K_dimension_.py").read()

PYTORCH_MODEL = open("../KernelBench/KernelBench/level1/23_Softmax.py").read()

TILELANG_SOURCE = open("../TileLang/kb/correct-tilelang-6.py").read()

SYSTEM_PROMPT = """You are an expert kernel engineer specializing in TileLang.
1. First, analyze the PyTorch model inside <think> and </think> tags. Here you can reason about the computation, identify optimization opportunities, and plan your kernel implementation.
2. When confident, output your optimized TileLang implementation inside <code> and </code> tags."""

USER_PROMPT = """Task: Optimize the given PyTorch model by implementing custom TileLang kernels.

Rules:
- You must create a ModelNew class that inherits from nn.Module
- Define and compile TileLang kernels in the __init__ method
- Call the compiled kernel in the forward method
- Focus on replacing PyTorch operators with efficient TileLang implementations
- Ensure your implementation maintains the same functionality as the original model
- Use proper tilelang.language (T) constructs for parallelism and memory access patterns

This is an example PyTorch model that you would receive:

<pytorch_source>
{SOURCE}
</pytorch_source>

This is an example of expected TileLang kernel implementation to speed up the matmul operation:

<tilelang_source>
{TILELANG_SOURCE}
</tilelang_source>

The PyTorch model that you must optimize using TileLang is:

<pytorch_model>
{PYTORCH_MODEL}
</pytorch_model>

You must use this format:

<think>
[Analyze the PyTorch model, identify operations to optimize, and plan your TileLang implementation]
</think>

<code>
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Define and compile TileLang kernel
        ...
    
    def forward(self, ...):
        # Call the compiled kernel
        ...
</code>
"""

# Retrieve the problem from the dataset
ds = datasets.load_dataset("ScalingIntelligence/KernelBench")

curr_level_ds = ds[f"level_{KB_LEVEL}"]
curr_problem_row = curr_level_ds.filter(lambda x: x["problem_id"] == KB_PROBLEM)
ref_arch_src = curr_problem_row["code"][0]
problem_name = curr_problem_row["name"][0]

user_prompt = USER_PROMPT.format(SOURCE=SOURCE, TILELANG_SOURCE=TILELANG_SOURCE, PYTORCH_MODEL=PYTORCH_MODEL)

# Create conversations with the correct prompt field
conversations = [
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
]

# Perform inference and save the results
outputs = llm.chat(conversations, sampling_params=SamplingParams(max_tokens=2000))

path = f"results/{MODEL_NAME.split('/')[-1]}"
os.makedirs(path, exist_ok=True)

for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text

    with open(f"{path}/{i}.txt", "w") as f:
        f.write(generated_text)

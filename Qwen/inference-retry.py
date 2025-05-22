"""
This module performs inference using vLLM, specifically for retrying failed attempts with error feedback.

It reads the previous attempt from the results directory and creates a new conversation
that includes the error message for the model to learn from.

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
2. When confident, output your optimized TileLang implementation inside <code> and </code> tags.
3. If you see an error message from a previous attempt, carefully analyze it and fix the issues in your new implementation."""

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

{ERROR_MESSAGE}

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

def get_latest_attempt():
    """Get the latest attempt number from the results directory."""
    path = f"results/{MODEL_NAME.split('/')[-1]}"
    if not os.path.exists(path):
        return 0
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    if not files:
        return 0
    return max(int(f.split('.')[0]) for f in files)

def read_previous_attempt(attempt_num):
    """Read the previous attempt from the results directory."""
    path = f"results/{MODEL_NAME.split('/')[-1]}/{attempt_num}.txt"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return f.read()

def main():
    # Get the latest attempt number
    latest_attempt = get_latest_attempt()
    previous_output = read_previous_attempt(latest_attempt)
    
    if previous_output is None:
        print("No previous attempt found. Please run inference.py first.")
        return
    
    # Get error message from user
    print("Please paste the error message from the previous attempt (press Ctrl+D when done):")
    error_message = ""
    try:
        while True:
            line = input()
            error_message += line + "\n"
    except EOFError:
        pass
    
    # Format the error message for the prompt
    error_section = f"""
Previous attempt failed with the following error:
<error>
{error_message}
</error>

Please fix these issues in your new implementation.
"""
    
    # Create the user prompt with error message
    user_prompt = USER_PROMPT.format(
        SOURCE=SOURCE,
        TILELANG_SOURCE=TILELANG_SOURCE,
        PYTORCH_MODEL=PYTORCH_MODEL,
        ERROR_MESSAGE=error_section
    )
    
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
    
    # Save with incremented attempt number
    new_attempt = latest_attempt + 1
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        with open(f"{path}/{new_attempt}.txt", "w") as f:
            f.write(generated_text)
    
    print(f"\nNew attempt saved as {path}/{new_attempt}.txt")

if __name__ == "__main__":
    main()

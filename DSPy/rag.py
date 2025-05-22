"""
DSPy RAG Pipeline for TileLang Kernel Optimization

This follows the official DSPy RAG tutorial structure for building
a retrieval-augmented generation system for TileLang optimizations.
"""

import os
import json
import dspy
from pathlib import Path
import numpy as np
from typing import List, Dict

# Configure DSPy with the LM at the start
import os

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)  # or 'openai/gpt-4' for better quality
dspy.configure(lm=lm)


# ============= Step 1: Load and Process Examples =============


def load_tilelang_examples(examples_dir: str) -> tuple:
    """Load TileLang examples and create a corpus for retrieval."""
    examples = []
    corpus_texts = []
    corpus_metadata = []

    # Each subdirectory is an example
    for example_dir in [d for d in Path(examples_dir).iterdir() if d.is_dir()]:
        example_id = example_dir.name

        # Read the required files
        original_path = example_dir / "original.py"
        tilelang_path = example_dir / "tilelang.py"
        metadata_path = example_dir / "metadata.json"

        # Skip if required files don't exist
        if not (original_path.exists() and tilelang_path.exists()):
            print(f"Warning: Skipping {example_id} - missing required files")
            continue

        # Read files
        with open(original_path, "r") as f:
            original_code = f.read()

        with open(tilelang_path, "r") as f:
            tilelang_code = f.read()

        # Read metadata if exists
        metadata = {"example_id": example_id}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata.update(json.load(f))

        # Create example for training/validation
        example = dspy.Example(
            original_code=original_code, tilelang_code=tilelang_code, metadata=metadata
        ).with_inputs("original_code")
        examples.append(example)

        # Create corpus entries for retrieval
        # Entry 1: Original code with metadata
        corpus_text = f"""Problem Type: {metadata.get('problem_type', 'unknown')}
Operation: {metadata.get('operation', 'unknown')}
Optimization Techniques: {', '.join(metadata.get('optimization_techniques', []))}
Key Insights: {' '.join(metadata.get('key_insights', []))}

Original PyTorch Code:
{original_code[:1000]}..."""

        corpus_texts.append(corpus_text)
        corpus_metadata.append({"type": "original", "example_id": example_id, "dir_path": str(example_dir)})

        # Entry 2: TileLang implementation
        corpus_text = f"""Problem Type: {metadata.get('problem_type', 'unknown')}
TileLang Optimized Implementation:
{tilelang_code[:1000]}..."""

        corpus_texts.append(corpus_text)
        corpus_metadata.append({"type": "tilelang", "example_id": example_id, "dir_path": str(example_dir)})

        print(f"Loaded example: {example_id}")

    return examples, corpus_texts, corpus_metadata


# ============= Step 2: Set up Retriever =============


def setup_retriever(
    corpus_texts: List[str],
    corpus_metadata: List[Dict],
    embedder_model: str = "openai/text-embedding-3-small",
):
    """Set up the DSPy retriever following the official tutorial."""

    # Following the tutorial structure
    max_characters = 6000  # for truncating documents
    topk_docs_to_retrieve = 3  # number of documents to retrieve

    # Truncate corpus texts
    corpus_texts = [text[:max_characters] for text in corpus_texts]

    print(f"Setting up retriever with {len(corpus_texts)} documents...")

    # Create embedder and retriever as in the tutorial
    embedder = dspy.Embedder(embedder_model, dimensions=512)
    search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus_texts, k=topk_docs_to_retrieve)

    # Store metadata separately for later use
    search.corpus_metadata = corpus_metadata

    return search


# ============= Step 3: Define RAG Module =============


class TileLangRAG(dspy.Module):
    """RAG module for TileLang optimization following the tutorial structure."""

    def __init__(self, search_fn):
        super().__init__()
        self.search = search_fn
        self.respond = dspy.ChainOfThought("context, pytorch_code -> tilelang_code")

    def forward(self, pytorch_code: str):
        # Search for relevant examples
        context_results = self.search(pytorch_code)

        # Get the actual code from the examples
        context_parts = []
        seen_examples = set()

        for i, passage in enumerate(context_results.passages):
            # Get metadata for this result
            if hasattr(self.search, "corpus_metadata") and i < len(self.search.corpus_metadata):
                metadata = self.search.corpus_metadata[i]
                example_id = metadata.get("example_id", "")

                # Avoid duplicates from same example
                if example_id in seen_examples:
                    continue
                seen_examples.add(example_id)

                # Load the full code if we have the path
                if "dir_path" in metadata:
                    example_dir = Path(metadata["dir_path"])

                    # Add a complete example to context
                    original_path = example_dir / "original.py"
                    tilelang_path = example_dir / "tilelang.py"

                    if original_path.exists() and tilelang_path.exists():
                        with open(original_path, "r") as f:
                            original = f.read()
                        with open(tilelang_path, "r") as f:
                            tilelang = f.read()

                        context_parts.append(
                            f"""Example {len(context_parts) + 1} - {example_id}:
Original PyTorch:
```python
{original[:1500]}
```

Optimized TileLang:
```python
{tilelang[:1500]}
```
"""
                        )
            else:
                # Fallback to passage text
                context_parts.append(f"Example {i+1}:\n{passage}")

        context = "\n---\n".join(context_parts)

        # Generate TileLang code
        return self.respond(context=context, pytorch_code=pytorch_code)


# ============= Step 4: Define Evaluation Metric =============


def tilelang_correctness_metric(example, pred, trace=None):
    """
    Evaluate if the generated TileLang code is correct.
    This is a simplified metric - you should implement proper validation.
    """
    score = 0.0

    # Check if ModelNew class exists
    if "class ModelNew" in pred.tilelang_code:
        score += 0.25

    # Check if it imports tilelang
    if "import tilelang" in pred.tilelang_code:
        score += 0.25

    # Check if it has a forward method
    if "def forward" in pred.tilelang_code:
        score += 0.25

    # Check if it uses T.prim_func or T.kernel
    if "@T.prim_func" in pred.tilelang_code or "@T.kernel" in pred.tilelang_code:
        score += 0.25

    return score


# ============= Step 5: Main Pipeline Setup =============


def setup_tilelang_pipeline(examples_dir: str, model: str = None, api_key: str = None):
    """Set up the complete TileLang optimization pipeline."""

    # Optionally reconfigure with a different model
    if model:
        if api_key:
            lm = dspy.LM(model, api_key=api_key)
        else:
            lm = dspy.LM(model)  # Uses OPENAI_API_KEY env var
        dspy.configure(lm=lm)

    # Load examples and create corpus
    print("Loading TileLang examples...")
    examples, corpus_texts, corpus_metadata = load_tilelang_examples(examples_dir)

    # Set up retriever
    search = setup_retriever(corpus_texts, corpus_metadata)

    # Create RAG module
    rag = TileLangRAG(search)

    return rag, examples


# ============= Step 6: Optimization =============


def optimize_pipeline(rag, examples, metric=None):
    """Optimize the RAG pipeline using DSPy optimizers."""

    if metric is None:
        metric = tilelang_correctness_metric

    # Split examples into train and validation
    import random

    random.Random(0).shuffle(examples)

    train_size = min(20, len(examples) // 5)  # 20% for training
    trainset = examples[:train_size]
    valset = examples[train_size:]

    print(f"Optimizing with {len(trainset)} training and {len(valset)} validation examples")

    # Use MIPROv2 optimizer as in the tutorial
    tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=8)

    optimized_rag = tp.compile(
        rag,
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        requires_permission_to_run=False,
    )

    return optimized_rag


# ============= Step 7: Usage Example =============


def main():
    """Example usage following the tutorial pattern."""

    # Set up the pipeline (uses the default model configured at top)
    rag, examples = setup_tilelang_pipeline(examples_dir="./examples")

    # Example PyTorch code to optimize
    pytorch_code = open("../KernelBench/KernelBench/level1/19_ReLU.py").read()

    # Generate optimization
    print("\nGenerating TileLang optimization...")
    result = rag(pytorch_code)

    print("\nGenerated TileLang Code:")
    print(result.tilelang_code)

    # Optionally optimize the pipeline
    if len(examples) > 10:
        print("\nOptimizing pipeline...")
        optimized_rag = optimize_pipeline(rag, examples)

        # Save the optimized model
        optimized_rag.save("optimized_tilelang_rag.json")
        print("Saved optimized model to optimized_tilelang_rag.json")

        # Test optimized version
        optimized_result = optimized_rag(pytorch_code)
        print("\nOptimized TileLang Code:")
        print(optimized_result.tilelang_code)

    # Inspect history to see prompts
    print("\nInspecting last prompt:")
    dspy.inspect_history(n=1)


# ============= Integration with your inference script =============


def generate_with_rag(pytorch_model_code: str, examples_dir: str = "../TileLang/examples"):
    """
    Function to use in your inference script instead of direct LLM calls.
    """
    # Load saved optimized model if available
    rag = TileLangRAG(None)  # Dummy init

    try:
        # Try to load optimized version
        rag.load("optimized_tilelang_rag.json")
        print("Loaded optimized RAG model")
    except:
        # Fall back to creating new one
        rag, _ = setup_tilelang_pipeline(examples_dir)
        print("Using unoptimized RAG model")

    # Generate TileLang code
    result = rag(pytorch_model_code)

    # Extract just the code part (similar to your original format)
    return result.tilelang_code


if __name__ == "__main__":
    main()

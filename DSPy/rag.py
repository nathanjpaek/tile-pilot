"""
DSPy RAG Pipeline for TileLang Kernel Optimization

This script sets up a retrieval-augmented generation system using DSPy
to optimize PyTorch models with TileLang kernels.
"""

import os
import json
import dspy
# from dspy.retrieve import ColBERTv2
from typing import List, Dict, Tuple
import chromadb
from chromadb.utils import embedding_functions
import glob
from pathlib import Path


# ============= Step 1: Document Processing =============


class TileLangDocumentProcessor:
    """Process and structure TileLang documentation and examples."""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.documents = []

    def process_tilelang_examples(self, examples_dir: str) -> List[Dict]:
        """Extract PyTorch -> TileLang transformation examples from folder structure."""
        examples = []

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

            # Read original PyTorch code
            with open(original_path, "r") as f:
                original_code = f.read()

            # Read TileLang optimized code
            with open(tilelang_path, "r") as f:
                tilelang_code = f.read()

            # Read metadata if it exists
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            # Extract additional metadata from code
            metadata.update(self._extract_metadata(tilelang_code))

            # Create example entry
            examples.append(
                {
                    "id": example_id,
                    "dir_path": str(example_dir),
                    "original_code": original_code,
                    "tilelang_code": tilelang_code,
                    "type": "example",
                    "metadata": metadata,
                }
            )

            print(f"Loaded example: {example_id}")

        return examples

    def process_documentation(self, docs_dir: str) -> List[Dict]:
        """Process TileLang documentation files."""
        docs = []

        for file_path in glob.glob(f"{docs_dir}/**/*.md", recursive=True):
            with open(file_path, "r") as f:
                content = f.read()

            # Extract code blocks and explanations
            docs.append(
                {
                    "id": Path(file_path).stem,
                    "file_path": file_path,
                    "content": content,
                    "type": "documentation",
                    "metadata": {
                        "topic": self._extract_topic(file_path),
                        "code_blocks": self._extract_code_blocks(content),
                    },
                }
            )

        return docs

    def _extract_metadata(self, content: str) -> Dict:
        """Extract metadata from code content."""
        metadata = {}

        # Extract operation types (matmul, softmax, etc.)
        if "matmul" in content.lower():
            metadata["operation"] = "matmul"
        elif "softmax" in content.lower():
            metadata["operation"] = "softmax"
        # Add more patterns as needed

        # Extract optimization techniques used
        if "tile" in content:
            metadata["techniques"] = metadata.get("techniques", []) + ["tiling"]
        if "vectorize" in content:
            metadata["techniques"] = metadata.get("techniques", []) + ["vectorization"]

        return metadata

    def _extract_topic(self, file_path: str) -> str:
        """Extract topic from file path."""
        return Path(file_path).parent.name

    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from markdown."""
        import re

        code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", content, re.DOTALL)
        return code_blocks


# ============= Step 2: Vector Store Setup =============


class TileLangVectorStore:
    """Manage vector storage for TileLang documents."""

    def __init__(self, collection_name: str = "tilelang_examples"):
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.create_collection(
            name=collection_name, embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store."""
        for i, doc in enumerate(documents):
            # Create searchable text
            searchable_text = self._create_searchable_text(doc)

            # Store the actual code in metadata for retrieval
            metadata = doc["metadata"].copy()
            metadata["doc_type"] = doc["type"]
            metadata["id"] = doc["id"]

            # For examples, store paths to the actual files
            if doc["type"] == "example":
                metadata["dir_path"] = doc["dir_path"]

            self.collection.add(
                documents=[searchable_text], metadatas=[metadata], ids=[f"{doc['type']}_{doc['id']}_{i}"]
            )

    def _create_searchable_text(self, doc: Dict) -> str:
        """Create searchable text from document."""
        if doc["type"] == "example":
            # Include both original and optimized code snippets
            original_snippet = doc.get("original_code", "")[:300]
            optimized_snippet = doc.get("tilelang_code", "")[:300]
            techniques = doc["metadata"].get("optimization_techniques", [])

            return f"""PyTorch to TileLang optimization example
Operation: {doc['metadata'].get('operation', 'unknown')}
Problem type: {doc['metadata'].get('problem_type', 'general')}
Techniques: {', '.join(techniques)}
Key insights: {' '.join(doc['metadata'].get('key_insights', []))}
Original: {original_snippet}
Optimized: {optimized_snippet}"""
        else:
            return doc["content"]

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents."""
        results = self.collection.query(query_texts=[query], n_results=k)
        return results


# ============= Step 3: DSPy RAG Setup =============


class TileLangRetriever(dspy.Retrieve):
    """Custom retriever for TileLang examples."""

    def __init__(self, vector_store: TileLangVectorStore, k: int = 3):
        self.vector_store = vector_store
        self.k = k
        super().__init__(k=k)

    def forward(self, query: str) -> List[dspy.Example]:
        """Retrieve relevant examples."""
        results = self.vector_store.search(query, k=self.k)

        examples = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]

            # If it's an example, load the actual code files
            if metadata.get("doc_type") == "example" and "dir_path" in metadata:
                example_dir = Path(metadata["dir_path"])

                # Read the actual files
                original_code = ""
                tilelang_code = ""

                original_path = example_dir / "original.py"
                tilelang_path = example_dir / "tilelang.py"

                if original_path.exists():
                    with open(original_path, "r") as f:
                        original_code = f.read()

                if tilelang_path.exists():
                    with open(tilelang_path, "r") as f:
                        tilelang_code = f.read()

                example = dspy.Example(
                    document=results["documents"][0][i],
                    original_code=original_code,
                    tilelang_code=tilelang_code,
                    metadata=metadata,
                    relevance_score=results["distances"][0][i] if "distances" in results else 1.0,
                )
            else:
                example = dspy.Example(
                    document=results["documents"][0][i],
                    metadata=metadata,
                    relevance_score=results["distances"][0][i] if "distances" in results else 1.0,
                )

            examples.append(example)

        return examples


# ============= Step 4: DSPy Signatures =============


class AnalyzePyTorch(dspy.Signature):
    """Analyze PyTorch code to understand optimization opportunities."""

    pytorch_code = dspy.InputField(desc="PyTorch model code to analyze")
    analysis = dspy.OutputField(
        desc="Analysis of the PyTorch code including operations and optimization opportunities"
    )


class GenerateTileLang(dspy.Signature):
    """Generate optimized TileLang implementation."""

    pytorch_code = dspy.InputField(desc="Original PyTorch model code")
    analysis = dspy.InputField(desc="Analysis of optimization opportunities")
    examples = dspy.InputField(desc="Relevant TileLang examples")
    tilelang_code = dspy.OutputField(desc="Optimized TileLang implementation")


# ============= Step 5: DSPy Module =============


class TileLangOptimizer(dspy.Module):
    """Main DSPy module for TileLang optimization."""

    def __init__(self, retriever: TileLangRetriever, num_examples: int = 3):
        super().__init__()
        self.retriever = retriever
        self.analyze = dspy.ChainOfThought(AnalyzePyTorch)
        self.generate = dspy.ChainOfThought(GenerateTileLang)
        self.num_examples = num_examples

    def forward(self, pytorch_code: str) -> dspy.Prediction:
        """Generate TileLang optimization for PyTorch code."""

        # Step 1: Analyze the PyTorch code
        analysis = self.analyze(pytorch_code=pytorch_code)

        # Step 2: Retrieve relevant examples
        # Create a query based on the analysis
        query = f"{pytorch_code[:200]} {analysis.analysis[:200]}"
        examples = self.retriever(query)

        # Format examples for the prompt
        formatted_examples = self._format_examples(examples[: self.num_examples])

        # Step 3: Generate TileLang code
        result = self.generate(
            pytorch_code=pytorch_code, analysis=analysis.analysis, examples=formatted_examples
        )

        return result

    def _format_examples(self, examples: List[dspy.Example]) -> str:
        """Format retrieved examples for the prompt."""
        formatted = []
        for i, ex in enumerate(examples):
            if hasattr(ex, "original_code") and hasattr(ex, "tilelang_code"):
                formatted.append(
                    f"""Example {i+1} - {ex.metadata.get('problem_type', 'Unknown')}:

Original PyTorch Code:
```python
{ex.original_code}
```

Optimized TileLang Code:
```python
{ex.tilelang_code}
```

Optimization Techniques: {', '.join(ex.metadata.get('optimization_techniques', []))}
"""
                )
            else:
                formatted.append(f"Example {i+1}:\n{ex.document}\n")

        return "\n".join(formatted)


# ============= Step 6: Main Pipeline =============


def setup_tilelang_pipeline(
    examples_dir: str, docs_dir: str, openai_api_key: str = None, model: str = "gpt-4"
) -> TileLangOptimizer:
    """Set up the complete TileLang optimization pipeline."""

    # Configure DSPy with OpenAI
    if openai_api_key:
        lm = dspy.OpenAI(model=model, api_key=openai_api_key)
    else:
        # Use environment variable
        lm = dspy.OpenAI(model=model)

    dspy.settings.configure(lm=lm)

    # Process documents
    print("Processing documents...")
    processor = TileLangDocumentProcessor(base_path=".")
    examples = processor.process_tilelang_examples(examples_dir)
    docs = processor.process_documentation(docs_dir)

    # Set up vector store
    print("Setting up vector store...")
    vector_store = TileLangVectorStore()
    vector_store.add_documents(examples + docs)

    # Create retriever
    retriever = TileLangRetriever(vector_store, k=3)

    # Create optimizer
    optimizer = TileLangOptimizer(retriever, num_examples=3)

    return optimizer


# ============= Step 7: Usage Example =============


def main():
    """Example usage of the TileLang optimization pipeline."""

    # Set up the pipeline
    optimizer = setup_tilelang_pipeline(
        examples_dir="../TileLang/examples",
        docs_dir="../TileLang/docs",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
    )

    # Example PyTorch code to optimize
    pytorch_code = open("../KernelBench/KernelBench/level1/19_ReLU.py").read()

    # Generate optimization
    result = optimizer(pytorch_code)

    print("Analysis:", result.analysis)
    print("\nGenerated TileLang Code:")
    print(result.tilelang_code)

    # Optionally, compile with DSPy optimizers
    # This requires a validation dataset
    if False:  # Set to True if you have validation data
        from dspy.teleprompt import BootstrapFewShot

        # Create training examples
        trainset = [
            dspy.Example(pytorch_code="...", tilelang_code="...").with_inputs("pytorch_code")
            # Add more examples
        ]

        # Optimize
        teleprompter = BootstrapFewShot(metric=tilelang_correctness_metric)
        optimized_optimizer = teleprompter.compile(optimizer, trainset=trainset)


def tilelang_correctness_metric(example, pred, trace=None):
    """Metric for evaluating TileLang generation quality."""
    # Implement your evaluation logic here
    # Could check for:
    # - Syntax correctness
    # - Presence of required class structure
    # - Use of TileLang primitives
    return 1.0  # Placeholder


if __name__ == "__main__":
    main()

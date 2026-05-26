#!/usr/bin/env python3
"""Create sample parquet data for training tests."""
import os
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = "runs/data"
os.makedirs(DATA_DIR, exist_ok=True)

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming artificial intelligence.",
    "Deep learning models achieve remarkable results.",
    "The diffusion process allows iterative refinement.",
    "Attention mechanisms focus on relevant parts.",
    "Gradient descent optimizes neural network weights.",
    "Transformers dominate modern natural language processing.",
    "Tokenization is crucial in NLP pipelines.",
    "Parallel processing enables faster inference.",
    "Large language models show emergent capabilities.",
    "The model was trained on a large corpus of text data.",
    "Neural networks learn hierarchical representations.",
    "Backpropagation updates weights through the network.",
    "Embedding layers map discrete tokens to vectors.",
    "The loss function guides the optimization process.",
    "Batch normalization stabilizes training dynamics.",
    "Dropout prevents overfitting in deep networks.",
    "Learning rate scheduling improves convergence.",
    "Multi-head attention captures diverse dependencies.",
    "The encoder-decoder architecture enables sequence translation.",
]

for i in range(3):
    filepath = os.path.join(DATA_DIR, f"data_{i}.parquet")
    table = pa.table({"text": sentences})
    pq.write_table(table, filepath)
    print(f"Created {filepath}: {len(sentences)} rows")

print("Done!")

"""Generate sample parquet files for the diffusion training script."""

import os
import pyarrow as pa
import pyarrow.parquet as pq

# Directory for parquet files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Sample text data about AI/ML
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world of artificial intelligence.",
    "Deep learning models have achieved remarkable results in NLP tasks.",
    "The diffusion process allows for iterative refinement of generated sequences.",
    "Attention mechanisms enable models to focus on relevant parts of input.",
    "Gradient descent is the optimization algorithm used to train neural networks.",
    "Transformers have become the dominant architecture in modern NLP.",
    "Tokenization is a crucial step in natural language processing pipelines.",
    "The transformer architecture enables parallel processing of sequences.",
    "Large language models have demonstrated emergent capabilities.",
    "Neural networks learn representations through backpropagation.",
    "Recurrent neural networks process sequential data with hidden states.",
    "Convolutional neural networks excel at image recognition tasks.",
    "Generative adversarial networks produce realistic synthetic data.",
    "Variational autoencoders learn latent representations of data.",
    "Self-attention allows every token to attend to every other token.",
    "Positional encodings help models understand token order in sequences.",
    "Batch normalization stabilizes training of deep neural networks.",
    "Dropout is a regularization technique that prevents overfitting.",
    "The encoder-decoder architecture is common in sequence-to-sequence tasks.",
    "Beam search is used for decoding in language generation tasks.",
    "Curriculum learning gradually increases task difficulty during training.",
    "Transfer learning allows models to leverage knowledge from other domains.",
    "Multi-head attention computes attention in parallel across different subspaces.",
    "The softmax function converts logits into probability distributions.",
    "Cross-entropy loss is commonly used for classification tasks.",
    "Adam optimizer combines momentum and adaptive learning rates.",
    "Layer normalization normalizes activations across features.",
    "Flash attention optimizes the attention computation for efficiency.",
    "Semi-supervised learning leverages both labeled and unlabeled data.",
]


def main():
    num_files = 5
    rows_per_file = len(SAMPLE_TEXTS)

    print(f"Creating {num_files} parquet files in {DATA_DIR}")
    print(f"Each file has {rows_per_file} rows")

    for i in range(num_files):
        filepath = os.path.join(DATA_DIR, f"train_{i:05d}.parquet")
        
        # Each file gets the full set of texts
        table = pa.table({"text": SAMPLE_TEXTS})
        pq.write_table(table, filepath)
        print(f"  Created {filepath} ({len(SAMPLE_TEXTS)} rows)")

    print(f"\nTotal parquet files: {num_files}")
    print("Sample data generation complete!")


if __name__ == "__main__":
    main()

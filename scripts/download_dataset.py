"""
Download a real dataset from HuggingFace and convert to parquet for training.

Downloads FineWeb sample articles, splits into train/val shards, and saves
as parquet files in data/ — the format expected by the existing dataloader.

Usage:
    # Download 50,000 FineWeb articles (~80MB)
    python -m scripts.download_dataset --num-examples 50000

    # Download 10,000 articles and keep existing data
    python -m scripts.download_dataset --num-examples 10000 --replace

    # Use a different dataset (must have a 'text' column)
    python -m scripts.download_dataset --hf-dataset HuggingFaceFW/fineweb \
        --hf-config sample-10BT --num-examples 50000
"""

import os
import sys
import argparse
import math
import pyarrow.parquet as pq
import pyarrow as pa

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat_diffusion.common import print0, get_base_dir


def download_and_convert(
    hf_dataset: str = "HuggingFaceFW/fineweb",
    hf_config: str = "sample-10BT",
    num_examples: int = 50000,
    output_dir: str = "",
    replace: bool = False,
    shard_size: int = 10000,
    val_split: float = 0.05,
):
    """
    Download a HuggingFace dataset and convert to parquet shards.

    Args:
        hf_dataset: HuggingFace dataset name
        hf_config: Dataset config/split name
        num_examples: Number of examples to download
        output_dir: Where to save parquet files (default: data/ in project root)
        replace: If True, delete existing parquet files first
        shard_size: Number of examples per parquet file
        val_split: Fraction of examples to use as validation
    """
    from datasets import load_dataset

    if not output_dir:
        # Default to data/ in the project root
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "data")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if replace:
        existing = [f for f in sorted(os.listdir(output_dir)) if f.endswith('.parquet')]
        if existing:
            print0(f"Removing {len(existing)} existing parquet files...")
            for f in existing:
                os.remove(os.path.join(output_dir, f))

    num_val = max(1, int(num_examples * val_split))
    num_train = num_examples - num_val
    num_train_shards = math.ceil(num_train / shard_size) if num_train > 0 else 0
    num_val_shards = math.ceil(num_val / shard_size) if num_val > 0 else 0
    total_shards = num_train_shards + num_val_shards

    print0(f"Downloading {num_examples:,} examples from {hf_dataset} ({hf_config})")
    print0(f"  Train: {num_train:,} examples -> {num_train_shards} shards")
    print0(f"  Val:   {num_val:,} examples   -> {num_val_shards} shards")
    print0(f"  Output: {output_dir}")
    print0(f"  Replace: {replace}")
    print0()

    # Load dataset with streaming
    ds = load_dataset(hf_dataset, hf_config, split="train", streaming=True)
    if num_examples:
        ds = ds.take(num_examples)

    examples_written = 0
    train_shard_idx = 0
    val_shard_idx = 0
    current_train = []
    current_val = []

    for i, example in enumerate(ds):
        text = example.get("text", "")
        if not text:
            continue

        if i < num_train:
            current_train.append({"text": text})
            if len(current_train) >= shard_size:
                shard_name = f"train_{train_shard_idx:05d}.parquet"
                shard_path = os.path.join(output_dir, shard_name)
                table = pa.Table.from_pylist(current_train)
                pq.write_table(table, shard_path)
                examples_written += len(current_train)
                pct = 100.0 * (i + 1) / num_examples
                print0(f"  [{pct:5.1f}%] Wrote {shard_name} ({len(current_train)} examples)")
                current_train = []
                train_shard_idx += 1
        else:
            current_val.append({"text": text})
            if len(current_val) >= shard_size:
                shard_name = f"val_{val_shard_idx:05d}.parquet"
                shard_path = os.path.join(output_dir, shard_name)
                table = pa.Table.from_pylist(current_val)
                pq.write_table(table, shard_path)
                examples_written += len(current_val)
                pct = 100.0 * (i + 1) / num_examples
                print0(f"  [{pct:5.1f}%] Wrote {shard_name} ({len(current_val)} examples)")
                current_val = []
                val_shard_idx += 1

    # Write remaining batches
    if current_train:
        shard_name = f"train_{train_shard_idx:05d}.parquet"
        shard_path = os.path.join(output_dir, shard_name)
        table = pa.Table.from_pylist(current_train)
        pq.write_table(table, shard_path)
        examples_written += len(current_train)
        print0(f"  [100.0%] Wrote {shard_name} ({len(current_train)} examples)")
    if current_val:
        shard_name = f"val_{val_shard_idx:05d}.parquet"
        shard_path = os.path.join(output_dir, shard_name)
        table = pa.Table.from_pylist(current_val)
        pq.write_table(table, shard_path)
        examples_written += len(current_val)
        print0(f"  [100.0%] Wrote {shard_name} ({len(current_val)} examples)")

    print0()
    print0(f"Done! {examples_written:,} examples written to {output_dir}")
    print0()

    # Show what's in the directory now
    files = sorted(f for f in os.listdir(output_dir) if f.endswith('.parquet'))
    train_files = [f for f in files if f.startswith('train_')]
    val_files = [f for f in files if f.startswith('val_')]
    print0(f"Parquet files: {len(train_files)} train + {len(val_files)} val")
    if train_files:
        sizes = [os.path.getsize(os.path.join(output_dir, f)) for f in train_files]
        total_mb = sum(sizes) / (1024 * 1024)
        print0(f"  Train: {len(train_files)} files, {total_mb:.1f} MB total")
    if val_files:
        sizes = [os.path.getsize(os.path.join(output_dir, f)) for f in val_files]
        total_mb = sum(sizes) / (1024 * 1024)
        print0(f"  Val:   {len(val_files)} files, {total_mb:.1f} MB total")

    # Print quick-start training command
    print0()
    print0("Now train with:")
    print0(f"  python -m scripts.diffusion_train --device-type=cuda \\")
    print0(f"      --depth=8 --max-seq-len=1024 --device-batch-size=16 \\")
    print0(f"      --num-iterations=1000 --lr=3e-4 --warmup-iters=100 \\")
    print0(f"      --save-every=500")
    print0()
    print0("=" * 60)
    print0("Training BPE tokenizer from downloaded data...")
    print0("=" * 60)

    # Train BPE tokenizer on the downloaded text
    train_bpe_from_parquet(output_dir)


def train_bpe_from_parquet(data_dir: str, vocab_size: int = 4094):
    """
    Train a BPE tokenizer from parquet files in data_dir.

    Saves tokenizer.json to <data_dir>/tokenizer_diffusion/.
    """
    import glob
    import time
    from nanochat_diffusion.tokenizer import Tokenizer

    # Collect all train parquet files
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "train_*.parquet")))
    if not parquet_files:
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

    if not parquet_files:
        print0("No parquet files found for BPE training. Skipping.")
        return

    print0(f"Reading {len(parquet_files)} parquet files for BPE training...")

    def text_iterator():
        for path in parquet_files:
            table = pq.read_table(path, columns=["text"])
            for batch in table.to_batches():
                texts = batch.column("text").to_pylist()
                for t in texts:
                    if t:
                        yield t

    # Create tokenizer and train
    t0 = time.time()
    tok = Tokenizer(data_dir="", verbose=False)
    tok.train(text_iterator(), vocab_size=vocab_size)

    # Save
    save_dir = os.path.join(data_dir, "tokenizer_diffusion")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tokenizer.json")
    tok.save(save_path)

    elapsed = time.time() - t0
    print0(f"BPE tokenizer trained in {elapsed:.1f}s")
    print0(f"  Vocab size: {tok._hf.get_vocab_size()}")
    print0(f"  Saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real dataset for training")
    parser.add_argument("--hf-dataset", type=str, default="HuggingFaceFW/fineweb",
                        help="HuggingFace dataset name")
    parser.add_argument("--hf-config", type=str, default="sample-10BT",
                        help="Dataset config/split")
    parser.add_argument("--num-examples", type=int, default=50000,
                        help="Number of examples to download")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory (default: data/)")
    parser.add_argument("--replace", action="store_true",
                        help="Delete existing parquet files first")
    parser.add_argument("--shard-size", type=int, default=10000,
                        help="Examples per parquet shard")
    args = parser.parse_args()

    download_and_convert(
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        num_examples=args.num_examples,
        output_dir=args.output_dir,
        replace=args.replace,
        shard_size=args.shard_size,
    )

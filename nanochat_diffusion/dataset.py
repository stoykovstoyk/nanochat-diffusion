"""
Dataset utilities for diffusion LLM.

Handles downloading, reading, and preprocessing datasets for training.
"""

import os
import json
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    path: str
    split: str = "train"
    max_seq_len: int = 1024
    vocab_size: int = 32768
    tokenizer_path: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatasetLoader:
    """
    General-purpose dataset loader.
    
    Supports:
    - Loading from parquet files
    - Loading from JSONL files
    - Loading from HuggingFace datasets
    - Custom dataset formatting
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = []
        self.metadata = {}
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset based on configuration."""
        # Load from parquet if available
        if self.config.path.endswith('.parquet'):
            self._load_parquet()
        # Load from JSONL
        elif self.config.path.endswith('.jsonl'):
            self._load_jsonl()
        # Load from CSV
        elif self.config.path.endswith('.csv'):
            self.__load_csv()
        # Load from HuggingFace
        elif self.config.path.startswith('hf:'):
            self._load_hf()
        # Try as directory
        elif os.path.isdir(self.config.path):
            self._load_directory()
        # Default: create synthetic data
        else:
            self._create_synthetic()
    
    def _load_parquet(self):
        """Load dataset from parquet files."""
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(self.config.path)
            table = pf.read()
            self.data = table.to_pydict()
        except ImportError:
            print("pyarrow not installed, creating synthetic dataset")
            self._create_synthetic()
    
    def _load_jsonl(self):
        """Load dataset from JSONL file."""
        self.data = []
        with open(self.config.path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))
    
    def _load_csv(self):
        """Load dataset from CSV file."""
        import csv
        self.data = []
        with open(self.config.path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
    
    def _load_hf(self):
        """Load dataset from HuggingFace."""
        dataset_name = self.config.path[3:]  # Remove 'hf:' prefix
        try:
            from datasets import load_dataset
            self.data = load_dataset(dataset_name, split=self.config.split)
        except ImportError:
            print("datasets not installed, creating synthetic dataset")
            self._create_synthetic()
    
    def _load_directory(self):
        """Load dataset from directory (e.g., parquet files)."""
        self.data = []
        for filename in os.listdir(self.config.path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.config.path, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.data.append(json.loads(line.strip()))
    
    def _create_synthetic(self):
        """Create synthetic dataset for testing."""
        self.data = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Machine learning is transforming the world of artificial intelligence."},
            {"text": "Deep learning models have achieved remarkable results in NLP tasks."},
            {"text": "The diffusion process allows for iterative refinement of generated sequences."},
            {"text": "Attention mechanisms enable models to focus on relevant parts of input."},
            {"text": "Gradient descent is the optimization algorithm used to train neural networks."},
            {"text": "Transformers have become the dominant architecture in modern NLP."},
            {"text": "Tokenization is a crucial step in natural language processing pipelines."},
            {"text": "The transformer architecture enables parallel processing of sequences."},
            {"text": "Large language models have demonstrated emergent capabilities."},
        ] * 100
    
    def get_data(self) -> List[Dict]:
        """Get the loaded dataset."""
        return self.data
    
    def get_split(self, split: str = None) -> List[Dict]:
        """Get a specific split of the dataset."""
        if split is None:
            split = self.config.split
        return self.data
    
    def get_examples(self, n: int = 100) -> List[Dict]:
        """Get N examples from the dataset."""
        return self.data[:n]
    
    def get_text_column(self) -> List[str]:
        """Get the text column from the dataset."""
        if self.data and len(self.data) > 0:
            first_item = self.data[0]
            if isinstance(first_item, dict):
                # Try common column names
                for key in ['text', 'content', 'input', 'prompt', 'document']:
                    if key in first_item:
                        return [item.get(key, '') for item in self.data]
            # Return first column
            if first_item:
                return [str(item) for item in self.data]
        return []
    
    def format_for_training(self, texts: List[str], tokenizer=None) -> List[List[int]]:
        """Format text data for training."""
        if tokenizer is None:
            return [[0] * self.config.max_seq_len for _ in texts]
        
        tokenized = []
        for text in texts:
            tokens = tokenizer.encode(text, prepend=True)
            # Pad or truncate
            if len(tokens) < self.config.max_seq_len:
                tokens = tokens + [0] * (self.config.max_seq_len - len(tokens))
            else:
                tokens = tokens[:self.config.max_seq_len]
            tokenized.append(tokens)
        
        return tokenized
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class DatasetManager:
    """
    Manages multiple datasets for training and evaluation.
    """
    
    def __init__(self):
        self.datasets = {}
        self.metadata = {}
    
    def add_dataset(self, name: str, config: DatasetConfig):
        """Add a dataset to the manager."""
        loader = DatasetLoader(config)
        self.datasets[name] = loader
        self.metadata[name] = {
            'path': config.path,
            'split': config.split,
            'max_seq_len': config.max_seq_len,
        }
    
    def get_dataset(self, name: str) -> DatasetLoader:
        """Get a dataset by name."""
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found")
        return self.datasets[name]
    
    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self.datasets.keys())
    
    def get_dataset_info(self, name: str) -> Dict:
        """Get information about a dataset."""
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found")
        return self.metadata[name]


# Global dataset registry
_DATASET_MANAGER = DatasetManager()


def register_dataset(name: str, path: str, **kwargs):
    """Register a dataset for use."""
    config = DatasetConfig(
        name=name,
        path=path,
        **kwargs
    )
    _DATASET_MANAGER.add_dataset(name, config)


def get_dataset(name: str) -> DatasetLoader:
    """Get a registered dataset."""
    return _DATASET_MANAGER.get_dataset(name)


def load_dataset(name: str, path: str = "", split: str = "train", 
                 max_seq_len: int = 1024, **kwargs) -> DatasetLoader:
    """Load a dataset with given parameters."""
    config = DatasetConfig(
        name=name,
        path=path or name,
        split=split,
        max_seq_len=max_seq_len,
        **kwargs
    )
    return DatasetLoader(config)


# Default dataset registry
register_dataset("c4", "data/c4", max_seq_len=1024)
register_dataset("fineweb", "data/fineweb", max_seq_len=2048)
register_dataset("smoltalk", "data/smoltalk", max_seq_len=1024)

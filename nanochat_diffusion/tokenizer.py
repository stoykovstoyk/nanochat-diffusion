"""
Tokenizer for nanochat_diffusion.
Adapted from karpathy/nanochat.
"""

import os
import struct
import numpy as np
from nanochat_diffusion.common import download_file_with_lock, print0

# UNK token ID for diffusion masking
UNK_TOKEN_ID = 32767

class Tokenizer:
    def __init__(self, data_dir, verbose=True):
        self.data_dir = data_dir
        self.load(data_dir, verbose=verbose)

    def load(self, data_dir, vocab_size=32768, special_tokens=None):
        """Load tokenizer from data directory"""
        if special_tokens is None:
            special_tokens = ["<|endoftext|>", "<|pad|>", "<|startofmessage|>", 
                            "<|endofmessage|>", "<|python_start|>", "<|python_end|>",
                            "<|output_start|>", "<|output_end|>", "<|assistant_start|>",
                            "<|assistant_end|>", "<|user_start|>", "<|user_end|>"]
        
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.pad_id = special_tokens.index("<|pad|>") if "<|pad|>" in special_tokens else 0
        
        # Create synthetic tokenizer for diffusion model
        # In production, this would load a real tokenizer
        self.encode_special = lambda x: special_tokens.index(x) if x in special_tokens else 0
        
        # Simple byte-level tokenizer for demonstration
        self.vocab = {i: i for i in range(vocab_size)}
        self.special_id_map = {s: i for i, s in enumerate(special_tokens)}
        
        print0(f"Initialized synthetic tokenizer with vocab_size={vocab_size}")

    def encode(self, texts, prepend=False, num_threads=1):
        """Encode text to token IDs"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for text in texts:
            # Simple byte-level encoding for demonstration
            tokens = [b for b in text.encode('utf-8')]
            if prepend:
                tokens = [self.get_bos_token_id()] + tokens
            results.append(tokens)
        return results

    def decode(self, tokens):
        """Decode token IDs to text"""
        if isinstance(tokens, (list, np.ndarray)):
            return bytes(tokens).decode('utf-8', errors='ignore')
        return bytes([tokens]).decode('utf-8', errors='ignore')

    def get_bos_token_id(self):
        return 0

    def get_unk_token_id(self):
        return UNK_TOKEN_ID

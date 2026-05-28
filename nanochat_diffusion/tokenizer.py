"""
BPE tokenizer for diffusion LLM.
Uses HuggingFace tokenizers library (already a dependency).
Trained from downloaded dataset in download_dataset.py.
Saves/loads from `tokenizer.json` in the data directory.
"""

import os
import json
from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers

# Diffusion UNK sentinel (outside BPE vocab, within padded embed range)
UNK_TOKEN_ID = 4095  # for BPE vocab_size=4094; padded to 4160


class Tokenizer:
    def __init__(self, data_dir="", verbose=True):
        self.vocab_size = 4096  # 0=BOS, 1-4094=BPE, 4095=UNK
        self.bos_id = 0
        self.unk_id = UNK_TOKEN_ID
        self.data_dir = data_dir

        tokenizer_path = os.path.join(data_dir, "tokenizer.json") if data_dir else ""
        if tokenizer_path and os.path.exists(tokenizer_path):
            self._hf = HFTokenizer.from_file(tokenizer_path)
            if verbose:
                print(f"Loaded BPE tokenizer from {tokenizer_path}, vocab_size={self._hf.get_vocab_size()}")
        else:
            # Create a fresh BPE for training
            self._hf = HFTokenizer(models.BPE(unk_token=None))
            self._hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            if verbose:
                print(f"Created new BPE tokenizer (vocab to be trained via train() method)")

    def train(self, texts, vocab_size=4094):
        """Train BPE from an iterator of strings."""
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[],
            min_frequency=2,
            show_progress=True,
        )
        self._hf.train_from_iterator(texts, trainer=trainer)

    def save(self, path):
        """Save trained tokenizer to disk."""
        os.makedirs(os.path.dirname(path) if os.path.sep in path else ".", exist_ok=True)
        self._hf.save(path)

    def encode(self, texts, prepend=False, num_threads=1):
        """Encode text to token IDs using BPE.
        
        BPE produces tokens 0-(vocab_size-2), we shift by 1 so 0=BOS.
        """
        single_str = isinstance(texts, str)
        if single_str:
            texts = [texts]

        encoded = self._hf.encode_batch(texts)
        results = []
        for enc in encoded:
            tokens = [t + 1 for t in enc.ids]  # shift: 0->1, 1->2, ...
            if prepend:
                tokens = [self.bos_id] + tokens
            results.append(tokens)

        if single_str:
            return results[0]
        return results

    def decode(self, tokens):
        """Decode token IDs back to text.
        
        Unshifts by 1 and decodes via BPE. Handles BOS at position 0.
        """
        if isinstance(tokens, (list,)):
            # Strip leading BOS if present
            if tokens and tokens[0] == self.bos_id:
                tokens = tokens[1:]
            # Unshift: our tokens are BPE+1
            bpe_ids = [max(0, t - 1) for t in tokens]
            return self._hf.decode(bpe_ids)
        return self._hf.decode([max(0, tokens - 1)])

    def get_bos_token_id(self):
        return self.bos_id

    def get_unk_token_id(self):
        return self.unk_id

    def get_vocab_size(self):
        return self.vocab_size

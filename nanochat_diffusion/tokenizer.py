"""
BPE tokenizer for diffusion LLM.
Uses HuggingFace tokenizers library (already a dependency).
Trained from downloaded dataset in download_dataset.py.
Saves/loads from `tokenizer.json` in the data directory.
"""

import os
import json
from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders

# Diffusion UNK sentinel (outside BPE vocab, within padded embed range)
UNK_TOKEN_ID = 32769  # outside BPE vocab (0-32767), padded to 32832


class Tokenizer:
    def __init__(self, data_dir="", verbose=True):
        self.vocab_size = UNK_TOKEN_ID + 1  # 32770: 0=BOS, 1-32768=BPE, 32769=UNK
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
            self._hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self._hf.decoder = decoders.ByteLevel()

    def train(self, texts, vocab_size=32768):
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
        
        Unshifts by 1 and decodes via BPE. Strips exactly one leading space
        added by ByteLevel pretokenizer (add_prefix_space=True).
        """
        if isinstance(tokens, (list,)):
            if tokens and tokens[0] == self.bos_id:
                tokens = tokens[1:]
            bpe_ids = [max(0, t - 1) for t in tokens]
            text = self._hf.decode(bpe_ids)
            # ByteLevel adds one leading space; strip it
            if len(text) > 0 and text[0] == ' ':
                text = text[1:]
            return text
        text = self._hf.decode([max(0, tokens - 1)])
        if len(text) > 0 and text[0] == ' ':
            text = text[1:]
        return text

    def get_bos_token_id(self):
        return self.bos_id

    def get_unk_token_id(self):
        return self.unk_id

    def get_vocab_size(self):
        return self.vocab_size

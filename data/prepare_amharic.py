"""
Amharic text data pipeline.

Downloads Amharic Wikipedia, tokenizes with the RL tokenizer,
and creates train/val splits as numpy arrays.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("data")


class AmharicRLTokenizer:
    """
    Greedy longest-match tokenizer using the RL-trained Amharic vocabulary.
    
    Loads vocab.txt (one subword per line) and performs greedy
    left-to-right longest-match tokenization.
    """
    
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    
    def __init__(self, vocab_path: str, config_path: Optional[str] = None):
        self.vocab: List[str] = []
        self.token_to_id = {}
        
        # Reserve special tokens
        self.vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        for i, tok in enumerate(self.vocab):
            self.token_to_id[tok] = i
        
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.rstrip("\n")
                if token and token not in self.token_to_id:
                    self.token_to_id[token] = len(self.vocab)
                    self.vocab.append(token)
        
        # Sort by length descending for greedy matching
        self._sorted_tokens = sorted(
            [(t, i) for t, i in self.token_to_id.items() 
             if i >= 4],  # Skip special tokens
            key=lambda x: len(x[0]),
            reverse=True,
        )
        
        self.vocab_size = len(self.vocab)
        logger.info(f"Loaded RL tokenizer: {self.vocab_size} tokens")
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Tokenize text using greedy longest-match."""
        ids = []
        if add_special:
            ids.append(self.BOS_ID)
        
        i = 0
        while i < len(text):
            matched = False
            for token, token_id in self._sorted_tokens:
                if text[i:i+len(token)] == token:
                    ids.append(token_id)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                # Single character fallback
                char = text[i]
                if char in self.token_to_id:
                    ids.append(self.token_to_id[char])
                else:
                    ids.append(self.UNK_ID)
                i += 1
        
        if add_special:
            ids.append(self.EOS_ID)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for i in ids:
            if 0 <= i < len(self.vocab):
                tok = self.vocab[i]
                if tok not in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"):
                    tokens.append(tok)
        return "".join(tokens)


def download_amharic_data(cache_dir: str = "data/raw") -> str:
    """Download Amharic Wikipedia from HuggingFace."""
    from datasets import load_dataset
    
    logger.info("Downloading Amharic Wikipedia...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.am",
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    logger.info(f"Downloaded {len(ds)} articles")
    
    # Extract text
    texts = []
    total_chars = 0
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) > 50:  # Skip very short articles
            texts.append(text.strip())
            total_chars += len(text)
    
    logger.info(f"Filtered to {len(texts)} articles, {total_chars:,} characters")
    
    # Save raw text
    os.makedirs(cache_dir, exist_ok=True)
    raw_path = os.path.join(cache_dir, "amharic_wiki.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n\n")
    
    logger.info(f"Saved raw text to {raw_path}")
    return raw_path


def tokenize_corpus(
    text_path: str,
    tokenizer: AmharicRLTokenizer,
    output_dir: str = "data/tokenized",
    val_ratio: float = 0.05,
) -> dict:
    """Tokenize the corpus and save as numpy arrays."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Tokenizing {text_path}...")
    
    all_ids = []
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Tokenize in chunks to show progress
    chunk_size = 100_000
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        ids = tokenizer.encode(chunk, add_special=False)
        all_ids.extend(ids)
        if (i // chunk_size) % 10 == 0:
            logger.info(f"  Tokenized {i:,}/{len(text):,} chars ({len(all_ids):,} tokens)")
    
    logger.info(f"Total: {len(all_ids):,} tokens from {len(text):,} characters")
    logger.info(f"Compression ratio: {len(text) / len(all_ids):.2f} chars/token")
    
    # Split train/val
    all_ids = np.array(all_ids, dtype=np.int32)
    split_idx = int(len(all_ids) * (1 - val_ratio))
    
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]
    
    # Save
    train_path = os.path.join(output_dir, "train_tokens.npy")
    val_path = os.path.join(output_dir, "val_tokens.npy")
    np.save(train_path, train_ids)
    np.save(val_path, val_ids)
    
    stats = {
        "total_tokens": len(all_ids),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "vocab_size": tokenizer.vocab_size,
        "compression_ratio": len(text) / len(all_ids),
    }
    
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved train: {train_path} ({len(train_ids):,} tokens)")
    logger.info(f"Saved val:   {val_path} ({len(val_ids):,} tokens)")
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", default="tokenizer", help="Path to RL tokenizer directory")
    parser.add_argument("--output-dir", default="data/tokenized", help="Output directory for tokenized data")
    parser.add_argument("--cache-dir", default="data/raw", help="Cache for raw downloads")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AmharicRLTokenizer(
        vocab_path=os.path.join(args.tokenizer_dir, "vocab.txt"),
        config_path=os.path.join(args.tokenizer_dir, "config.json"),
    )
    
    # Download data
    raw_path = download_amharic_data(cache_dir=args.cache_dir)
    
    # Tokenize
    stats = tokenize_corpus(
        text_path=raw_path,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
    )
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"  Vocab size:     {stats['vocab_size']:,}")
    print(f"  Train tokens:   {stats['train_tokens']:,}")
    print(f"  Val tokens:     {stats['val_tokens']:,}")
    print(f"  Compression:    {stats['compression_ratio']:.2f} chars/token")
    print("=" * 60)


if __name__ == "__main__":
    main()

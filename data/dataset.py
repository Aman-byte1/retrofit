"""
PyTorch Dataset for tokenized Amharic LM training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenizedLMDataset(Dataset):
    """
    Language modeling dataset from pre-tokenized numpy arrays.
    
    Returns (input_ids, target_ids) pairs of fixed sequence length,
    where target_ids = input_ids shifted by 1.
    """
    
    def __init__(self, tokens_path: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.tokens = np.load(tokens_path).astype(np.int64)
        
        # Number of complete sequences we can form
        self.n_sequences = (len(self.tokens) - 1) // seq_len
        # Trim to exact multiple
        self.tokens = self.tokens[:self.n_sequences * seq_len + 1]
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        x = torch.from_numpy(self.tokens[start:end].copy())
        y = torch.from_numpy(self.tokens[start+1:end+1].copy())
        
        return x, y

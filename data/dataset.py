"""
PyTorch Dataset & DataLoader utilities for tokenized language model training.
Supports raw binary (.bin) memory mapping and numpy (.npy) arrays.
"""

import os
from typing import Optional, Union, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MemmapDataset(Dataset):
    """
    High-performance memory-mapped language modeling dataset.
    Loads .bin (raw uint16/int32/int64) or .npy files with zero memory copying.
    
    Yields (x, y) where y = x shifted by 1.
    """

    def __init__(
        self,
        file_path: str,
        seq_len: int = 512,
        dtype: np.dtype = np.uint16,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.file_path = file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        if file_path.endswith(".npy"):
            self.data = np.load(file_path, mmap_mode="r")
        else:
            # Assume raw binary file (.bin)
            file_size_bytes = os.path.getsize(file_path)
            element_size = np.dtype(dtype).itemsize
            num_elements = file_size_bytes // element_size
            self.data = np.memmap(file_path, dtype=dtype, mode="r", shape=(num_elements,))

        self.num_sequences = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len

        x_np = np.array(self.data[start:end], dtype=np.int64)
        y_np = np.array(self.data[start + 1 : end + 1], dtype=np.int64)

        return torch.from_numpy(x_np), torch.from_numpy(y_np)


# Alias for backward compatibility
TokenizedLMDataset = MemmapDataset


class InfiniteDataLoader:
    """Infinite generator wrapper around standard PyTorch DataLoader."""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

    def __len__(self):
        return len(self.dataloader)

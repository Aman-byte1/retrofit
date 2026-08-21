"""
Dataset loading and preparation for voice cloning experiments.

Handles:
- Loading the IWSLT evaluation dataset from HuggingFace
- Loading multi-speaker training data
- Audio preprocessing and batching

NOTE: We disable the HF datasets audio decoder entirely and decode
audio manually with soundfile, to avoid the torchcodec dependency.
"""

import io
import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def _decode_audio_bytes(audio_field: Dict) -> Tuple[np.ndarray, int]:
    """
    Manually decode audio from HF datasets' raw format.
    
    When audio columns are loaded with decode=False, each entry is:
      {"bytes": b"...", "path": "filename.wav"}
    
    We decode the bytes with soundfile.
    """
    raw_bytes = audio_field.get("bytes")
    if raw_bytes is None:
        raise ValueError("Audio field has no 'bytes' key")
    
    audio_array, sr = sf.read(io.BytesIO(raw_bytes))
    
    # Ensure mono
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    
    return audio_array.astype(np.float32), int(sr)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    waveform = torch.from_numpy(audio).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    resampled = resampler(waveform).squeeze(0).numpy()
    return resampled


class IWSLTEvalDataset(Dataset):
    """
    Load the IWSLT 2026 voice cloning evaluation dataset.
    
    Dataset: amanuelbyte/omnivoice-best-of-n-dev-eval
    Contains: ref_audio, best_audio, text, language, best_model, best_score
    """
    
    def __init__(
        self,
        language: Optional[str] = None,
        target_sr: int = 24000,
        max_samples: Optional[int] = None,
    ):
        from datasets import load_dataset, Audio
        
        logger.info("Loading IWSLT evaluation dataset...")
        self.ds = load_dataset(
            "amanuelbyte/omnivoice-best-of-n-dev-eval",
            split="train",
        )
        
        # Disable automatic audio decoding (avoids torchcodec dependency)
        audio_columns = [c for c in self.ds.column_names
                        if isinstance(self.ds.features[c], Audio)]
        for col in audio_columns:
            self.ds = self.ds.cast_column(col, Audio(decode=False))
            logger.info(f"  Disabled auto-decode for column: {col}")
        
        # Filter by language using index-based selection (no row decoding)
        if language:
            lang_column = self.ds["language"]
            indices = [i for i, lang in enumerate(lang_column) if lang == language]
            self.ds = self.ds.select(indices)
            logger.info(f"Filtered to language={language}: {len(self.ds)} samples")
        
        # Limit samples
        if max_samples and max_samples < len(self.ds):
            self.ds = self.ds.select(range(max_samples))
        
        self.target_sr = target_sr
        logger.info(f"IWSLT eval dataset loaded: {len(self.ds)} samples")
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx) -> Dict:
        item = self.ds[idx]
        
        # Manually decode audio using soundfile
        ref_array, ref_sr = _decode_audio_bytes(item["ref_audio"])
        best_array, best_sr = _decode_audio_bytes(item["best_audio"])
        
        # Resample to target SR
        ref_array = _resample(ref_array, ref_sr, self.target_sr)
        best_array = _resample(best_array, best_sr, self.target_sr)
        
        return {
            "id": item["id"],
            "language": item["language"],
            "text": item["text"],
            "ref_audio": ref_array,
            "ref_sr": self.target_sr,
            "best_audio": best_array,
            "best_sr": self.target_sr,
            "best_model": item["best_model"],
            "best_score": item["best_score"],
        }


class MultiSpeakerTrainDataset(Dataset):
    """
    Multi-speaker training dataset.
    
    Can load from:
    - IWSLT eval dataset (for quick experiments)
    - Any HuggingFace dataset with audio + text columns
    - Local directory of audio files
    """
    
    def __init__(
        self,
        source: str = "iwslt",
        language: str = "fr",
        target_sr: int = 24000,
        max_samples: Optional[int] = None,
        min_duration: float = 1.0,
        max_duration: float = 15.0,
        train_split: float = 0.9,
        is_train: bool = True,
    ):
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        if source == "iwslt":
            self._load_from_iwslt(language, max_samples, train_split, is_train)
        elif source.startswith("local:"):
            self._load_from_local(source[6:], max_samples)
        else:
            self._load_from_hf(source, language, max_samples, train_split, is_train)
    
    def _load_from_iwslt(
        self,
        language: str,
        max_samples: Optional[int],
        train_split: float,
        is_train: bool,
    ):
        """Use the IWSLT eval dataset as training data (for quick experiments)."""
        from datasets import load_dataset, Audio
        
        logger.info("Loading IWSLT dataset as training data...")
        ds = load_dataset(
            "amanuelbyte/omnivoice-best-of-n-dev-eval",
            split="train",
        )
        
        # Disable auto audio decoding
        audio_columns = [c for c in ds.column_names
                        if isinstance(ds.features[c], Audio)]
        for col in audio_columns:
            ds = ds.cast_column(col, Audio(decode=False))
        
        if language:
            lang_column = ds["language"]
            indices = [i for i, l in enumerate(lang_column) if l == language]
            ds = ds.select(indices)
        
        # Split into train/val
        n_total = len(ds)
        n_train = int(n_total * train_split)
        
        if is_train:
            ds = ds.select(range(n_train))
        else:
            ds = ds.select(range(n_train, n_total))
        
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        
        self.samples = []
        for item in ds:
            try:
                audio, sr = _decode_audio_bytes(item["ref_audio"])
                duration = len(audio) / sr
                
                if self.min_duration <= duration <= self.max_duration:
                    self.samples.append({
                        "audio": audio,
                        "sr": sr,
                        "text": item["text"],
                        "speaker_id": item["id"][:5],  # Use prefix as pseudo speaker ID
                        "language": item["language"],
                    })
            except Exception as e:
                logger.warning(f"Failed to decode audio sample: {e}")
                continue
        
        logger.info(f"IWSLT training data: {len(self.samples)} samples")
    
    def _load_from_hf(
        self,
        dataset_name: str,
        language: str,
        max_samples: Optional[int],
        train_split: float,
        is_train: bool,
    ):
        """Load from a HuggingFace dataset."""
        from datasets import load_dataset, Audio
        
        logger.info(f"Loading HF dataset: {dataset_name} ({language})...")
        
        try:
            ds = load_dataset(dataset_name, language, split="train", streaming=False)
        except Exception:
            ds = load_dataset(dataset_name, split="train")
            if "language" in ds.column_names or "locale" in ds.column_names:
                lang_col = "language" if "language" in ds.column_names else "locale"
                lang_column = ds[lang_col]
                indices = [i for i, l in enumerate(lang_column) if l == language]
                ds = ds.select(indices)
        
        # Disable auto audio decoding for all audio columns
        audio_columns = [c for c in ds.column_names
                        if isinstance(ds.features[c], Audio)]
        for col in audio_columns:
            ds = ds.cast_column(col, Audio(decode=False))
        
        n_total = len(ds)
        n_train = int(n_total * train_split)
        
        if is_train:
            ds = ds.select(range(min(n_train, len(ds))))
        else:
            ds = ds.select(range(n_train, min(n_total, len(ds))))
        
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        
        self.samples = []
        # Determine column names
        audio_col = next(
            (c for c in ["audio", "ref_audio"] if c in ds.column_names),
            "audio"
        )
        text_col = next(
            (c for c in ["sentence", "text", "transcription"] if c in ds.column_names),
            "text"
        )
        speaker_col = next(
            (c for c in ["client_id", "speaker_id", "speaker"] if c in ds.column_names),
            None
        )
        
        for item in ds:
            try:
                audio, sr = _decode_audio_bytes(item[audio_col])
                
                duration = len(audio) / sr
                if not (self.min_duration <= duration <= self.max_duration):
                    continue
                
                self.samples.append({
                    "audio": audio,
                    "sr": sr,
                    "text": item[text_col],
                    "speaker_id": str(item[speaker_col]) if speaker_col else "unknown",
                    "language": language,
                })
            except Exception as e:
                continue
        
        logger.info(f"HF training data: {len(self.samples)} samples")
    
    def _load_from_local(self, directory: str, max_samples: Optional[int]):
        """Load from a local directory of audio files."""
        audio_dir = Path(directory)
        self.samples = []
        
        for audio_path in sorted(audio_dir.rglob("*.wav"))[:max_samples]:
            try:
                audio, sr = sf.read(str(audio_path))
                duration = len(audio) / sr
                
                if not (self.min_duration <= duration <= self.max_duration):
                    continue
                
                # Try to find corresponding text file
                text_path = audio_path.with_suffix(".txt")
                text = text_path.read_text().strip() if text_path.exists() else ""
                
                # Use parent directory as speaker ID
                speaker_id = audio_path.parent.name
                
                self.samples.append({
                    "audio": audio.astype(np.float32),
                    "sr": sr,
                    "text": text,
                    "speaker_id": speaker_id,
                    "language": "unknown",
                })
            except Exception as e:
                logger.warning(f"Failed to load {audio_path}: {e}")
        
        logger.info(f"Local training data: {len(self.samples)} samples from {directory}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]
        
        audio = sample["audio"]
        sr = sample["sr"]
        
        # Resample if needed
        audio = _resample(audio, sr, self.target_sr)
        
        return {
            "audio": audio,
            "text": sample["text"],
            "speaker_id": sample["speaker_id"],
            "language": sample["language"],
        }


def create_eval_dataloader(
    language: Optional[str] = None,
    max_samples: Optional[int] = None,
    target_sr: int = 24000,
) -> DataLoader:
    """Create an evaluation dataloader for the IWSLT benchmark."""
    dataset = IWSLTEvalDataset(
        language=language,
        target_sr=target_sr,
        max_samples=max_samples,
    )
    # No batching for eval — each sample has different audio length
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def create_train_dataloader(
    source: str = "iwslt",
    language: str = "fr",
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    target_sr: int = 24000,
    num_workers: int = 4,
    is_train: bool = True,
) -> DataLoader:
    """Create a training dataloader."""
    dataset = MultiSpeakerTrainDataset(
        source=source,
        language=language,
        target_sr=target_sr,
        max_samples=max_samples,
        is_train=is_train,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
    )

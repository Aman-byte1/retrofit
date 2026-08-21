"""
Dataset loading and preparation for voice cloning experiments.

Handles:
- Loading the IWSLT evaluation dataset from HuggingFace
- Loading multi-speaker training data
- Audio preprocessing and batching
"""

import torch
import torchaudio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


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
        from datasets import load_dataset
        
        logger.info("Loading IWSLT evaluation dataset...")
        self.ds = load_dataset(
            "amanuelbyte/omnivoice-best-of-n-dev-eval",
            split="train",
        )
        
        # Filter by language if specified
        if language:
            self.ds = self.ds.filter(lambda x: x["language"] == language)
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
        
        # Extract audio arrays
        ref_audio = self._process_audio(item["ref_audio"])
        best_audio = self._process_audio(item["best_audio"])
        
        return {
            "id": item["id"],
            "language": item["language"],
            "text": item["text"],
            "ref_audio": ref_audio["array"],
            "ref_sr": ref_audio["sr"],
            "best_audio": best_audio["array"],
            "best_sr": best_audio["sr"],
            "best_model": item["best_model"],
            "best_score": item["best_score"],
        }
    
    def _process_audio(self, audio_field: Dict) -> Dict:
        """Process an audio field from the HF dataset."""
        array = np.array(audio_field["array"], dtype=np.float32)
        sr = audio_field["sampling_rate"]
        
        # Resample if needed
        if sr != self.target_sr:
            array = self._resample(array, sr, self.target_sr)
            sr = self.target_sr
        
        return {"array": array, "sr": sr}
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(waveform).squeeze(0).numpy()
        return resampled


class MultiSpeakerTrainDataset(Dataset):
    """
    Multi-speaker training dataset for LoRA fine-tuning.
    
    Can load from:
    - Local directory with audio files organized by speaker
    - HuggingFace dataset (e.g., mozilla-foundation/common_voice_17_0)
    - The IWSLT eval dataset itself (for quick experiments)
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
        from datasets import load_dataset
        
        logger.info("Loading IWSLT dataset as training data...")
        ds = load_dataset(
            "amanuelbyte/omnivoice-best-of-n-dev-eval",
            split="train",
        )
        
        if language:
            ds = ds.filter(lambda x: x["language"] == language)
        
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
            ref = item["ref_audio"]
            audio = np.array(ref["array"], dtype=np.float32)
            sr = ref["sampling_rate"]
            duration = len(audio) / sr
            
            if self.min_duration <= duration <= self.max_duration:
                self.samples.append({
                    "audio": audio,
                    "sr": sr,
                    "text": item["text"],
                    "speaker_id": item["id"][:5],  # Use prefix as pseudo speaker ID
                    "language": item["language"],
                })
        
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
        from datasets import load_dataset
        
        logger.info(f"Loading HF dataset: {dataset_name} ({language})...")
        
        try:
            ds = load_dataset(dataset_name, language, split="train", streaming=False)
        except Exception:
            ds = load_dataset(dataset_name, split="train")
            if "language" in ds.column_names or "locale" in ds.column_names:
                lang_col = "language" if "language" in ds.column_names else "locale"
                ds = ds.filter(lambda x: x[lang_col] == language)
        
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
        audio_col = "audio" if "audio" in ds.column_names else "path"
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
                if isinstance(item[audio_col], dict):
                    audio = np.array(item[audio_col]["array"], dtype=np.float32)
                    sr = item[audio_col]["sampling_rate"]
                else:
                    continue
                
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
        import soundfile as sf
        
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
        if sr != self.target_sr:
            waveform = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(waveform).squeeze(0).numpy()
        
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

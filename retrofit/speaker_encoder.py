"""
Speaker Encoder: Frozen ECAPA-TDNN for extracting speaker embeddings.

This is the first block in the retrofit pipeline:
  Reference Audio (10s) → [Speaker Encoder (frozen)] → Speaker Embedding (192-dim)

Uses SpeechBrain's pre-trained ECAPA-TDNN model trained on VoxCeleb.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import logging
from typing import Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeakerEncoder(nn.Module):
    """
    Frozen speaker verification encoder (ECAPA-TDNN).
    
    Extracts a fixed-dimensional speaker embedding from any audio clip.
    All parameters are frozen — this module is never trained.
    """
    
    EMBEDDING_DIM = 192  # ECAPA-TDNN output dimension
    EXPECTED_SR = 16000  # Expected sample rate
    
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._classifier = None
        
        # Resampler cache (created lazily per source sample rate)
        self._resamplers = {}
    
    def _load_model(self):
        """Lazy-load the speaker verification model."""
        if self._classifier is None:
            logger.info(f"Loading speaker encoder: {self.model_name}")
            from speechbrain.inference.speaker import EncoderClassifier
            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=str(Path.home() / ".cache" / "speechbrain" / "spkrec"),
                run_opts={"device": self.device},
            )
            # Freeze all parameters
            for param in self._classifier.mods.parameters():
                param.requires_grad = False
            logger.info(f"Speaker encoder loaded (frozen, {self.EMBEDDING_DIM}-dim output)")
    
    def _ensure_16khz(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Resample audio to 16kHz if needed."""
        if sr == self.EXPECTED_SR:
            return audio
        
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.EXPECTED_SR)
        
        resampler = self._resamplers[sr].to(audio.device)
        return resampler(audio)
    
    @torch.no_grad()
    def forward(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sr: int = 16000,
    ) -> torch.Tensor:
        """
        Extract speaker embedding from audio.
        
        Args:
            audio: Waveform tensor [batch, samples] or [samples], or numpy array
            sr: Sample rate of the input audio
            
        Returns:
            Speaker embedding tensor [batch, 192]
        """
        self._load_model()
        
        # Convert numpy to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure correct shape [batch, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Resample to 16kHz
        audio = self._ensure_16khz(audio, sr)
        
        # Move to device
        audio = audio.to(self.device)
        
        # Extract embedding
        embedding = self._classifier.encode_batch(audio)
        
        # Shape: [batch, 1, 192] → [batch, 192]
        embedding = embedding.squeeze(1)
        
        return embedding
    
    @torch.no_grad()
    def compute_similarity(
        self,
        audio_a: Union[torch.Tensor, np.ndarray],
        audio_b: Union[torch.Tensor, np.ndarray],
        sr: int = 16000,
    ) -> float:
        """Compute cosine similarity between two audio clips' speaker embeddings."""
        emb_a = self.forward(audio_a, sr)
        emb_b = self.forward(audio_b, sr)
        
        similarity = nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)
        return similarity.mean().item()

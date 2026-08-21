"""
Evaluation metrics for voice cloning quality.

Measures:
- CER (Character Error Rate): intelligibility via Whisper ASR
- Speaker Similarity: cosine similarity of speaker embeddings
- Combined Score: 0.5 * (1 - CER) + 0.5 * speaker_similarity (IWSLT metric)
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import soundfile as sf

logger = logging.getLogger(__name__)


class SpeakerSimilarityScorer:
    """
    Compute speaker similarity using SpeechBrain's ECAPA-TDNN model.
    
    Extracts speaker embeddings from audio and computes cosine similarity.
    """
    
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy-load the speaker verification model."""
        if self._model is None:
            logger.info(f"Loading speaker encoder: {self.model_name}")
            from speechbrain.inference.speaker import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=Path.home() / ".cache" / "speechbrain" / "spkrec",
                run_opts={"device": self.device},
            )
        return self._model
    
    def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """Extract speaker embedding from audio waveform."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # SpeechBrain expects 16kHz
        if sr != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        audio = audio.to(self.device)
        embedding = self.model.encode_batch(audio)
        return embedding.squeeze()
    
    def compute_similarity(
        self,
        audio_ref: np.ndarray,
        audio_gen: np.ndarray,
        sr: int = 16000,
    ) -> float:
        """Compute cosine similarity between two audio clips' speaker embeddings."""
        emb_ref = self.extract_embedding(audio_ref, sr)
        emb_gen = self.extract_embedding(audio_gen, sr)
        
        similarity = torch.nn.functional.cosine_similarity(
            emb_ref.unsqueeze(0), emb_gen.unsqueeze(0)
        ).item()
        
        return max(0.0, similarity)  # Clamp to [0, 1]


class CERScorer:
    """
    Compute Character Error Rate using Whisper ASR.
    
    Transcribes generated audio and compares against the reference text.
    """
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
    
    @property
    def model(self):
        """Lazy-load Whisper model."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model
    
    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio to text using Whisper."""
        # faster-whisper expects float32 numpy array at 16kHz
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        audio = audio.astype(np.float32)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        
        text = " ".join(segment.text.strip() for segment in segments)
        return text
    
    def compute_cer(
        self,
        audio: np.ndarray,
        reference_text: str,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> float:
        """Compute Character Error Rate between ASR output and reference text."""
        hypothesis = self.transcribe(audio, sr, language)
        cer = _character_error_rate(reference_text, hypothesis)
        return cer


def _character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate using edit distance.
    
    CER = (substitutions + insertions + deletions) / len(reference)
    """
    ref = reference.strip().lower()
    hyp = hypothesis.strip().lower()
    
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    
    # Dynamic programming edit distance
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )
    
    return d[len(ref)][len(hyp)] / len(ref)


def compute_combined_score(cer: float, speaker_similarity: float) -> float:
    """
    Compute the combined quality score (IWSLT 2026 metric).
    
    Combined = 0.5 * (1 - CER) + 0.5 * Speaker_Similarity
    """
    return 0.5 * (1.0 - min(cer, 1.0)) + 0.5 * speaker_similarity


class MetricsAggregator:
    """Collect and aggregate metrics across evaluation samples."""
    
    def __init__(self):
        self.records: List[Dict] = []
    
    def add(
        self,
        sample_id: str,
        language: str,
        cer: float,
        speaker_similarity: float,
        method: str = "default",
        **kwargs,
    ):
        """Add a single evaluation result."""
        self.records.append({
            "sample_id": sample_id,
            "language": language,
            "cer": cer,
            "speaker_similarity": speaker_similarity,
            "combined_score": compute_combined_score(cer, speaker_similarity),
            "method": method,
            **kwargs,
        })
    
    def summary(self, group_by: str = "method") -> Dict:
        """Compute summary statistics grouped by a field."""
        import pandas as pd
        
        if not self.records:
            return {}
        
        df = pd.DataFrame(self.records)
        
        summary = {}
        for group_val, group_df in df.groupby(group_by):
            summary[group_val] = {
                "count": len(group_df),
                "avg_cer": group_df["cer"].mean(),
                "std_cer": group_df["cer"].std(),
                "avg_speaker_sim": group_df["speaker_similarity"].mean(),
                "std_speaker_sim": group_df["speaker_similarity"].std(),
                "avg_combined": group_df["combined_score"].mean(),
                "std_combined": group_df["combined_score"].std(),
            }
        
        return summary
    
    def summary_by_language(self) -> Dict:
        """Compute summary statistics grouped by language."""
        return self.summary(group_by="language")
    
    def to_dataframe(self):
        """Convert records to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.records)
    
    def save(self, path: str):
        """Save results to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} evaluation results to {path}")

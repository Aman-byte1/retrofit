"""
RetrofitVoiceCloner: The complete retrofit architecture.

Adds zero-shot voice cloning to TTS models that DON'T have it.

Architecture (matching the diagram):
    ┌──────────────┐
    │   Speaker    │
    │   Encoder    │──► Speaker Embedding (192-dim)
    │  (frozen)    │        │
    └──────────────┘        ▼
                     ┌──────────────┐
                     │   Adapter     │  ◄── ONLY this is trained
                     │   Layers      │
                     └──────┬───────┘
                            │
    ┌──────────────┐        ▼
    │   Existing   │◄── Injected speaker conditioning (FiLM)
    │   TTS Model  │
    │  (frozen)    │
    └──────┬───────┘
           ▼
      Cloned Speech

Base TTS: MMS-TTS (Meta's single-speaker VITS, supports 1000+ languages)
Speaker Encoder: ECAPA-TDNN (pre-trained on VoxCeleb)
Adapter: FiLM-conditioned MLP (the ONLY trainable part)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import tempfile
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .speaker_encoder import SpeakerEncoder
from .adapters import SpeakerAdapter, AdditiveAdapter

logger = logging.getLogger(__name__)

# MMS-TTS model IDs for supported languages
MMS_MODELS = {
    "fr": "facebook/mms-tts-fra",
    "ar": "facebook/mms-tts-ara",
    "zh": "facebook/mms-tts-cmn",
    "en": "facebook/mms-tts-eng",
    "es": "facebook/mms-tts-spa",
    "de": "facebook/mms-tts-deu",
    "ja": "facebook/mms-tts-jpn",
    "ko": "facebook/mms-tts-kor",
    "ru": "facebook/mms-tts-rus",
    "pt": "facebook/mms-tts-por",
}


class RetrofitVoiceCloner(nn.Module):
    """
    Retrofits voice cloning into a non-cloning TTS model.
    
    The TTS model (MMS-TTS/VITS) is frozen. The speaker encoder is frozen.
    ONLY the adapter layers are trained.
    
    During inference:
    1. Speaker encoder extracts embedding from reference audio
    2. Adapter projects embedding into TTS conditioning space
    3. Conditioning is injected into the frozen TTS via FiLM hooks
    4. TTS generates speech in the cloned voice
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = config.get("device", "cuda")
        
        # ── 1. Load the frozen TTS model (no voice cloning) ──
        self.language = config.get("language", "fr")
        self._load_tts_model()
        
        # ── 2. Load the frozen speaker encoder ──
        spk_cfg = config.get("speaker_encoder", {})
        self.speaker_encoder = SpeakerEncoder(
            model_name=spk_cfg.get("model", "speechbrain/spkrec-ecapa-voxceleb"),
            device=self.device,
        )
        
        # ── 3. Create the trainable adapter ──
        adapter_cfg = config.get("adapter", {})
        adapter_type = adapter_cfg.get("type", "film")
        
        tts_hidden_dim = self.tts_config.hidden_size
        speaker_dim = SpeakerEncoder.EMBEDDING_DIM  # 192
        
        if adapter_type == "film":
            self.adapter = SpeakerAdapter(
                speaker_dim=speaker_dim,
                hidden_dim=adapter_cfg.get("hidden_dim", 256),
                output_dim=tts_hidden_dim,
                n_injection_points=adapter_cfg.get("n_injection_points", 3),
                dropout=adapter_cfg.get("dropout", 0.1),
            )
        elif adapter_type == "additive":
            self.adapter = AdditiveAdapter(
                speaker_dim=speaker_dim,
                hidden_dim=adapter_cfg.get("hidden_dim", 256),
                output_dim=tts_hidden_dim,
                dropout=adapter_cfg.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # Storage for conditioning during forward pass
        self._current_conditioning = None
        self._hooks = []
        
        # Register injection hooks into the TTS model
        self._register_injection_hooks()
        
        # Move adapter to device
        self.adapter = self.adapter.to(self.device)
        
        # Log architecture summary
        self._log_summary()
    
    def _load_tts_model(self):
        """Load the frozen MMS-TTS (VITS) model."""
        from transformers import VitsModel, VitsTokenizer
        
        model_id = MMS_MODELS.get(self.language)
        if model_id is None:
            raise ValueError(
                f"Language '{self.language}' not supported. "
                f"Available: {list(MMS_MODELS.keys())}"
            )
        
        logger.info(f"Loading frozen TTS model: {model_id}")
        self.tts_model = VitsModel.from_pretrained(model_id)
        self.tts_tokenizer = VitsTokenizer.from_pretrained(model_id)
        self.tts_config = self.tts_model.config
        
        # Freeze ALL TTS parameters
        self.tts_model.eval()
        for param in self.tts_model.parameters():
            param.requires_grad = False
        
        self.tts_model = self.tts_model.to(self.device)
        
        tts_params = sum(p.numel() for p in self.tts_model.parameters())
        logger.info(f"TTS model loaded and frozen ({tts_params:,} params, 0 trainable)")
    
    def _register_injection_hooks(self):
        """
        Register forward hooks to inject speaker conditioning into the TTS model.
        
        Injection points for VITS:
        0. After text encoder → conditions the linguistic representation
        1. Before flow module → conditions the latent space
        2. Before decoder → conditions the waveform generation
        """
        # Remove any existing hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        adapter_type = self.config.get("adapter", {}).get("type", "film")
        
        # Injection point 0: Text encoder output
        if hasattr(self.tts_model, 'text_encoder'):
            hook = self.tts_model.text_encoder.register_forward_hook(
                self._make_injection_hook(0, adapter_type)
            )
            self._hooks.append(hook)
            logger.info("  Hook registered: text_encoder output")
        
        # Injection point 1: Flow module
        if hasattr(self.tts_model, 'flow'):
            hook = self.tts_model.flow.register_forward_hook(
                self._make_injection_hook(1, adapter_type)
            )
            self._hooks.append(hook)
            logger.info("  Hook registered: flow output")
        
        # Injection point 2: Decoder
        if hasattr(self.tts_model, 'decoder'):
            # Hook on the decoder's input (before generation)
            hook = self.tts_model.decoder.register_forward_pre_hook(
                self._make_pre_injection_hook(2, adapter_type)
            )
            self._hooks.append(hook)
            logger.info("  Hook registered: decoder input")
    
    def _make_injection_hook(self, injection_idx: int, adapter_type: str):
        """Create a forward hook that injects speaker conditioning into outputs."""
        def hook_fn(module, input, output):
            if self._current_conditioning is None:
                return output
            
            conditioning = self._current_conditioning
            
            # ── Extract the hidden state tensor from various output types ──
            
            # Case 1: ModelOutput / dataclass (e.g. VitsTextEncoderOutput)
            # Has .last_hidden_state attribute
            if hasattr(output, 'last_hidden_state'):
                hidden = output.last_hidden_state
                hidden = self._apply_conditioning(
                    hidden, conditioning, injection_idx, adapter_type
                )
                if hidden is not None:
                    output.last_hidden_state = hidden
                return output
            
            # Case 2: Tuple of tensors
            if isinstance(output, tuple):
                if len(output) == 0 or not isinstance(output[0], torch.Tensor):
                    return output
                hidden = output[0]
                rest = output[1:]
                hidden = self._apply_conditioning(
                    hidden, conditioning, injection_idx, adapter_type
                )
                if hidden is not None:
                    return (hidden,) + rest
                return output
            
            # Case 3: Raw tensor
            if isinstance(output, torch.Tensor):
                result = self._apply_conditioning(
                    output, conditioning, injection_idx, adapter_type
                )
                return result if result is not None else output
            
            # Unknown type — skip injection
            return output
        
        return hook_fn
    
    def _make_pre_injection_hook(self, injection_idx: int, adapter_type: str):
        """Create a pre-hook that modifies inputs to a module."""
        def hook_fn(module, input):
            if self._current_conditioning is None:
                return input
            
            conditioning = self._current_conditioning
            
            if isinstance(input, tuple) and len(input) > 0:
                first = input[0]
                if isinstance(first, torch.Tensor):
                    result = self._apply_conditioning(
                        first, conditioning, injection_idx, adapter_type
                    )
                    if result is not None:
                        return (result,) + input[1:]
            
            return input
        
        return hook_fn
    
    def _apply_conditioning(
        self,
        hidden: torch.Tensor,
        conditioning: torch.Tensor,
        injection_idx: int,
        adapter_type: str,
    ) -> Optional[torch.Tensor]:
        """
        Apply speaker conditioning to a hidden state tensor.
        
        Returns the conditioned tensor, or None if dimensions don't match.
        """
        # Check dimension compatibility
        # Conditioning is [batch, output_dim], hidden could be [batch, seq, dim] or [batch, dim]
        cond_dim = conditioning.shape[-1]
        hidden_dim = hidden.shape[-1]
        
        if cond_dim != hidden_dim:
            # Dimension mismatch — skip this injection point
            return None
        
        try:
            if adapter_type == "film" and isinstance(self.adapter, SpeakerAdapter):
                if injection_idx < self.adapter.n_injection_points:
                    return self.adapter.apply_film(hidden, conditioning, injection_idx)
            elif adapter_type == "additive" and isinstance(self.adapter, AdditiveAdapter):
                return self.adapter.apply_conditioning(hidden, conditioning)
            else:
                # Fallback: scaled additive
                cond_expanded = conditioning.unsqueeze(1) if hidden.dim() == 3 else conditioning
                return hidden + cond_expanded * 0.1
        except Exception as e:
            logger.debug(f"Conditioning failed at injection point {injection_idx}: {e}")
            return None
    
    def _log_summary(self):
        """Log the architecture summary."""
        tts_params = sum(p.numel() for p in self.tts_model.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        total_params = tts_params + adapter_params
        
        logger.info("=" * 50)
        logger.info("Retrofit Voice Cloner Architecture")
        logger.info("=" * 50)
        logger.info(f"  TTS Model:        {MMS_MODELS.get(self.language)} (FROZEN)")
        logger.info(f"  TTS Params:       {tts_params:,} (0 trainable)")
        logger.info(f"  Speaker Encoder:  ECAPA-TDNN (FROZEN)")
        logger.info(f"  Adapter Params:   {adapter_params:,} (100% trainable)")
        logger.info(f"  Adapter Ratio:    {adapter_params/total_params*100:.2f}% of total")
        logger.info(f"  Injection Points: {len(self._hooks)}")
        logger.info("=" * 50)
    
    def forward(
        self,
        text: str,
        ref_audio: Union[torch.Tensor, np.ndarray],
        ref_sr: int = 16000,
    ) -> torch.Tensor:
        """
        Full forward pass: text + reference audio → cloned speech.
        
        Args:
            text: Text to synthesize
            ref_audio: Reference audio for voice cloning
            ref_sr: Sample rate of reference audio
            
        Returns:
            Generated waveform tensor
        """
        # Step 1: Extract speaker embedding (frozen)
        speaker_emb = self.speaker_encoder(ref_audio, sr=ref_sr)
        
        # Step 2: Project through adapter (trainable)
        conditioning = self.adapter(speaker_emb)
        
        # Step 3: Store conditioning for hooks
        self._current_conditioning = conditioning
        
        # Step 4: Tokenize text
        inputs = self.tts_tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Step 5: Run TTS (frozen, hooks inject conditioning)
        with torch.no_grad():
            output = self.tts_model(input_ids)
        
        # Clear conditioning
        self._current_conditioning = None
        
        return output.waveform.squeeze()
    
    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        ref_audio: Union[torch.Tensor, np.ndarray],
        ref_text: str = "",
        ref_sr: int = 16000,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech (evaluation-friendly interface).
        
        Returns:
            Tuple of (audio_numpy_array, sample_rate)
        """
        waveform = self.forward(text, ref_audio, ref_sr)
        
        audio_np = waveform.cpu().numpy()
        sr = self.tts_config.sampling_rate  # MMS-TTS outputs at its native SR
        
        return audio_np, sr
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get ONLY the adapter's trainable parameters."""
        return list(self.adapter.parameters())
    
    def save_adapter(self, path: str):
        """Save only the adapter weights (tiny file)."""
        torch.save({
            "adapter_state_dict": self.adapter.state_dict(),
            "config": {
                "language": self.language,
                "adapter_type": self.config.get("adapter", {}).get("type", "film"),
                "adapter_params": sum(p.numel() for p in self.adapter.parameters()),
            },
        }, path)
        
        size_kb = Path(path).stat().st_size / 1024
        logger.info(f"Saved adapter ({size_kb:.1f} KB) to {path}")
    
    def load_adapter(self, path: str):
        """Load adapter weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        logger.info(f"Loaded adapter from {path}")
    
    def train_mode(self):
        """Set adapter to training mode (TTS stays in eval)."""
        self.adapter.train()
        self.tts_model.eval()  # Always frozen
    
    def eval_mode(self):
        """Set everything to eval mode."""
        self.adapter.eval()
        self.tts_model.eval()
    
    def switch_language(self, language: str):
        """
        Switch the TTS backbone to a different language.
        
        The adapter transfers across languages — this is a key result
        of the research: train once, deploy to any language.
        """
        if language == self.language:
            return
        
        logger.info(f"Switching TTS language: {self.language} → {language}")
        
        # Remove old hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        # Load new TTS model
        self.language = language
        self._load_tts_model()
        
        # Re-register hooks
        self._register_injection_hooks()
        
        logger.info(f"Language switched to {language}")

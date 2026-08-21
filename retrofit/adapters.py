"""
Adapter Layers: Trainable projection from speaker embedding to TTS conditioning.

This is the ONLY trainable component in the retrofit pipeline:
  Speaker Embedding (192-dim) → [Adapter (trained)] → TTS Conditioning Vector

The adapter projects the speaker identity into the TTS model's hidden space
using FiLM (Feature-wise Linear Modulation) conditioning.
"""

import torch
import torch.nn as nn
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SpeakerAdapter(nn.Module):
    """
    Projects speaker embeddings into the TTS model's conditioning space.
    
    Architecture:
        speaker_emb (192) → MLP → conditioning (hidden_dim)
        + FiLM parameters (gamma, beta) for each injection point
    
    This is the ONLY module that gets trained. Everything else is frozen.
    """
    
    def __init__(
        self,
        speaker_dim: int = 192,        # ECAPA-TDNN output dim
        hidden_dim: int = 256,          # Adapter hidden dim
        output_dim: int = 192,          # TTS model's hidden size (VITS)
        n_injection_points: int = 3,    # Number of places to inject conditioning
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.speaker_dim = speaker_dim
        self.output_dim = output_dim
        self.n_injection_points = n_injection_points
        
        # Main projection: speaker embedding → conditioning space
        self.projection = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # FiLM layers: one (gamma, beta) pair per injection point
        # FiLM: output = gamma * hidden_state + beta
        self.film_gammas = nn.ModuleList([
            nn.Linear(output_dim, output_dim)
            for _ in range(n_injection_points)
        ])
        self.film_betas = nn.ModuleList([
            nn.Linear(output_dim, output_dim)
            for _ in range(n_injection_points)
        ])
        
        # Initialize FiLM to identity (gamma=1, beta=0) so model starts unchanged
        for gamma_layer in self.film_gammas:
            nn.init.ones_(gamma_layer.weight.data.diagonal())
            nn.init.zeros_(gamma_layer.bias)
        for beta_layer in self.film_betas:
            nn.init.zeros_(beta_layer.weight)
            nn.init.zeros_(beta_layer.bias)
        
        self._log_param_count()
    
    def _log_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"SpeakerAdapter: {total:,} trainable parameters")
    
    def forward(self, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project speaker embedding to conditioning space.
        
        Args:
            speaker_embedding: [batch, speaker_dim] from the speaker encoder
            
        Returns:
            Conditioning vector [batch, output_dim]
        """
        return self.projection(speaker_embedding)
    
    def apply_film(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        injection_index: int,
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning to hidden states at a specific injection point.
        
        FiLM: output = gamma(conditioning) * hidden_states + beta(conditioning)
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] from TTS model
            conditioning: [batch, output_dim] from self.forward()
            injection_index: Which injection point (0, 1, 2, ...)
            
        Returns:
            Conditioned hidden states [batch, seq_len, hidden_dim]
        """
        gamma = self.film_gammas[injection_index](conditioning)  # [batch, hidden_dim]
        beta = self.film_betas[injection_index](conditioning)    # [batch, hidden_dim]
        
        # Expand for broadcasting: [batch, 1, hidden_dim]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        return gamma * hidden_states + beta


class AdditiveAdapter(nn.Module):
    """
    Simpler adapter that uses additive conditioning only.
    
    Instead of FiLM (scale + shift), just adds the projected
    speaker embedding to the hidden states.
    
    Useful as an ablation baseline.
    """
    
    def __init__(
        self,
        speaker_dim: int = 192,
        hidden_dim: int = 256,
        output_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Scale factor to control conditioning strength
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, speaker_embedding: torch.Tensor) -> torch.Tensor:
        return self.projection(speaker_embedding)
    
    def apply_conditioning(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Add conditioning to hidden states."""
        # [batch, output_dim] → [batch, 1, output_dim]
        cond = conditioning.unsqueeze(1) * self.scale
        return hidden_states + cond


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for training the adapter.
    
    Objective: Same-speaker pairs should have similar conditioning vectors,
    different-speaker pairs should have dissimilar conditioning vectors.
    
    This trains the adapter WITHOUT needing to run the TTS model,
    making training fast and memory-efficient.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        speaker_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            embeddings: [batch, dim] conditioning vectors from the adapter
            speaker_ids: [batch] integer speaker IDs
            
        Returns:
            Scalar loss
        """
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        
        # Compute similarity matrix [batch, batch]
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive mask: same speaker = positive pair
        speaker_ids = speaker_ids.view(-1)
        pos_mask = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
        
        # Remove self-similarities from diagonal
        pos_mask.fill_diagonal_(0)
        
        # Check if there are any positive pairs
        if pos_mask.sum() == 0:
            # No same-speaker pairs in batch — use uniformity loss instead
            return self._uniformity_loss(embeddings)
        
        # For each anchor, compute log-sum-exp over all non-self entries
        # and subtract the log of positive pair similarities
        exp_sim = torch.exp(sim_matrix)
        exp_sim.fill_diagonal_(0)  # Exclude self
        
        # Log denominator: log(sum of all exp similarities)
        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)
        
        # Log numerator: log(mean of positive pair similarities)
        pos_sim = (exp_sim * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        log_numer = torch.log(pos_sim + 1e-8)
        
        # InfoNCE loss
        loss = -(log_numer - log_denom).mean()
        
        return loss
    
    def _uniformity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Fallback: encourage uniform distribution on the hypersphere."""
        sq_pdist = torch.pdist(embeddings, p=2).pow(2)
        return sq_pdist.mul(-2).exp().mean().log()


class SpeakerConsistencyLoss(nn.Module):
    """
    Ensure the adapter produces consistent outputs for the same speaker.
    
    Given two audio clips from the same speaker, the adapter's conditioning
    vectors should be similar (high cosine similarity).
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        cond_a: torch.Tensor,
        cond_b: torch.Tensor,
        same_speaker: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cond_a, cond_b: [batch, dim] conditioning from different utterances
            same_speaker: [batch] boolean — True if same speaker
        """
        cos_sim = nn.functional.cosine_similarity(cond_a, cond_b, dim=-1)
        
        # Same speaker: maximize similarity
        same_loss = (1 - cos_sim) * same_speaker.float()
        
        # Different speaker: push apart (with margin)
        diff_loss = torch.clamp(cos_sim - self.margin, min=0) * (~same_speaker).float()
        
        return (same_loss + diff_loss).mean()

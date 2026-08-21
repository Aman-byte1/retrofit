"""
Mamba (Selective State Space Model) Language Model (~50M params).

Implementation based on:
  "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
  https://github.com/state-spaces/mamba

Uses the official `mamba_ssm` package for the core SSM block.
Falls back to a pure-PyTorch implementation if mamba_ssm is not available.
Equipped with QwenRMSNorm for layer normalization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .transformer_lm import QwenRMSNorm


class MambaBlock(nn.Module):
    """
    Mamba Selective SSM block.
    
    Tries to use the official `mamba_ssm.Mamba` implementation.
    Falls back to a pure-PyTorch selective SSM if not available.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        
        self._use_official = False
        
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._use_official = True
        except ImportError:
            self._build_fallback()
    
    def _build_fallback(self):
        """Build pure-PyTorch selective SSM."""
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
    
    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_proj = self.x_proj(x)
        B_ssm = x_proj[..., :self.d_state]
        C_ssm = x_proj[..., self.d_state:2*self.d_state]
        dt = F.softplus(self.dt_proj(x_proj[..., -1:]))
        
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(L):
            dt_t = dt[:, t]
            B_t = B_ssm[:, t]
            C_t = C_ssm[:, t]
            x_t = x[:, t]
            
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            h = dA * h + dB * x_t.unsqueeze(-1)
            y = (C_t.unsqueeze(1) * h).sum(-1)
            y = y + self.D * x_t
            ys.append(y)
        
        return torch.stack(ys, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_official:
            return self.mamba(x)
        
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :L]
        x_inner = x_inner.transpose(1, 2)
        x_inner = F.silu(x_inner)
        
        y = self._ssm_scan(x_inner)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaLayer(nn.Module):
    """Pre-norm Mamba layer."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = QwenRMSNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand)
    
    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MambaLM(nn.Module):
    """
    Mamba Language Model.
    
    Architecture: Embedding → N × MambaLayer → RMSNorm → LM Head
    """
    ARCH_NAME = "mamba"
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        self.norm = QwenRMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        h = self.drop(self.tok_emb(x))
        
        for layer in self.layers:
            h = layer(h)
        
        h = self.norm(h)
        logits = self.lm_head(h)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        
        return logits, loss
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mamba(vocab_size: int, target_params: int = 50_000_000) -> MambaLM:
    """Create a Mamba LM targeting ~48M-50M parameters."""
    return MambaLM(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=28,
        d_state=16,
        d_conv=4,
        expand=2,
    )

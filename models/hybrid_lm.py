"""
Hybrid Mamba-Transformer Language Model (~50M params).

Interleaves Mamba SSM layers with Qwen3 Attention layers.
- Mamba layers: O(N) linear complexity for sequence processing
- Attention layers: Qwen3.5 Attention with QK-Norm and RoPE for associative recall

Inspired by Jamba (AI21), Zamba (Zyphra), and Nemotron architectures.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .transformer_lm import QwenRMSNorm, Qwen3MLP, Qwen3Attention
from .mamba_lm import MambaBlock


class HybridMambaLayer(nn.Module):
    """Pre-norm Mamba layer for the hybrid model."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = QwenRMSNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.ffn_norm = QwenRMSNorm(d_model)
        self.ffn = Qwen3MLP(d_model, d_model * 2)
    
    def forward(self, x):
        x = x + self.mamba(self.norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HybridAttentionLayer(nn.Module):
    """Pre-norm Qwen3 attention layer for the hybrid model."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 4096):
        super().__init__()
        self.attn_norm = QwenRMSNorm(d_model)
        self.attn = Qwen3Attention(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads // 2,
            head_dim=d_model // n_heads,
        )
        self.ffn_norm = QwenRMSNorm(d_model)
        self.ffn = Qwen3MLP(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HybridMambaTransformerLM(nn.Module):
    """
    Hybrid Mamba-Transformer Language Model.
    
    Architecture:
        Embedding → [Mamba, Mamba, QwenAttn, Mamba, Mamba, QwenAttn, ...] → RMSNorm → LM Head
    """
    ARCH_NAME = "hybrid"
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 18,
        n_heads: int = 8,
        d_ff: int = 1408,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        attn_every: int = 3,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.attn_every = attn_every
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList()
        self.layer_types = []
        
        for i in range(n_layers):
            if (i + 1) % attn_every == 0:
                self.layers.append(
                    HybridAttentionLayer(d_model, n_heads, d_ff, max_seq_len)
                )
                self.layer_types.append("attention")
            else:
                self.layers.append(
                    HybridMambaLayer(d_model, d_state, d_conv, expand)
                )
                self.layer_types.append("mamba")
        
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
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
        
        return logits, loss
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_breakdown(self):
        n_mamba = sum(1 for t in self.layer_types if t == "mamba")
        n_attn = sum(1 for t in self.layer_types if t == "attention")
        return {"mamba_layers": n_mamba, "attention_layers": n_attn, "total": len(self.layer_types)}


def create_hybrid(vocab_size: int, target_params: int = 50_000_000) -> HybridMambaTransformerLM:
    """Create a Hybrid Mamba-Transformer LM targeting ~49M-50M parameters."""
    return HybridMambaTransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=15,
        n_heads=8,
        d_ff=1408,
        d_state=16,
        d_conv=4,
        expand=2,
        attn_every=3,
        max_seq_len=4096,
    )

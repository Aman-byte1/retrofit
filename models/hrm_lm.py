"""
HRM (Hierarchical Recurrent Memory) Language Model (~50M params).

Simplified implementation of the HRM architecture from:
  "HRM-Text: Efficient Pretraining Beyond Scaling" (Wang et al., 2026)
  https://github.com/sapientinc/HRM-Text

Key idea: Dual-timescale processing with:
  - H (High-level) module: Slow, abstract planning (processes every K tokens)
  - L (Low-level) module: Fast, detailed token-by-token processing
  
The H module provides top-down context to the L module via cross-attention,
while the L module provides bottom-up summaries to the H module.

Uses FlashAttention 2 / PyTorch SDPA (patched from FA3 for A100 compatibility).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .transformer_lm import RMSNorm, SwiGLU, precompute_rope, apply_rope


class HRMAttention(nn.Module):
    """Multi-Head Attention with RoPE for HRM modules."""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        cos, sin = precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
    
    def forward(self, x: torch.Tensor, causal: bool = True):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.wo(out.transpose(1, 2).contiguous().view(B, S, D))


class CrossAttention(nn.Module):
    """Cross-attention for H→L or L→H communication."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor):
        B, S, D = x.shape
        _, C, _ = context.shape
        
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(context).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(context).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.wo(out.transpose(1, 2).contiguous().view(B, S, D))


class HBlock(nn.Module):
    """High-level (H) module block: slow, abstract processing."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 4096):
        super().__init__()
        self.self_attn_norm = RMSNorm(d_model)
        self.self_attn = HRMAttention(d_model, n_heads, max_seq_len)
        
        self.cross_attn_norm = RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads)
        
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, h: torch.Tensor, l_summary: Optional[torch.Tensor] = None):
        # Self-attention
        h = h + self.self_attn(self.self_attn_norm(h))
        
        # Cross-attention from L summary (bottom-up)
        if l_summary is not None:
            h = h + self.cross_attn(self.cross_attn_norm(h), l_summary)
        
        # FFN
        h = h + self.ffn(self.ffn_norm(h))
        return h


class LBlock(nn.Module):
    """Low-level (L) module block: fast, detailed processing."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 4096):
        super().__init__()
        self.self_attn_norm = RMSNorm(d_model)
        self.self_attn = HRMAttention(d_model, n_heads, max_seq_len)
        
        self.cross_attn_norm = RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads)
        
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, l: torch.Tensor, h_context: Optional[torch.Tensor] = None):
        # Self-attention
        l = l + self.self_attn(self.self_attn_norm(l))
        
        # Cross-attention from H context (top-down)
        if h_context is not None:
            l = l + self.cross_attn(self.cross_attn_norm(l), h_context)
        
        # FFN
        l = l + self.ffn(self.ffn_norm(l))
        return l


class HRMLM(nn.Module):
    """
    Hierarchical Recurrent Memory Language Model.
    
    Architecture:
        Embedding → [L-blocks ↔ H-blocks (interleaved)] → RMSNorm → LM Head
    
    The H module operates on a compressed view (every K-th token),
    while the L module operates on all tokens. They communicate
    via cross-attention at each depth level.
    """
    
    ARCH_NAME = "hrm"
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_h_layers: int = 6,
        n_l_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1408,
        stride: int = 4,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.stride = stride
        self.n_h_layers = n_h_layers
        self.n_l_layers = n_l_layers
        
        assert n_h_layers == n_l_layers, "H and L must have same depth for interleaving"
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Compression projection for H module input
        self.compress = nn.Linear(d_model * stride, d_model, bias=False)
        # Expansion projection for H→L
        self.expand = nn.Linear(d_model, d_model, bias=False)
        
        # H (high-level) blocks
        self.h_blocks = nn.ModuleList([
            HBlock(d_model, n_heads, d_ff, max_seq_len // stride)
            for _ in range(n_h_layers)
        ])
        
        # L (low-level) blocks
        self.l_blocks = nn.ModuleList([
            LBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_l_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _compress_for_h(self, x: torch.Tensor) -> torch.Tensor:
        """Compress token sequence for the H module by grouping every `stride` tokens."""
        B, S, D = x.shape
        # Pad to multiple of stride
        pad = (self.stride - S % self.stride) % self.stride
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        
        S_padded = x.shape[1]
        # Reshape: [B, S/stride, stride*D] -> project to [B, S/stride, D]
        x = x.view(B, S_padded // self.stride, self.stride * D)
        return self.compress(x)
    
    def _expand_for_l(self, h: torch.Tensor, target_len: int) -> torch.Tensor:
        """Expand H representations to match L sequence length via repeat-interleave."""
        # h: [B, S/stride, D] -> repeat each position `stride` times
        expanded = h.repeat_interleave(self.stride, dim=1)
        # Trim to target length
        return self.expand(expanded[:, :target_len])
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, S = x.shape
        
        # Token embeddings
        l_state = self.drop(self.tok_emb(x))  # [B, S, D]
        
        # Initial H state via compression
        h_state = self._compress_for_h(l_state)  # [B, S/stride, D]
        
        # Interleaved processing
        for i in range(self.n_h_layers):
            # L processes with top-down context from H
            h_context = self._expand_for_l(h_state, S)
            l_state = self.l_blocks[i](l_state, h_context)
            
            # H processes with bottom-up summary from L
            l_summary = self._compress_for_h(l_state)
            h_state = self.h_blocks[i](h_state, l_summary)
        
        # Output from L module (full resolution)
        h = self.norm(l_state)
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


def create_hrm(vocab_size: int, target_params: int = 50_000_000) -> HRMLM:
    """Create an HRM LM targeting approximately `target_params` parameters."""
    # ~50M: d=512, 6+6 layers, d_ff=1024, stride=4
    # Cross-attention adds ~25% params per layer vs pure self-attn
    model = HRMLM(
        vocab_size=vocab_size,
        d_model=512,
        n_h_layers=6,
        n_l_layers=6,
        n_heads=8,
        d_ff=1024,
        stride=4,
        max_seq_len=4096,
    )
    return model

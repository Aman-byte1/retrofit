"""
Modern Transformer Language Model (~50M params).

Llama-style architecture:
- RoPE (Rotary Position Embeddings)
- SwiGLU MLP activations
- RMSNorm (pre-norm)
- Grouped Query Attention with PyTorch SDPA
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings."""
    B, H, S, D = x.shape
    cos = cos[:S].unsqueeze(0).unsqueeze(0)  # [1, 1, S, D//2]
    sin = sin[:S].unsqueeze(0).unsqueeze(0)
    
    x1 = x[..., :D//2]
    x2 = x[..., D//2:]
    
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    
    return torch.cat([out1, out2], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear unit."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerAttention(nn.Module):
    """Multi-Head Attention with RoPE and SDPA."""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        assert d_model % n_heads == 0
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
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.wo(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 4096):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = TransformerAttention(d_model, n_heads, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """
    Full Transformer Language Model.
    
    Architecture: Embedding → N × TransformerBlock → RMSNorm → LM Head
    """
    
    ARCH_NAME = "transformer"
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 16,
        n_heads: int = 8,
        d_ff: int = 1408,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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
                ignore_index=0,  # PAD
            )
        
        return logits, loss
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer(vocab_size: int, target_params: int = 50_000_000) -> TransformerLM:
    """Create a Transformer LM targeting approximately `target_params` parameters."""
    # ~50M config: d=512, layers=16, d_ff=1408, heads=8
    # Param estimate: 
    #   embed = vocab * 512 (shared with lm_head)
    #   per_layer = 4*512^2 + 3*512*1408 = 1,048,576 + 2,162,688 = ~3.2M
    #   16 layers = ~51M + embed
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=16,
        n_heads=8,
        d_ff=1408,
        max_seq_len=4096,
    )
    return model

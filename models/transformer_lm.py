"""
Qwen3 / Qwen3.5 Transformer Language Model (~50M params).

Exact implementation matching Qwen3 / Qwen3.5 architecture:
- QK-Norm (RMSNorm applied to query and key heads before attention)
- Grouped Query Attention (GQA) with RoPE (theta = 1,000,000)
- SwiGLU MLP (gate_proj, up_proj, down_proj with SiLU activation)
- RMSNorm pre-normalization
- Scaled Dot-Product Attention with PyTorch SDPA (FlashAttention-2 backend)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QwenRMSNorm(nn.Module):
    """
    Qwen3 / Qwen3.5 RMSNorm.
    Computes variance in float32 for numerical stability.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def precompute_rope(dim: int, max_seq_len: int, theta: float = 1000000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE frequencies (Qwen3 uses theta = 1,000,000)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embeddings (RoPE)."""
    # x shape: [B, num_heads, S, head_dim]
    B, H, S, D = x.shape
    cos = cos[:S].unsqueeze(0).unsqueeze(0)  # [1, 1, S, D//2]
    sin = sin[:S].unsqueeze(0).unsqueeze(0)  # [1, 1, S, D//2]
    
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]
    
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for Grouped Query Attention (GQA).
    hidden_states: [B, num_key_value_heads, S, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    B, num_kv_heads, S, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(B, num_kv_heads, n_rep, S, head_dim)
    return hidden_states.reshape(B, num_kv_heads * n_rep, S, head_dim)


class Qwen3MLP(nn.Module):
    """Qwen3 / Qwen3.5 SwiGLU MLP."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    """
    Qwen3 / Qwen3.5 Attention Module with:
    - Grouped Query Attention (GQA)
    - QK-Norm (RMSNorm on head dimension for Q and K)
    - RoPE with base theta = 1,000,000
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # Projections (bias=False in modern Qwen3)
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        
        # QK-Norm: Distinctive feature of Qwen3 architecture!
        self.q_norm = QwenRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = QwenRMSNorm(self.head_dim, eps=rms_norm_eps)
        
        # Precompute RoPE
        cos, sin = precompute_rope(self.head_dim, max_position_embeddings, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        
        # 1. Project
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 2. QK-Norm (applied per head)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 3. Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        
        # 4. Grouped Query Attention repeat KV heads if GQA
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        
        # 5. Scaled Dot Product Attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        
        return self.o_proj(out)


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 / Qwen3.5 Transformer Decoder Layer."""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_layernorm = QwenRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = QwenRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-norm with residual
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class TransformerLM(nn.Module):
    """
    Qwen3.5 Architecture Language Model.
    
    Architecture:
        Embedding → N × Qwen3DecoderLayer → QwenRMSNorm → LM Head
    """
    ARCH_NAME = "transformer"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 16,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = 4,
        d_ff: int = 1408,
        max_seq_len: int = 4096,
        rope_theta: float = 1000000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=d_model,
                intermediate_size=d_ff,
                num_attention_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                max_position_embeddings=max_seq_len,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(n_layers)
        ])
        
        self.norm = QwenRMSNorm(d_model, eps=rms_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (standard in efficient LLMs)
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
                ignore_index=0,  # PAD
            )
        
        return logits, loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias for backward compatibility
RMSNorm = QwenRMSNorm
SwiGLU = Qwen3MLP
TransformerAttention = Qwen3Attention


def create_transformer(vocab_size: int, target_params: int = 50_000_000) -> TransformerLM:
    """Create a Qwen3.5 Transformer LM targeting ~50M parameters."""
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=16,
        n_heads=8,
        n_kv_heads=4,
        d_ff=1408,
        max_seq_len=4096,
        rope_theta=1000000.0,
    )

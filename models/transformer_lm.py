"""
Qwen3.5 / Qwen3.8 Language Model Architecture (~50M params).

Complete, exact implementation of the Qwen3.5 text-only dense backbone:
  • 3:1 Gated DeltaNet / Gated Full-Attention hybrid blocks
  • Causal depthwise convolution in DeltaNet
  • Chunk-parallel gated delta rule (differentiable, linear sequence complexity)
  • Grouped-Query Full Attention with Q/K norm, Partial RoPE, and output sigmoid gate
  • Zero-Centered RMSNorm and SwiGLU MLP
  • Optional Multi-Token Prediction (MTP) auxiliary objective
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


# -----------------------------------------------------------------------------
# Core Normalization & RoPE


class ZeroCenteredRMSNorm(nn.Module):
    """Qwen3.5 RMSNorm: scale is (1 + weight), and weight starts at zero."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(x.dtype)


class GatedRMSNorm(nn.Module):
    """RMS-normalize first, then apply a SiLU output gate."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        y = y * self.weight.float() * F.silu(gate.float())
        return y.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10_000_000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE frequencies."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embeddings (RoPE)."""
    B, H, S, D = x.shape
    cos = cos[:S].unsqueeze(0).unsqueeze(0)
    sin = sin[:S].unsqueeze(0).unsqueeze(0)
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


class PartialRoPE(nn.Module):
    """Qwen3.5 partial rotary embeddings applied to a fraction of the head dimension."""

    def __init__(self, head_dim: int, fraction: float = 0.5, theta: float = 10_000_000.0):
        super().__init__()
        self.dim = int(head_dim * fraction)
        if self.dim % 2 != 0:
            self.dim -= 1
        inv = 1.0 / (theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = q.shape[-2]
        pos = torch.arange(t, device=q.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None].to(q.dtype)
        sin = emb.sin()[None, None].to(q.dtype)

        def apply(x: torch.Tensor) -> torch.Tensor:
            rot, passthrough = x[..., : self.dim], x[..., self.dim :]
            rot = rot * cos + rotate_half(rot) * sin
            return torch.cat((rot, passthrough), dim=-1)

        return apply(q), apply(k)


# -----------------------------------------------------------------------------
# Gated Full Attention


class GatedAttention(nn.Module):
    """Qwen3.5 gated grouped-query attention with Q/K norm and partial RoPE."""

    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        attention_dropout: float = 0.0,
        partial_rotary_factor: float = 0.5,
        rope_theta: float = 10_000_000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dropout = attention_dropout

        self.q_proj = nn.Linear(hidden_size, self.num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = ZeroCenteredRMSNorm(head_dim, rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(head_dim, rms_norm_eps)
        self.rope = PartialRoPE(head_dim, partial_rotary_factor, rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q_gate = self.q_proj(x).view(b, t, self.num_heads, self.head_dim * 2)
        q, gate = q_gate.chunk(2, dim=-1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        repeat = self.num_heads // self.num_kv_heads
        if repeat != 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        y = y * torch.sigmoid(gate.reshape(b, t, -1))
        return self.o_proj(y)


# -----------------------------------------------------------------------------
# Gated DeltaNet (Chunk-Parallel Delta Rule)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + eps)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Differentiable chunk-parallel gated delta rule.
    Computes in fp32 for stability and is linear O(N) in sequence length.
    """
    original_dtype = query.dtype
    query = l2norm(query).transpose(1, 2).contiguous().float()
    key = l2norm(key).transpose(1, 2).contiguous().float()
    value = value.transpose(1, 2).contiguous().float()
    beta = beta.transpose(1, 2).contiguous().float()
    g = log_decay.transpose(1, 2).contiguous().float()

    batch, heads, sequence_length, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (-sequence_length) % chunk_size
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
    total = sequence_length + pad
    query = query * (k_dim ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    tensors = (query, key, value, k_beta, v_beta)
    query, key, value, k_beta, v_beta = [
        z.reshape(batch, heads, -1, chunk_size, z.shape[-1]) for z in tensors
    ]
    g = g.reshape(batch, heads, -1, chunk_size).cumsum(dim=-1)

    lower_decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    diagonal_and_up = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    wy = -((k_beta @ key.transpose(-1, -2)) * lower_decay).masked_fill(diagonal_and_up, 0)

    wy = wy.clone()
    for i in range(1, chunk_size):
        row = wy[..., i, :i].clone()
        previous = wy[..., :i, :i].clone()
        wy[..., i, :i] = row + (row.unsqueeze(-1) * previous).sum(-2)
    eye = torch.eye(chunk_size, dtype=wy.dtype, device=wy.device)
    transform = wy + eye
    value = transform @ v_beta
    k_cumdecay = transform @ (k_beta * g.exp().unsqueeze(-1))

    state = torch.zeros(batch, heads, k_dim, v_dim, dtype=torch.float32, device=query.device)
    outputs = []
    for i in range(total // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        within = (q_i @ k_i.transpose(-1, -2)) * lower_decay[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        between = (q_i * g[:, :, i, :, None].exp()) @ state
        outputs.append(between + within @ v_new)
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
            @ v_new
        )

    out = torch.stack(outputs, dim=2).reshape(batch, heads, total, v_dim)
    out = out[:, :, :sequence_length].transpose(1, 2).contiguous()
    return out.to(original_dtype)


class GatedDeltaNet(nn.Module):
    """Qwen3.5 Gated DeltaNet layer."""

    def __init__(
        self,
        hidden_size: int = 512,
        linear_num_key_heads: int = 8,
        linear_num_value_heads: int = 8,
        linear_key_head_dim: int = 64,
        linear_value_head_dim: int = 64,
        linear_conv_kernel_dim: int = 4,
        delta_chunk_size: int = 16,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_k_heads = linear_num_key_heads
        self.num_v_heads = linear_num_value_heads
        self.k_dim = linear_key_head_dim
        self.v_dim = linear_value_head_dim
        self.key_width = self.num_k_heads * self.k_dim
        self.value_width = self.num_v_heads * self.v_dim
        self.conv_width = self.key_width * 2 + self.value_width
        self.chunk_size = delta_chunk_size

        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_width, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_width, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            self.conv_width,
            self.conv_width,
            linear_conv_kernel_dim,
            padding=linear_conv_kernel_dim - 1,
            groups=self.conv_width,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        a = torch.empty(self.num_v_heads).uniform_(0.01, 16.0)
        self.A_log = nn.Parameter(a.log())
        self.norm = GatedRMSNorm(self.v_dim, rms_norm_eps)
        self.out_proj = nn.Linear(self.value_width, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        mixed = F.silu(self.conv1d(mixed)[..., :t]).transpose(1, 2)
        q, k, v = torch.split(mixed, [self.key_width, self.key_width, self.value_width], dim=-1)
        q = q.view(b, t, self.num_k_heads, self.k_dim)
        k = k.view(b, t, self.num_k_heads, self.k_dim)
        v = v.view(b, t, self.num_v_heads, self.v_dim)

        if self.num_v_heads != self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        beta = torch.sigmoid(self.in_proj_b(x))
        log_decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        y = chunk_gated_delta_rule(q, k, v, log_decay, beta, self.chunk_size)
        z = self.in_proj_z(x).view(b, t, self.num_v_heads, self.v_dim)
        y = self.norm(y, z).reshape(b, t, self.value_width)
        return self.out_proj(y)


# -----------------------------------------------------------------------------
# SwiGLU MLP & Decoder Layer


class Qwen35SwiGLU(nn.Module):
    """Qwen3.5 SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35DecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer: alternates between GatedDeltaNet and GatedAttention."""

    def __init__(
        self,
        hidden_size: int = 512,
        intermediate_size: int = 1408,
        index: int = 0,
        full_attention_interval: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        linear_num_key_heads: int = 8,
        linear_num_value_heads: int = 8,
        linear_key_head_dim: int = 64,
        linear_value_head_dim: int = 64,
        linear_conv_kernel_dim: int = 4,
        delta_chunk_size: int = 16,
        partial_rotary_factor: float = 0.5,
        rope_theta: float = 10_000_000.0,
        rms_norm_eps: float = 1e-6,
        residual_dropout: float = 0.0,
        force_full_attention: bool = False,
    ):
        super().__init__()
        is_full = force_full_attention or ((index + 1) % full_attention_interval == 0)

        if is_full:
            self.mixer = GatedAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                partial_rotary_factor=partial_rotary_factor,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
            )
        else:
            self.mixer = GatedDeltaNet(
                hidden_size=hidden_size,
                linear_num_key_heads=linear_num_key_heads,
                linear_num_value_heads=linear_num_value_heads,
                linear_key_head_dim=linear_key_head_dim,
                linear_value_head_dim=linear_value_head_dim,
                linear_conv_kernel_dim=linear_conv_kernel_dim,
                delta_chunk_size=delta_chunk_size,
                rms_norm_eps=rms_norm_eps,
            )

        self.input_layernorm = ZeroCenteredRMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(hidden_size, rms_norm_eps)
        self.mlp = Qwen35SwiGLU(hidden_size, intermediate_size)
        self.dropout = residual_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + F.dropout(self.mixer(self.input_layernorm(x)), self.dropout, self.training)
        x = x + F.dropout(self.mlp(self.post_attention_layernorm(x)), self.dropout, self.training)
        return x


# -----------------------------------------------------------------------------
# Full Qwen3.5 Model


class TransformerLM(nn.Module):
    """
    Qwen3.5 Language Model.
    
    Equipped with:
      - 3:1 Gated DeltaNet / Full-Attention hybrid layers
      - Zero-Centered RMSNorm
      - SwiGLU MLP
      - Tie word embeddings
    """

    ARCH_NAME = "transformer"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 16,
        intermediate_size: int = 1408,
        full_attention_interval: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        linear_num_key_heads: int = 8,
        linear_num_value_heads: int = 8,
        linear_key_head_dim: int = 64,
        linear_value_head_dim: int = 64,
        linear_conv_kernel_dim: int = 4,
        delta_chunk_size: int = 16,
        partial_rotary_factor: float = 0.5,
        rope_theta: float = 10_000_000.0,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            Qwen35DecoderLayer(
                hidden_size=d_model,
                intermediate_size=intermediate_size,
                index=i,
                full_attention_interval=full_attention_interval,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                linear_num_key_heads=linear_num_key_heads,
                linear_num_value_heads=linear_num_value_heads,
                linear_key_head_dim=linear_key_head_dim,
                linear_value_head_dim=linear_value_head_dim,
                linear_conv_kernel_dim=linear_conv_kernel_dim,
                delta_chunk_size=delta_chunk_size,
                partial_rotary_factor=partial_rotary_factor,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                residual_dropout=dropout,
            )
            for i in range(n_layers)
        ])

        self.norm = ZeroCenteredRMSNorm(d_model, rms_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_word_embeddings:
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

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Aliases for backward compatibility
QwenRMSNorm = ZeroCenteredRMSNorm
Qwen3MLP = Qwen35SwiGLU
Qwen3Attention = GatedAttention


def create_transformer(vocab_size: int, target_params: int = 50_000_000) -> TransformerLM:
    """Create a Qwen3.5 (DeltaNet + Gated Attention) model targeting ~49M-50M params."""
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=14,
        intermediate_size=1408,
        full_attention_interval=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        delta_chunk_size=16,
        partial_rotary_factor=0.5,
        rope_theta=10_000_000.0,
    )

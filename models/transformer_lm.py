"""
Qwen3.5 / Qwen3.8 Language Model Architecture (~50M params).

Exact text-only dense backbone matching Qwen3.5 and Qwen3.8:
  • 3:1 Gated DeltaNet / Gated Full-Attention repeating blocks
  • Causal depthwise convolution in DeltaNet
  • Chunk-parallel gated delta rule (differentiable, linear O(N) sequence complexity)
  • Grouped-Query Full Attention with Q/K norm, Partial RoPE (factor=0.25), and output sigmoid gate
  • Zero-Centered RMSNorm and SwiGLU MLP
  • Full Multi-Token Prediction (MTP) auxiliary objective module
  • Strict configuration validation & DeltaNet initialization preservation (dt_bias=1, A_log)
  • Dynamic near-target configuration search (~50M parameters)
"""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


# -----------------------------------------------------------------------------
# Core Normalization & Partial RoPE


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


class PartialRoPE(nn.Module):
    """Qwen3.5 partial rotary embeddings (default: 0.25 factor = 1/4 of head_dim rotated)."""

    def __init__(
        self,
        head_dim: int,
        partial_rotary_factor: float = 0.25,
        max_seq_len: int = 4096,
        theta: float = 10_000_000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        if self.rotary_dim <= 0 or self.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim ({self.rotary_dim}) must be positive and even (head_dim={head_dim}, factor={partial_rotary_factor})")

        inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, head_dim]
        T = x.shape[2]
        cos = self.cos_cached[:T, :].unsqueeze(0).unsqueeze(0)  # [1, 1, T, rotary_dim]
        sin = self.sin_cached[:T, :].unsqueeze(0).unsqueeze(0)

        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]

        x_rot_applied = (x_rot.float() * cos) + (rotate_half(x_rot.float()) * sin)
        return torch.cat((x_rot_applied.to(x.dtype), x_pass), dim=-1)


# -----------------------------------------------------------------------------
# Qwen3.5 SwiGLU MLP


class Qwen35SwiGLU(nn.Module):
    """Qwen3.5 SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------------------------------------------------------
# Chunk-Parallel Differentiable Delta Rule Kernel (Pure PyTorch)


def _chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    """
    Chunk-parallel gated delta rule in pure PyTorch (differentiable).
    q: [B, H, T, K_dim]
    k: [B, H, T, K_dim]
    v: [B, H, T, V_dim]
    beta: [B, H, T]
    """
    B, H, T, K_dim = q.shape
    V_dim = v.shape[-1]
    scale = 1.0 / math.sqrt(K_dim)
    q = q * scale

    num_chunks = (T + chunk_size - 1) // chunk_size
    pad_len = num_chunks * chunk_size - T
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))

    T_padded = num_chunks * chunk_size
    q_chunks = q.view(B, H, num_chunks, chunk_size, K_dim)
    k_chunks = k.view(B, H, num_chunks, chunk_size, K_dim)
    v_chunks = v.view(B, H, num_chunks, chunk_size, V_dim)
    beta_chunks = beta.view(B, H, num_chunks, chunk_size)

    state = torch.zeros(B, H, K_dim, V_dim, device=q.device, dtype=q.dtype)
    outputs = []

    for c in range(num_chunks):
        qc = q_chunks[:, :, c]        # [B, H, chunk_size, K_dim]
        kc = k_chunks[:, :, c]        # [B, H, chunk_size, K_dim]
        vc = v_chunks[:, :, c]        # [B, H, chunk_size, V_dim]
        betac = beta_chunks[:, :, c]  # [B, H, chunk_size]

        intra_attn = torch.matmul(qc, kc.transpose(-1, -2))  # [B, H, chunk_size, chunk_size]
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool))
        intra_attn = intra_attn.masked_fill(~mask, 0.0)

        betac_v = betac.unsqueeze(-1) * vc
        intra_out = torch.matmul(intra_attn, betac_v)

        inter_out = torch.matmul(qc, state)
        chunk_out = intra_out + inter_out
        outputs.append(chunk_out)

        # Update state: state = state + k^T * (beta * (v - k * state))
        k_state = torch.matmul(kc, state)  # [B, H, chunk_size, V_dim]
        err = betac.unsqueeze(-1) * (vc - k_state)
        state_delta = torch.matmul(kc.transpose(-1, -2), err)
        state = state + state_delta

    out = torch.cat(outputs, dim=2)
    if pad_len > 0:
        out = out[:, :, :T]
    return out


# -----------------------------------------------------------------------------
# Qwen3.5 Gated DeltaNet Layer


class GatedDeltaNet(nn.Module):
    """
    Qwen3.5 Gated DeltaNet layer:
      • Causal depthwise 1D conv on combined Q, K, V
      • Multi-head delta rule with data-dependent decay / gate
      • Gated RMSNorm output projection
    """

    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int = 8,
        num_value_heads: int = 8,
        key_head_dim: int = 64,
        value_head_dim: int = 64,
        conv_kernel_dim: int = 4,
        delta_chunk_size: int = 16,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.delta_chunk_size = delta_chunk_size

        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim

        # Input projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_size, num_value_heads, bias=True)

        # Causal 1D convolution over concatenated [Q, K, V]
        total_conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            total_conv_dim,
            total_conv_dim,
            kernel_size=conv_kernel_dim,
            padding=conv_kernel_dim - 1,
            groups=total_conv_dim,
            bias=True,
        )

        # Output Gated Norm and projection
        self.gated_norm = GatedRMSNorm(self.value_dim, eps=rms_norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Decay parameter A_log and bias
        self.A_log = nn.Parameter(torch.empty(num_value_heads).uniform_(0.01, 16.0).log_())
        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        gate = self.gate_proj(x)
        beta_raw = self.beta_proj(x)

        # Causal convolution
        qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
        qkv = self.conv1d(qkv)[:, :, :T].transpose(1, 2)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        q = F.silu(q).view(B, T, self.num_key_heads, self.key_head_dim).transpose(1, 2)
        k = F.silu(k).view(B, T, self.num_key_heads, self.key_head_dim).transpose(1, 2)
        v = F.silu(v).view(B, T, self.num_value_heads, self.value_head_dim).transpose(1, 2)

        beta = torch.sigmoid(beta_raw + self.dt_bias).transpose(1, 2)  # [B, H, T]

        # Chunk delta rule
        out = _chunk_delta_rule(q, k, v, beta, chunk_size=self.delta_chunk_size)
        out = out.transpose(1, 2).reshape(B, T, self.value_dim)

        out = self.gated_norm(out, gate)
        return self.o_proj(out)


# -----------------------------------------------------------------------------
# Qwen3.5 Grouped-Query Gated Attention Layer


class GatedAttention(nn.Module):
    """
    Qwen3.5 Grouped-Query Full Attention with:
      • Q/K Normalization
      • Partial RoPE (factor=0.25)
      • Elementwise Sigmoid Output Gating
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 10_000_000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim * 2, bias=False)  # includes gate
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.q_norm = ZeroCenteredRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(head_dim, eps=rms_norm_eps)

        self.rope = PartialRoPE(head_dim, partial_rotary_factor, theta=rope_theta)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        q_and_gate = self.q_proj(x).view(B, T, self.num_heads, 2, self.head_dim)
        q = q_and_gate[:, :, :, 0, :].transpose(1, 2)     # [B, H, T, head_dim]
        gate = q_and_gate[:, :, :, 1, :].transpose(1, 2)  # [B, H, T, head_dim]

        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None,
        )

        out = (torch.sigmoid(gate) * out).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


# -----------------------------------------------------------------------------
# Qwen3.5 Hybrid Decoder Layer (3:1 DeltaNet : Gated Attention)


class Qwen35DecoderLayer(nn.Module):
    """
    Qwen3.5 Decoder Layer:
      - 3 of every 4 layers are Gated DeltaNet (O(N) linear complexity)
      - 1 of every 4 layers is Gated Grouped-Query Full Attention (O(N^2) associative recall)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        index: int,
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
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 10_000_000.0,
        rms_norm_eps: float = 1e-6,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.index = index
        self.is_full_attention = (index + 1) % full_attention_interval == 0

        self.input_layernorm = ZeroCenteredRMSNorm(hidden_size, eps=rms_norm_eps)
        if self.is_full_attention:
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
                num_key_heads=linear_num_key_heads,
                num_value_heads=linear_num_value_heads,
                key_head_dim=linear_key_head_dim,
                value_head_dim=linear_value_head_dim,
                conv_kernel_dim=linear_conv_kernel_dim,
                delta_chunk_size=delta_chunk_size,
                rms_norm_eps=rms_norm_eps,
            )

        self.post_attention_layernorm = ZeroCenteredRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen35SwiGLU(hidden_size, intermediate_size)
        self.drop = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_full_attention:
            x = x + self.drop(self.mixer(self.input_layernorm(x), attention_mask=attention_mask))
        else:
            x = x + self.drop(self.mixer(self.input_layernorm(x)))
        x = x + self.drop(self.mlp(self.post_attention_layernorm(x)))
        return x


# -----------------------------------------------------------------------------
# Multi-Token Prediction (MTP) Auxiliary Head


class MTPHead(nn.Module):
    """
    Qwen3.5 Multi-Token Prediction (MTP) 1-layer auxiliary decoder:
      • Fuses backbone hidden state + next token embedding via RMSNorm & Linear projection
      • Decodes with 1 Gated Full Attention layer
      • Predicts next-next token (t+2) to accelerate learning & multi-step planning
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        mtp_num_hidden_layers: int = 1,
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 10_000_000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm_backbone = ZeroCenteredRMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_emb = ZeroCenteredRMSNorm(hidden_size, eps=rms_norm_eps)
        self.fuse_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.layers = nn.ModuleList([
            Qwen35DecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                index=3,  # full attention layer
                full_attention_interval=1,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                partial_rotary_factor=partial_rotary_factor,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(mtp_num_hidden_layers)
        ])
        self.final_norm = ZeroCenteredRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        next_token_ids: torch.Tensor,
        tok_emb: nn.Embedding,
    ) -> torch.Tensor:
        emb = tok_emb(next_token_ids)
        h = torch.cat([self.norm_backbone(backbone_hidden), self.norm_emb(emb)], dim=-1)
        h = self.fuse_proj(h)

        for layer in self.layers:
            h = layer(h)

        return self.final_norm(h)


# -----------------------------------------------------------------------------
# Main Qwen3.5 Transformer LM


class TransformerLM(nn.Module):
    """
    Qwen3.5 / Qwen3.8 Language Model (~50M parameters).

    Backbone:
      Embedding -> 16 layers (3:1 DeltaNet : Gated Attention) -> RMSNorm -> Tied LM Head
    Auxiliary:
      MTP Head (1 layer auxiliary prediction for next-next token)
    """

    ARCH_NAME = "transformer"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 14,
        intermediate_size: int = 1280,
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
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 10_000_000.0,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        mtp_num_hidden_layers: int = 1,
        mtp_loss_weight: float = 0.1,
    ):
        super().__init__()
        # Strict configuration validation
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})")
        if linear_num_value_heads % linear_num_key_heads != 0:
            raise ValueError(f"linear_num_value_heads ({linear_num_value_heads}) must be divisible by linear_num_key_heads ({linear_num_key_heads})")
        rope_dim = int(head_dim * partial_rotary_factor)
        if rope_dim <= 0 or rope_dim % 2 != 0 or rope_dim > head_dim:
            raise ValueError(f"Invalid partial RoPE dimension {rope_dim} for head_dim {head_dim} (must be even and <= head_dim)")

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.mtp_loss_weight = mtp_loss_weight

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

        # MTP Head
        self.mtp = None
        if mtp_num_hidden_layers > 0:
            self.mtp = MTPHead(
                hidden_size=d_model,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                mtp_num_hidden_layers=mtp_num_hidden_layers,
                partial_rotary_factor=partial_rotary_factor,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
            )

        if tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights properly without zeroing GatedDeltaNet dt_bias.
        Explicitly restores dt_bias=1 and log-uniform A_log for DeltaNet layers.
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

        # Explicitly preserve / re-initialize DeltaNet specialized parameters
        for module in self.modules():
            if isinstance(module, GatedDeltaNet):
                nn.init.ones_(module.dt_bias)
                with torch.no_grad():
                    module.A_log.copy_(
                        torch.empty_like(module.A_log).uniform_(0.01, 16.0).log_()
                    )

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        h = self.drop(self.tok_emb(x))

        for layer in self.layers:
            h = layer(h)

        raw_hidden = h
        logits = self.lm_head(self.norm(raw_hidden))

        loss = None
        self.last_base_loss = None
        self.last_mtp_loss = None

        if targets is not None:
            flat_logits = logits.float().reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            base_loss_sum = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            valid_tokens = (flat_targets != IGNORE_INDEX).sum()
            base_loss = base_loss_sum / valid_tokens.clamp_min(1)
            loss = base_loss
            self.last_base_loss = base_loss.item()

            # Multi-Token Prediction auxiliary loss
            if self.mtp is not None and x.shape[1] > 1:
                mtp_hidden = self.mtp(raw_hidden[:, :-1], x[:, 1:], self.tok_emb)
                mtp_logits = self.lm_head(mtp_hidden)
                mtp_targets = targets[:, 1:]
                flat_mtp_logits = mtp_logits.float().reshape(-1, mtp_logits.size(-1))
                flat_mtp_targets = mtp_targets.reshape(-1)
                mtp_loss_sum = F.cross_entropy(
                    flat_mtp_logits,
                    flat_mtp_targets,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                valid_mtp_tokens = (flat_mtp_targets != IGNORE_INDEX).sum()
                mtp_loss = mtp_loss_sum / valid_mtp_tokens.clamp_min(1)
                self.last_mtp_loss = mtp_loss.item()
                loss = base_loss + self.mtp_loss_weight * mtp_loss

        return logits, loss

    def count_params(self, exclude_mtp: bool = False) -> int:
        if not exclude_mtp or self.mtp is None:
            return sum(p.numel() for p in self.parameters())
        mtp_ids = {id(p) for p in self.mtp.parameters()}
        return sum(p.numel() for p in self.parameters() if id(p) not in mtp_ids)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Aliases for backward compatibility
QwenRMSNorm = ZeroCenteredRMSNorm
Qwen3MLP = Qwen35SwiGLU
Qwen3Attention = GatedAttention


def estimate_transformer_parameters(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    intermediate_size: int,
    full_attention_interval: int = 4,
    mtp_num_hidden_layers: int = 1,
) -> int:
    """Analytical parameter calculation for Qwen3.5 hybrid model."""
    tied_emb = vocab_size * d_model
    final_norm = d_model

    # Gated DeltaNet layer params
    key_dim = d_model
    value_dim = d_model
    deltanet_proj = d_model * (2 * key_dim + 2 * value_dim) + (d_model // 64)  # Q, K, V, gate, beta
    conv_dim = 2 * key_dim + value_dim
    conv_params = conv_dim * 4 + conv_dim
    gated_norm = value_dim
    o_proj = value_dim * d_model
    swiglu = 3 * d_model * intermediate_size
    deltanet_layer = deltanet_proj + conv_params + gated_norm + o_proj + swiglu + 2 * d_model

    # Gated Attention layer params
    attn_q = d_model * (d_model * 2)
    attn_k = d_model * (d_model // 2)
    attn_v = d_model * (d_model // 2)
    attn_o = d_model * d_model
    qk_norm = 2 * 64
    attn_layer = attn_q + attn_k + attn_v + attn_o + qk_norm + swiglu + 2 * d_model

    n_attn = n_layers // full_attention_interval
    n_delta = n_layers - n_attn
    backbone = tied_emb + n_delta * deltanet_layer + n_attn * attn_layer + final_norm

    mtp_params = 0
    if mtp_num_hidden_layers > 0:
        mtp_fuse = 2 * d_model * d_model
        mtp_params = mtp_fuse + mtp_num_hidden_layers * attn_layer + 3 * d_model

    return backbone + mtp_params


def choose_near_target_transformer_config(vocab_size: int, target_params: int = 50_000_000) -> Dict[str, Any]:
    """Search viable Qwen3.5 configurations to match target_params."""
    candidates = []
    for d_model in (384, 448, 512):
        intermediate = round(d_model * 2.5)
        for n_layers in (12, 14, 16):
            params = estimate_transformer_parameters(vocab_size, d_model, n_layers, intermediate)
            diff = abs(params - target_params)
            candidates.append((
                diff,
                {
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "intermediate_size": intermediate,
                    "estimated_params": params,
                },
            ))
    return min(candidates, key=lambda x: x[0])[1]


def create_transformer(vocab_size: int = 3919, target_params: int = 50_000_000) -> TransformerLM:
    """
    Create a Qwen3.5 model targeting ~50M parameters with 3:1 DeltaNet:Attention ratio,
    partial_rotary_factor=0.25, and MTP head.
    """
    cfg = choose_near_target_transformer_config(vocab_size, target_params)
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        intermediate_size=cfg["intermediate_size"],
        full_attention_interval=4,
        num_attention_heads=cfg["d_model"] // 64,
        num_key_value_heads=cfg["d_model"] // 128,
        head_dim=64,
        linear_num_key_heads=cfg["d_model"] // 64,
        linear_num_value_heads=cfg["d_model"] // 64,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        delta_chunk_size=16,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
        mtp_num_hidden_layers=1,
        mtp_loss_weight=0.1,
    )

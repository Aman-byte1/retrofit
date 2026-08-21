"""
HRM-Text: Hierarchical Reasoning Model for Language (~50M params).

Official PyTorch adaptation of Sapient Intelligence's HRM-Text:
  https://github.com/sapientinc/HRM-Text (Apache-2.0)
  "HRM-Text: Efficient Pretraining Beyond Scaling" (Wang et al., 2026)

Key Architecture Details:
  • Dual-timescale HRM recurrence: H2L3 by default
  • Separate high-level (H) and low-level (L) Transformer modules
  • Additive H/L state injection (z_l + z_h and z_h + z_l)
  • MagicNorm: parameterless Pre-RMSNorm blocks + module-exit RMSNorm
  • Warmup deep credit assignment (truncated BPTT horizon: bp_min -> bp_max)
  • PrefixLM attention masking (bidirectional prompt, causal response)
  • Sigmoid-gated multi-head self-attention with full RoPE
  • Gradient checkpointing support for efficient memory reuse
  • SwiGLU FFN and truncated LeCun-normal initialization
  • Scaled, untied token embedding and linear output head
"""

import math
import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils


IGNORE_INDEX = -100


def find_multiple(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@torch.no_grad()
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Fast approximate truncated normal matching official HRM-Text implementation."""
    return tensor.normal_().fmod_(3.0).mul_(1.014762601732121 * std)


class ParameterlessRMSNorm(nn.Module):
    """Parameterless RMSNorm (MagicNorm component)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), eps=self.eps)


class LinearInit(nn.Module):
    """Bias-free Linear layer with truncated LeCun-normal initialization."""

    def __init__(self, in_features: int, out_features: int, init_std: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        trunc_normal_init_(self.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class ScaledEmbedding(nn.Module):
    """Scaled token embedding matching official HRM-Text."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        init_std = 1.0 / math.sqrt(hidden_size)
        self.scale = 1.0 / init_std
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        trunc_normal_init_(self.weight, std=init_std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.scale * F.embedding(ids, self.weight)


class RotaryEmbedding(nn.Module):
    """Full Rotary Position Embedding for HRM attention."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {head_dim}")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", embedding.cos(), persistent=False)
        self.register_buffer("sin_cached", embedding.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = cos_sin
    result = x.float() * cos.unsqueeze(2) + rotate_half(x.float()) * sin.unsqueeze(2)
    return result.to(x.dtype)


def build_prefix_lm_mask(
    prefix_lengths: torch.Tensor,
    valid_lengths: torch.Tensor,
    sequence_length: int,
    mode: str = "prefix",
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    Build PrefixLM attention mask for PyTorch SDPA:
      - Prompt tokens attend bidirectionally to all other prompt tokens.
      - Response tokens attend causally to previous prompt + response tokens.
      - Padding tokens are excluded.
    Returns (mask, is_causal).
    """
    if mode == "causal" and bool(torch.all(valid_lengths == sequence_length)):
        return None, True

    device = prefix_lengths.device
    query_pos = torch.arange(sequence_length, device=device).view(1, sequence_length, 1)
    key_pos = torch.arange(sequence_length, device=device).view(1, 1, sequence_length)

    # Causal lower-triangular mask
    causal = key_pos <= query_pos

    if mode == "prefix":
        prefix = prefix_lengths.view(-1, 1, 1)
        bidirectional_prefix = (query_pos < prefix) & (key_pos < prefix)
        allowed = causal | bidirectional_prefix
    else:
        allowed = causal

    valid = valid_lengths.view(-1, 1, 1)
    allowed = allowed & (query_pos < valid) & (key_pos < valid)

    # Return boolean mask of shape [batch, 1, seq_len, seq_len]
    return allowed.unsqueeze(1), False


class GatedSelfAttention(nn.Module):
    """Full Multi-Head Attention with elementwise sigmoid output gate (HRM-Text)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        in_std = 1.0 / math.sqrt(hidden_size)
        self.gqkv_proj = LinearInit(hidden_size, 4 * hidden_size, init_std=in_std)
        self.o_proj = LinearInit(hidden_size, hidden_size, init_std=in_std)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch, time, _ = x.shape
        gqkv = self.gqkv_proj(x).view(batch, time, 4, self.num_heads, self.head_dim)
        gate, query, key, value = gqkv.unbind(dim=2)

        query = apply_rope(query, cos_sin).transpose(1, 2)
        key = apply_rope(key, cos_sin).transpose(1, 2)
        value = value.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal if attention_mask is None else False,
        )
        output = output.transpose(1, 2)
        output = (torch.sigmoid(gate) * output).reshape(batch, time, -1)
        return self.o_proj(output)


class HRMSwiGLU(nn.Module):
    """HRM-Text SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = LinearInit(hidden_size, 2 * intermediate_size, init_std=1.0 / math.sqrt(hidden_size))
        self.down_proj = LinearInit(intermediate_size, hidden_size, init_std=1.0 / math.sqrt(intermediate_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class HRMTransformerBlock(nn.Module):
    """Pre-Norm Transformer Block with parameterless RMSNorm (MagicNorm)."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.attention = GatedSelfAttention(hidden_size, num_heads)
        self.mlp = HRMSwiGLU(hidden_size, intermediate_size)
        self.norm = ParameterlessRMSNorm(norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm(x), cos_sin, attention_mask, is_causal)
        return x + self.mlp(self.norm(x))


class MagicNormTransformer(nn.Module):
    """Sequence of Transformer Blocks capped by an exit ParameterlessRMSNorm (MagicNorm)."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10_000.0,
        norm_eps: float = 1e-6,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.rope = RotaryEmbedding(hidden_size // num_heads, max_seq_len, rope_theta)
        self.layers = nn.ModuleList([
            HRMTransformerBlock(hidden_size, num_heads, intermediate_size, norm_eps)
            for _ in range(num_layers)
        ])
        self.final_norm = ParameterlessRMSNorm(norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        cos_sin = self.rope(position_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                def layer_forward(hidden, _layer=layer):
                    return _layer(hidden, cos_sin, attention_mask, is_causal)
                x = checkpoint_utils.checkpoint(layer_forward, x, use_reentrant=False)
            else:
                x = layer(x, cos_sin, attention_mask, is_causal)
        return self.final_norm(x)


class HRMCore(nn.Module):
    """
    Dual-timescale hierarchical recurrence engine with Truncated BPTT:
      - H_level: slow abstract planning (processes H_cycles times)
      - L_level: fast detailed token processing (processes L_cycles times per H-cycle)
    """

    def __init__(
        self,
        hidden_size: int,
        layers_per_module: int,
        num_heads: int,
        intermediate_size: int,
        H_cycles: int = 2,
        L_cycles: int = 3,
        max_seq_len: int = 4096,
        rope_theta: float = 10_000.0,
        norm_eps: float = 1e-6,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        self.H_level = MagicNormTransformer(
            hidden_size, layers_per_module, num_heads, intermediate_size, max_seq_len, rope_theta, norm_eps,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.L_level = MagicNormTransformer(
            hidden_size, layers_per_module, num_heads, intermediate_size, max_seq_len, rope_theta, norm_eps,
            gradient_checkpointing=gradient_checkpointing,
        )

        z_l = torch.empty(hidden_size, dtype=torch.bfloat16)
        trunc_normal_init_(z_l, std=1.0)
        self.register_buffer("zL_init", z_l, persistent=True)

    def forward(
        self,
        embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        bp_steps: Optional[int] = None,
    ) -> torch.Tensor:
        z_h = embeddings
        z_l = self.zL_init.to(dtype=embeddings.dtype).view(1, 1, -1).expand_as(embeddings)

        total_recurrent_steps = self.H_cycles * (self.L_cycles + 1)
        if bp_steps is None:
            bp_steps = 5

        h_bp_steps = min(self.H_cycles, max(1, bp_steps - 1))
        l_bp_steps = max(1, bp_steps - h_bp_steps)
        total_l_steps = self.H_cycles * self.L_cycles
        outer_grad_enabled = torch.is_grad_enabled()

        for h_index in range(self.H_cycles):
            first_l = h_index * self.L_cycles
            for l_index in range(first_l, first_l + self.L_cycles):
                enabled = outer_grad_enabled and (l_index >= total_l_steps - l_bp_steps)
                with torch.set_grad_enabled(enabled):
                    z_l = self.L_level(
                        z_l + z_h, position_ids, attention_mask, is_causal
                    )
            enabled = outer_grad_enabled and (h_index >= self.H_cycles - h_bp_steps)
            with torch.set_grad_enabled(enabled):
                z_h = self.H_level(
                    z_h + z_l, position_ids, attention_mask, is_causal
                )
        return z_h


class HRMLM(nn.Module):
    """
    Hierarchical Reasoning Model for Language (HRM-Text).
    
    Complete architecture:
      ScaledEmbedding -> HRMCore (H2L3 dual-timescale recurrence) -> Linear LM Head
    """

    ARCH_NAME = "hrm"

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        total_layers: int = 24,
        num_heads: int = 6,
        expansion: float = 4.0,
        H_cycles: int = 2,
        L_cycles: int = 3,
        max_seq_len: int = 4096,
        rope_theta: float = 10_000.0,
        norm_eps: float = 1e-6,
        bp_min_steps: int = 2,
        bp_max_steps: int = 5,
        bp_warmup_ratio: float = 0.2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if total_layers % 2 != 0:
            raise ValueError(f"total_layers ({total_layers}) must be even (split evenly between H and L modules)")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        if (hidden_size // num_heads) % 2 != 0:
            raise ValueError(f"head_dim ({hidden_size // num_heads}) must be even for RoPE")

        total_recurrent_steps = H_cycles * (L_cycles + 1)
        if not (2 <= bp_min_steps <= bp_max_steps <= total_recurrent_steps):
            raise ValueError(f"Invalid truncated-BPTT horizon range [{bp_min_steps}, {bp_max_steps}] for {total_recurrent_steps} recurrent steps")

        self.vocab_size = vocab_size
        self.d_model = hidden_size
        self.total_layers = total_layers
        self.layers_per_module = total_layers // 2
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bp_min_steps = bp_min_steps
        self.bp_max_steps = bp_max_steps
        self.bp_warmup_ratio = bp_warmup_ratio
        self.gradient_checkpointing = gradient_checkpointing

        # Intermediate size matching official formula
        raw_intermediate = round(expansion * hidden_size * 2.0 / 3.0)
        self.intermediate_size = find_multiple(raw_intermediate, 256)

        # Embedding & Core & Head (untied as in HRM-Text)
        self.tok_emb = ScaledEmbedding(vocab_size, hidden_size)
        self.core = HRMCore(
            hidden_size=hidden_size,
            layers_per_module=self.layers_per_module,
            num_heads=num_heads,
            intermediate_size=self.intermediate_size,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.lm_head = LinearInit(hidden_size, vocab_size, init_std=1.0 / math.sqrt(hidden_size))

    def backward_horizon(self, step: int, total_steps: int) -> int:
        """Warmup deep credit assignment."""
        warmup_steps = total_steps * self.bp_warmup_ratio
        progress = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
        span = self.bp_max_steps - self.bp_min_steps
        return self.bp_min_steps + int(progress * span)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        valid_lengths: Optional[torch.Tensor] = None,
        attention_mode: str = "prefix",
        bp_steps: Optional[int] = None,
    ):
        batch, time = x.shape
        if time > self.max_seq_len:
            raise ValueError(f"Sequence length {time} exceeds max_seq_len {self.max_seq_len}")

        if bp_steps is None:
            bp_steps = self.bp_max_steps

        if prefix_lengths is None:
            prefix_lengths = torch.zeros(batch, dtype=torch.long, device=x.device)
        if valid_lengths is None:
            valid_lengths = torch.full((batch,), time, dtype=torch.long, device=x.device)

        attention_mask, is_causal = build_prefix_lm_mask(
            prefix_lengths, valid_lengths, time, mode=attention_mode
        )
        position_ids = torch.arange(time, device=x.device).unsqueeze(0).expand(batch, -1)

        embeddings = self.tok_emb(x)
        hidden = self.core(
            embeddings, position_ids, attention_mask=attention_mask, is_causal=is_causal, bp_steps=bp_steps
        )
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

        return logits, loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias for backward compatibility
HRMTextModel = HRMLM


def estimate_hrm_parameters(vocab_size: int, hidden_size: int, total_layers: int, expansion: float = 4.0) -> int:
    """Exact trainable parameter count calculation for HRM."""
    raw_ff = round(expansion * hidden_size * 2.0 / 3.0)
    ff = find_multiple(raw_ff, 256)
    embedding_and_head = 2 * vocab_size * hidden_size
    attention_per_block = 5 * hidden_size * hidden_size
    swiglu_per_block = 3 * hidden_size * ff
    return embedding_and_head + total_layers * (attention_per_block + swiglu_per_block)


def choose_near_target_hrm_config(vocab_size: int, target_params: int = 50_000_000) -> Dict[str, Any]:
    """Search viable HRM configurations and pick the one closest to target_params."""
    candidates = []
    for hidden in range(256, 769, 64):
        heads = hidden // 64
        for total_layers in range(12, 30, 2):
            params = estimate_hrm_parameters(vocab_size, hidden, total_layers)
            diff = abs(params - target_params)
            candidates.append((diff, params, {"hidden_size": hidden, "total_layers": total_layers, "num_heads": heads}))
    return min(candidates, key=lambda x: x[0])[2]


def create_hrm(vocab_size: int, target_params: int = 50_000_000) -> HRMLM:
    """Create an HRM-Text model targeting approximately target_params (~50M)."""
    cfg = choose_near_target_hrm_config(vocab_size, target_params)
    return HRMLM(
        vocab_size=vocab_size,
        hidden_size=cfg["hidden_size"],
        total_layers=cfg["total_layers"],
        num_heads=cfg["num_heads"],
        expansion=4.0,
        H_cycles=2,
        L_cycles=3,
        max_seq_len=4096,
        bp_min_steps=2,
        bp_max_steps=5,
    )

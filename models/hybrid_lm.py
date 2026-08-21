"""
Hybrid Mamba-Transformer Language Model (~50M params).

Interleaves Mamba SSM layers with Qwen3.5 Attention layers:
  • Mamba layers: Linear O(N) sequence processing with continuous state compression
  • Attention layers: Exact Qwen3.5 Gated Attention with QK-Norm and Partial RoPE (factor=0.25)
  • Consistent SwiGLU MLP across all blocks
  • Idempotent residual depth scaling (1 / sqrt(2 * n_layers)) across all mixers and FFNs
  • Preserves Mamba's specialized A_log, D, and dt_proj initialization
  • Dynamic near-target configuration search (~50M parameters)
"""

import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_lm import QwenRMSNorm, Qwen3MLP, Qwen3Attention
from .mamba_lm import MambaBlock


IGNORE_INDEX = -100


class HybridMambaLayer(nn.Module):
    """Pre-norm Mamba + SwiGLU layer for the hybrid model."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        backend: str = "auto",
    ):
        super().__init__()
        self.norm = QwenRMSNorm(d_model)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backend=backend,
        )
        self.ffn_norm = QwenRMSNorm(d_model)
        self.ffn = Qwen3MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba(self.norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HybridAttentionLayer(nn.Module):
    """Pre-norm Qwen3.5 attention + SwiGLU layer for the hybrid model."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 10_000_000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = QwenRMSNorm(d_model, eps=rms_norm_eps)
        self.attn = Qwen3Attention(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads // 2,
            head_dim=d_model // n_heads,
            partial_rotary_factor=partial_rotary_factor,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )
        self.ffn_norm = QwenRMSNorm(d_model, eps=rms_norm_eps)
        self.ffn = Qwen3MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HybridMambaTransformerLM(nn.Module):
    """
    Hybrid Mamba-Transformer Language Model (~50M params).
    
    Architecture:
      Embedding → [Mamba+FFN, Mamba+FFN, QwenAttn+FFN, ...] → RMSNorm → Tied LM Head
    """

    ARCH_NAME = "hybrid"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 15,
        n_heads: int = 8,
        d_ff: int = 1408,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        attn_every: int = 3,
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 10_000_000.0,
        dropout: float = 0.0,
        mamba_backend: str = "auto",
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if attn_every <= 0:
            raise ValueError(f"attn_every must be positive, got {attn_every}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_heads % 2 != 0:
            raise ValueError(f"n_heads ({n_heads}) must be even for GQA num_key_value_heads=n_heads//2")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")
        if mamba_backend not in {"auto", "official", "torch"}:
            raise ValueError(f"mamba_backend must be 'auto', 'official', or 'torch', got {mamba_backend!r}")

        head_dim = d_model // n_heads
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError(f"Partial RoPE dimension must be positive and even, got {rotary_dim}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.attn_every = attn_every

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.layer_types = []

        for i in range(n_layers):
            if (i + 1) % attn_every == 0:
                self.layers.append(
                    HybridAttentionLayer(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        partial_rotary_factor=partial_rotary_factor,
                        rope_theta=rope_theta,
                    )
                )
                self.layer_types.append("attention")
            else:
                self.layers.append(
                    HybridMambaLayer(
                        d_model=d_model,
                        d_ff=d_ff,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        backend=mamba_backend,
                    )
                )
                self.layer_types.append("mamba")

        self.norm = QwenRMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """
        Initialize embeddings and linear projections with idempotent depth scaling.
        Preserves Mamba's specialized A_log, D, and dt_proj initialization.
        """
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        residual_scale = math.sqrt(2 * len(self.layers))

        with torch.no_grad():
            for layer in self.layers:
                # Attention projections
                if hasattr(layer, "attn"):
                    nn.init.normal_(layer.attn.q_proj.weight, mean=0.0, std=0.02)
                    nn.init.normal_(layer.attn.k_proj.weight, mean=0.0, std=0.02)
                    nn.init.normal_(layer.attn.v_proj.weight, mean=0.0, std=0.02)
                    nn.init.kaiming_uniform_(layer.attn.o_proj.weight, a=math.sqrt(5))
                    layer.attn.o_proj.weight.div_(residual_scale)

                # Mamba projections
                if hasattr(layer, "mamba"):
                    block = layer.mamba
                    mixer = block.mamba if block._use_official else block
                    if hasattr(mixer, "out_proj") and hasattr(mixer.out_proj, "weight"):
                        nn.init.kaiming_uniform_(mixer.out_proj.weight, a=math.sqrt(5))
                        mixer.out_proj.weight.div_(residual_scale)

                # FFN projections
                if hasattr(layer, "ffn"):
                    nn.init.normal_(layer.ffn.gate_proj.weight, mean=0.0, std=0.02)
                    nn.init.normal_(layer.ffn.up_proj.weight, mean=0.0, std=0.02)
                    nn.init.kaiming_uniform_(layer.ffn.down_proj.weight, a=math.sqrt(5))
                    layer.ffn.down_proj.weight.div_(residual_scale)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        h = self.drop(self.tok_emb(x))

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            flat_logits = logits.float().reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            loss_sum = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            valid_tokens = (flat_targets != IGNORE_INDEX).sum()
            loss = loss_sum / valid_tokens.clamp_min(1)

        return logits, loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_breakdown(self) -> Dict[str, int]:
        n_mamba = sum(1 for t in self.layer_types if t == "mamba")
        n_attn = sum(1 for t in self.layer_types if t == "attention")
        return {"mamba_layers": n_mamba, "attention_layers": n_attn, "total": len(self.layer_types)}


def configure_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Build AdamW optimizer respecting Mamba's _no_weight_decay attributes."""
    decay = []
    no_decay = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if getattr(p, "_no_weight_decay", False) or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


def estimate_hybrid_parameters(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    attn_every: int = 3,
    expand: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
) -> int:
    """Analytical parameter count for Hybrid Mamba-Transformer."""
    d_inner = d_model * expand
    dt_rank = math.ceil(d_model / 16)
    head_dim = d_model // n_heads
    kv_heads = n_heads // 2

    # Mamba block parameters
    in_proj = d_model * (2 * d_inner)
    conv1d = d_inner * d_conv + d_inner
    x_proj = d_inner * (dt_rank + 2 * d_state)
    dt_proj = dt_rank * d_inner + d_inner
    A_log = d_inner * d_state
    D = d_inner
    out_proj = d_inner * d_model
    mamba_block = in_proj + conv1d + x_proj + dt_proj + A_log + D + out_proj

    # Attention block parameters (Qwen gated attention)
    q_proj = d_model * (n_heads * head_dim * 2)
    k_proj = d_model * (kv_heads * head_dim)
    v_proj = d_model * (kv_heads * head_dim)
    o_proj = (n_heads * head_dim) * d_model
    qk_norms = 2 * head_dim
    attn_block = q_proj + k_proj + v_proj + o_proj + qk_norms

    # FFN block parameters (SwiGLU)
    ffn_block = 3 * d_model * d_ff

    # Layer totals
    mamba_layer = d_model + mamba_block + d_model + ffn_block
    attn_layer = d_model + attn_block + d_model + ffn_block

    n_attn = n_layers // attn_every
    n_mamba = n_layers - n_attn

    tied_embedding = vocab_size * d_model
    final_norm = d_model

    return tied_embedding + n_mamba * mamba_layer + n_attn * attn_layer + final_norm


def choose_near_target_hybrid_config(vocab_size: int, target_params: int = 50_000_000) -> Dict[str, Any]:
    """Search viable Hybrid configurations to match target_params."""
    candidates = []
    for d_model, n_heads in ((384, 6), (448, 8), (512, 8)):
        d_ff = round(d_model * 2.75)
        for n_layers in range(12, 24, 3):
            params = estimate_hybrid_parameters(vocab_size, d_model, n_layers, n_heads, d_ff)
            diff = abs(params - target_params)
            candidates.append((
                diff,
                {
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "d_ff": d_ff,
                    "estimated_params": params,
                },
            ))
    return min(candidates, key=lambda x: x[0])[1]


def create_hybrid(vocab_size: int = 3919, target_params: int = 50_000_000, mamba_backend: str = "auto") -> HybridMambaTransformerLM:
    """Create a Hybrid Mamba-Transformer LM targeting ~50M parameters."""
    cfg = choose_near_target_hybrid_config(vocab_size, target_params)
    return HybridMambaTransformerLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        d_state=16,
        d_conv=4,
        expand=2,
        attn_every=3,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
        mamba_backend=mamba_backend,
    )

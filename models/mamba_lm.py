"""
Mamba (Selective State Space Model) Language Model (~50M params).

Implementation aligned with official Mamba:
  "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
  https://github.com/state-spaces/mamba

Features:
  • Official `mamba_ssm` backend with high-performance CUDA fused selective scan
  • Mathematically faithful pure-PyTorch fallback with exact dt_rank = ceil(d_model / 16)
  • Specialized Mamba initialization (dt_proj log-spaced inverse softplus, S4D A_log)
  • Idempotent output projection depth-scaling (1 / sqrt(n_layers)) for both backends
  • Protection of A_log and D parameters from weight decay (_no_weight_decay = True)
  • Safety against reinitialization (_no_reinit = True on dt_proj.bias)
  • Self-contained RMSNorm (eps=1e-5)
  • Robust loss calculation and analytical parameter estimation
  • Dynamic near-target configuration search (~50M parameters)
"""

import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


class RMSNorm(nn.Module):
    """Self-contained Root Mean Square Layer Normalization (standard Mamba norm)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class MambaBlock(nn.Module):
    """
    Mamba Selective SSM block.
    
    Supports:
      - Official `mamba_ssm.Mamba` kernel when available.
      - Exact PyTorch fallback with full dt_rank = ceil(d_model / 16) and official initialization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        backend: str = "auto",
    ):
        super().__init__()
        if backend not in {"auto", "official", "torch"}:
            raise ValueError(f"backend must be 'auto', 'official', or 'torch', got {backend!r}")
        if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError("Mamba dimensions (d_model, d_state, d_conv, expand) must all be positive")
        if not (0 < dt_min <= dt_max):
            raise ValueError(f"Require 0 < dt_min <= dt_max, got dt_min={dt_min}, dt_max={dt_max}")
        if dt_init_floor <= 0:
            raise ValueError(f"dt_init_floor must be positive, got {dt_init_floor}")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank
        if self.dt_rank <= 0:
            raise ValueError(f"dt_rank must be positive, got {self.dt_rank}")

        self.backend = backend
        self._use_official = False

        if backend in {"auto", "official"}:
            try:
                from mamba_ssm import Mamba
                self.mamba = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=self.dt_rank,
                )
                self._use_official = True
            except ImportError as exc:
                if backend == "official":
                    raise ImportError("Install `mamba-ssm` or use backend='auto'/'torch'") from exc
                self._build_fallback(dt_min, dt_max, dt_init_floor)
        else:
            self._build_fallback(dt_min, dt_max, dt_init_floor)

    def _build_fallback(self, dt_min: float, dt_max: float, dt_init_floor: float):
        """Build exact pure-PyTorch selective SSM with official Mamba parameterization."""
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Depthwise 1D causal convolution (with bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # Selective projection: projects to (dt_rank + 2 * d_state)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # Time-step delta projection: dt_rank -> d_inner (with bias)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Specialized dt_proj initialization
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt_proj.bias to inverse softplus of log-uniform sampled values
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D diagonal state matrix A (log-parameterized: A = -exp(A_log))
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # Skip parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _ssm_scan(self, x: torch.Tensor, dt: torch.Tensor, B_ssm: torch.Tensor, C_ssm: torch.Tensor) -> torch.Tensor:
        """
        Differentiable sequential selective scan in pure PyTorch.
        x: [B, L, d_inner]
        dt: [B, L, d_inner]
        B_ssm: [B, L, d_state]
        C_ssm: [B, L, d_state]
        """
        B, L, D = x.shape
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=torch.float32)
        ys = []

        for t in range(L):
            dt_t = dt[:, t].float()    # [B, d_inner]
            B_t = B_ssm[:, t].float()  # [B, d_state]
            C_t = C_ssm[:, t].float()  # [B, d_state]
            x_t = x[:, t].float()      # [B, d_inner]

            # Mamba selective-scan discretization:
            # exponential discretization for A and delta-scaled input B
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))     # [B, d_inner, d_state]
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)              # [B, d_inner, d_state]

            # Recurrent state update: h = dA * h + dB * x
            h = dA * h + dB * x_t.unsqueeze(-1)

            # Output computation: y = C * h + D * x
            y = (C_t.unsqueeze(1) * h).sum(-1) + self.D.float() * x_t  # [B, d_inner]
            ys.append(y.to(x.dtype))

        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_official:
            return self.mamba(x)

        B, L, _ = x.shape

        # 1. Dual projection (x_inner and gate z)
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # 2. Causal depthwise 1D conv
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :L]
        x_inner = F.silu(x_inner.transpose(1, 2))

        # 3. Selective projection for dt, B, C
        projected = self.x_proj(x_inner)
        dt_raw, B_ssm, C_ssm = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))

        # 4. Selective scan
        y = self._ssm_scan(x_inner, dt, B_ssm, C_ssm)

        # 5. Output gating with SiLU
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaLayer(nn.Module):
    """Pre-norm Mamba Layer."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, backend: str = "auto"):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, backend=backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class MambaLM(nn.Module):
    """
    Mamba Language Model (~50M parameters).
    
    Architecture:
      Embedding -> N × MambaLayer -> RMSNorm -> Tied LM Head
    """

    ARCH_NAME = "mamba"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 28,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        backend: str = "auto",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, d_conv, expand, backend=backend)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """
        Initialize embedding properly and apply idempotent depth scaling to output projections.
        Preserves Mamba's specialized A_log and dt_proj initialization.
        """
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        with torch.no_grad():
            for layer in self.layers:
                block = layer.mamba
                mixer = block.mamba if block._use_official else block
                if hasattr(mixer, "out_proj") and hasattr(mixer.out_proj, "weight"):
                    nn.init.kaiming_uniform_(mixer.out_proj.weight, a=math.sqrt(5))
                    mixer.out_proj.weight.div_(math.sqrt(self.n_layers))

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


def estimate_mamba_parameters(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    expand: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
) -> int:
    """Analytical parameter count for Mamba with tied embeddings and conv bias."""
    d_inner = d_model * expand
    dt_rank = math.ceil(d_model / 16)

    in_proj = d_model * (2 * d_inner)
    conv1d = d_inner * d_conv + d_inner       # weight + bias
    x_proj = d_inner * (dt_rank + 2 * d_state)
    dt_proj = dt_rank * d_inner + d_inner     # weight + bias
    A_log = d_inner * d_state
    D = d_inner
    out_proj = d_inner * d_model
    layer_norm = d_model

    per_layer = in_proj + conv1d + x_proj + dt_proj + A_log + D + out_proj + layer_norm

    tied_embedding = vocab_size * d_model
    final_norm = d_model

    return tied_embedding + n_layers * per_layer + final_norm


def choose_near_target_mamba_config(vocab_size: int, target_params: int = 50_000_000) -> Dict[str, Any]:
    """Search viable Mamba configurations to match target_params."""
    candidates = []
    for d_model in (384, 448, 512, 576):
        for n_layers in range(16, 36, 2):
            params = estimate_mamba_parameters(vocab_size, d_model, n_layers)
            diff = abs(params - target_params)
            candidates.append((diff, params, {"d_model": d_model, "n_layers": n_layers}))
    return min(candidates, key=lambda x: x[0])[2]


def create_mamba(vocab_size: int = 3919, target_params: int = 50_000_000, backend: str = "auto") -> MambaLM:
    """Create a Mamba LM targeting ~50M parameters."""
    cfg = choose_near_target_mamba_config(vocab_size, target_params)
    return MambaLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        d_state=16,
        d_conv=4,
        expand=2,
        backend=backend,
    )

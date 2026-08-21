"""Model registry for architecture comparison."""

from .transformer_lm import TransformerLM, create_transformer
from .hrm_lm import HRMLM, create_hrm
from .mamba_lm import MambaLM, create_mamba
from .hybrid_lm import HybridMambaTransformerLM, create_hybrid

MODEL_REGISTRY = {
    "transformer": create_transformer,
    "hrm": create_hrm,
    "mamba": create_mamba,
    "hybrid": create_hybrid,
}

def create_model(arch: str, vocab_size: int, target_params: int = 50_000_000):
    """Create a model by architecture name."""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch](vocab_size, target_params)

__all__ = [
    "TransformerLM", "HRMLM", "MambaLM", "HybridMambaTransformerLM",
    "create_model", "MODEL_REGISTRY",
]

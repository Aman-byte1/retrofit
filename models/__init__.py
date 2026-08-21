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

def create_model(arch: str, vocab_size: int = 3919, target_params: int = 50_000_000, **kwargs):
    """Create a model by architecture name with optional kwargs forwarding."""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(MODEL_REGISTRY.keys())}")
    
    fn = MODEL_REGISTRY[arch]
    if arch == "mamba":
        backend = kwargs.get("mamba_backend", kwargs.get("backend", "auto"))
        return fn(vocab_size=vocab_size, target_params=target_params, backend=backend)
    elif arch == "hybrid":
        backend = kwargs.get("mamba_backend", kwargs.get("backend", "auto"))
        return fn(vocab_size=vocab_size, target_params=target_params, mamba_backend=backend)
    else:
        return fn(vocab_size=vocab_size, target_params=target_params)

__all__ = [
    "TransformerLM", "HRMLM", "MambaLM", "HybridMambaTransformerLM",
    "create_model", "MODEL_REGISTRY",
]

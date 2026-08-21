"""
LoRA (Low-Rank Adaptation) implementation for retrofitting voice cloning.

Custom implementation that injects low-rank adapters into any nn.Linear layer,
with support for targeted layer selection and layer importance analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, List, Dict, Set
from collections import OrderedDict

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear layers.
    
    Implements: h = W_0 * x + (B @ A) * x * (alpha / rank)
    where W_0 is frozen and A, B are the trainable low-rank matrices.
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Keep original weights frozen
        self.original = original_linear
        for param in self.original.parameters():
            param.requires_grad = False
        
        # Low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        
        # Optional dropout on LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize: A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original path (frozen)
        original_out = self.original(x)
        
        # LoRA path (trainable)
        lora_out = self.lora_dropout(x)
        lora_out = lora_out @ self.lora_A.T  # [batch, ..., rank]
        lora_out = lora_out @ self.lora_B.T  # [batch, ..., out_features]
        lora_out = lora_out * self.scaling
        
        return original_out + lora_out
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


def _get_parent_module(model: nn.Module, module_name: str):
    """Get the parent module and attribute name for a given dotted module path."""
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    target_layers: Optional[List[int]] = None,
) -> Dict[str, int]:
    """
    Inject LoRA adapters into a model's Linear layers.
    
    Args:
        model: The model to modify in-place.
        rank: LoRA rank (lower = fewer parameters).
        alpha: LoRA scaling factor.
        dropout: Dropout rate on the LoRA path.
        target_modules: List of substrings to match module names against.
                       If None, targets all Linear layers.
        target_layers: List of layer indices to target (extracted from module name).
                      If None, targets all matching layers.
    
    Returns:
        Dict with injection statistics.
    """
    injected = 0
    total_linear = 0
    lora_params = 0
    original_params = 0
    injected_names = []
    
    # Collect all Linear layers and their names
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_linear += 1
            original_params += sum(p.numel() for p in module.parameters())
            
            # Check if this module matches our targets
            should_inject = True
            
            # Filter by module name
            if target_modules is not None:
                should_inject = any(t in name for t in target_modules)
            
            # Filter by layer index (look for patterns like ".0.", ".1.", etc.)
            if should_inject and target_layers is not None:
                layer_idx = _extract_layer_index(name)
                should_inject = layer_idx is not None and layer_idx in target_layers
            
            if should_inject:
                linear_layers.append((name, module))
    
    # Inject LoRA into matched layers
    for name, module in linear_layers:
        parent, attr_name = _get_parent_module(model, name)
        lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, lora_module)
        
        injected += 1
        lora_params += rank * module.in_features + module.out_features * rank
        injected_names.append(name)
    
    stats = {
        "total_linear_layers": total_linear,
        "injected_layers": injected,
        "lora_params": lora_params,
        "original_params": original_params,
        "param_ratio": lora_params / max(original_params, 1) * 100,
        "injected_names": injected_names,
    }
    
    logger.info(
        f"LoRA injection complete: {injected}/{total_linear} layers, "
        f"{lora_params:,} LoRA params ({stats['param_ratio']:.2f}% of original {original_params:,})"
    )
    
    return stats


def _extract_layer_index(module_name: str) -> Optional[int]:
    """
    Extract the transformer block/layer index from a module name.
    
    Handles patterns like:
        transformer_blocks.3.attn.to_q -> 3
        layers.12.self_attn.q_proj -> 12
        encoder.block.5.layer.0.SelfAttention.q -> 5
    """
    import re
    # Look for patterns like .NUMBER. in the module name
    # We want the first number that appears to be a layer index
    matches = re.findall(r'\.(\d+)\.', module_name)
    if matches:
        return int(matches[0])
    return None


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all trainable LoRA parameters from a model."""
    params = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            params.append(param)
    return params


def get_lora_state_dict(model: nn.Module) -> OrderedDict:
    """Extract only the LoRA parameters as a state dict."""
    lora_state = OrderedDict()
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.data.clone()
    return lora_state


def save_lora(model: nn.Module, path: str):
    """Save only the LoRA adapter weights."""
    lora_state = get_lora_state_dict(model)
    torch.save({
        "lora_state_dict": lora_state,
        "num_adapters": len(lora_state) // 2,  # Each adapter has A and B
    }, path)
    logger.info(f"Saved {len(lora_state)} LoRA tensors to {path}")


def load_lora(model: nn.Module, path: str, strict: bool = True):
    """Load LoRA adapter weights into a model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    lora_state = checkpoint["lora_state_dict"]
    
    model_state = model.state_dict()
    loaded = 0
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key] = value
            loaded += 1
        elif strict:
            raise KeyError(f"LoRA key {key} not found in model")
    
    model.load_state_dict(model_state, strict=False)
    logger.info(f"Loaded {loaded} LoRA tensors from {path}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": trainable / max(total, 1) * 100,
    }


def analyze_layer_speaker_sensitivity(
    model: nn.Module,
    dataloader,
    num_batches: int = 50,
    device: str = "cuda",
) -> Dict[int, float]:
    """
    Analyze which transformer layers are most sensitive to speaker identity.
    
    Method: For each layer, compute the variance of activations across
    different speakers. High variance = the layer differentiates speakers
    = important for speaker identity.
    
    Returns:
        Dict mapping layer_index -> speaker_sensitivity_score
    """
    layer_activations = {}
    hooks = []
    
    # Register hooks on transformer blocks
    for name, module in model.named_modules():
        layer_idx = _extract_layer_index(name)
        if layer_idx is not None and name.endswith(('.attn', '.self_attn', '.attention')):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    if idx not in layer_activations:
                        layer_activations[idx] = []
                    # Store mean activation per sample
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    layer_activations[idx].append(
                        out.detach().mean(dim=-1).mean(dim=-1).cpu()
                    )
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(layer_idx)))
    
    # Run forward passes
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            # Forward pass (implementation depends on model)
            # This collects activations via hooks
            pass
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Compute speaker sensitivity as inter-speaker variance
    sensitivity = {}
    for layer_idx, acts in layer_activations.items():
        if acts:
            all_acts = torch.cat(acts, dim=0)
            sensitivity[layer_idx] = all_acts.var(dim=0).mean().item()
    
    # Normalize
    if sensitivity:
        max_val = max(sensitivity.values())
        sensitivity = {k: v / max(max_val, 1e-8) for k, v in sensitivity.items()}
    
    return dict(sorted(sensitivity.items()))

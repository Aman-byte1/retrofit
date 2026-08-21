"""
Unified training script for all 4 architectures.

Records EVERYTHING:
- Loss per step, per epoch
- Validation perplexity
- Learning rate schedule
- Gradient norms
- GPU memory usage per step
- Tokens/second throughput
- Wall-clock time per epoch
- Parameter count breakdown per layer
- Weight statistics (mean, std, min, max per layer)
- Activation statistics
- Training throughput (samples/sec, tokens/sec)
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """Detailed parameter count per named module."""
    breakdown = {}
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            breakdown[name] = params
    return breakdown


def get_weight_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Get weight statistics for each parameter."""
    stats = {}
    for name, param in model.named_parameters():
        with torch.no_grad():
            p = param.float()
            stats[name] = {
                "mean": p.mean().item(),
                "std": p.std().item(),
                "min": p.min().item(),
                "max": p.max().item(),
                "norm": p.norm().item(),
                "numel": p.numel(),
            }
    return stats


def get_gpu_stats() -> Dict[str, Any]:
    """Get GPU memory and utilization stats."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated(0) / 1e6,
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved(0) / 1e6,
        "gpu_max_memory_allocated_mb": torch.cuda.max_memory_allocated(0) / 1e6,
        "gpu_max_memory_reserved_mb": torch.cuda.max_memory_reserved(0) / 1e6,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    log_interval: int = 10,
    scaler=None,
) -> Dict[str, Any]:
    """Train for one epoch, recording all metrics."""
    model.train()
    
    total_loss = 0.0
    total_tokens = 0
    step_logs = []
    
    epoch_start = time.time()
    
    total_steps = getattr(model, "_total_training_steps", len(dataloader))
    
    for step, (x, y) in enumerate(dataloader):
        step_start = time.time()
        global_step = epoch * len(dataloader) + step
        
        # Calculate TBPTT horizon for HRM models if supported
        kwargs = {}
        if hasattr(model, "backward_horizon"):
            kwargs["bp_steps"] = model.backward_horizon(global_step, total_steps)
        
        x = x.to(device)
        y = y.to(device)
        batch_tokens = x.numel()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, loss = model(x, targets=y, **kwargs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(x, targets=y, **kwargs)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        step_time = time.time() - step_start
        tokens_per_sec = batch_tokens / step_time
        
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        
        # Record step metrics
        step_log = {
            "epoch": epoch,
            "step": step,
            "global_step": epoch * len(dataloader) + step,
            "loss": loss.item(),
            "perplexity": math.exp(min(loss.item(), 20)),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": optimizer.param_groups[0]["lr"],
            "tokens_per_sec": tokens_per_sec,
            "batch_tokens": batch_tokens,
            "step_time_sec": step_time,
            "gpu_memory_mb": torch.cuda.memory_allocated(0) / 1e6 if torch.cuda.is_available() else 0,
            "gpu_max_memory_mb": torch.cuda.max_memory_allocated(0) / 1e6 if torch.cuda.is_available() else 0,
        }
        step_logs.append(step_log)
        
        if step % log_interval == 0:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            logger.info(
                f"Epoch {epoch} | Step {step}/{len(dataloader)} | "
                f"Loss={loss.item():.4f} | PPL={step_log['perplexity']:.1f} | "
                f"LR={step_log['lr']:.2e} | GradNorm={step_log['grad_norm']:.3f} | "
                f"Tok/s={tokens_per_sec:.0f} | GPU={step_log['gpu_memory_mb']:.0f}MB"
            )
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    epoch_stats = {
        "epoch": epoch,
        "avg_loss": avg_loss,
        "avg_perplexity": math.exp(min(avg_loss, 20)),
        "total_tokens": total_tokens,
        "epoch_time_sec": epoch_time,
        "avg_tokens_per_sec": total_tokens / epoch_time,
        "num_steps": len(dataloader),
    }
    
    return epoch_stats, step_logs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    eval_start = time.time()
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        batch_tokens = x.numel()
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
        
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
    
    eval_time = time.time() - eval_start
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    return {
        "val_loss": avg_loss,
        "val_perplexity": math.exp(min(avg_loss, 20)),
        "val_tokens": total_tokens,
        "eval_time_sec": eval_time,
    }


def train_model(
    arch: str,
    vocab_size: int,
    train_path: str,
    val_path: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    seq_len: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
    log_interval: int = 10,
):
    """Full training pipeline for a single architecture."""
    from data.dataset import TokenizedLMDataset
    from models import create_model
    
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"TRAINING: {arch.upper()}")
    logger.info("=" * 70)
    
    # Create model
    model = create_model(arch, vocab_size)
    total_params = model.count_params()
    trainable_params = model.count_trainable_params()
    
    logger.info(f"Architecture: {arch}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Device: {device}")
    
    model = model.to(device)
    
    # Dataset
    train_ds = TokenizedLMDataset(train_path, seq_len=seq_len)
    val_ds = TokenizedLMDataset(val_path, seq_len=seq_len)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    logger.info(f"Train: {len(train_ds)} sequences ({len(train_ds) * seq_len:,} tokens)")
    logger.info(f"Val:   {len(val_ds)} sequences ({len(val_ds) * seq_len:,} tokens)")
    
    # Optimizer with parameter grouping (weight decay only on 2D+ tensors, 0.0 on norms, biases, and _no_weight_decay)
    decay_params = []
    nodecay_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if getattr(p, "_no_weight_decay", False) or p.dim() < 2:
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=(0.9, 0.95),
    )
    
    # Cosine schedule with warmup
    total_steps = epochs * len(train_dl)
    model._total_training_steps = total_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    # Record EVERYTHING
    run_metadata = {
        "architecture": arch,
        "arch_class": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "vocab_size": vocab_size,
        "d_model": model.d_model,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "grad_clip": grad_clip,
        "total_steps": total_steps,
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
        "train_tokens": len(train_ds) * seq_len,
        "val_tokens": len(val_ds) * seq_len,
        "device": str(device),
        "dtype": "bfloat16",
        "start_time": datetime.now().isoformat(),
        "param_breakdown": count_parameters_by_layer(model),
    }
    
    if torch.cuda.is_available():
        run_metadata["gpu_name"] = torch.cuda.get_device_name(0)
        run_metadata["gpu_total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Get layer types for hybrid
    if hasattr(model, "get_layer_breakdown"):
        run_metadata["layer_breakdown"] = model.get_layer_breakdown()
    if hasattr(model, "layer_types"):
        run_metadata["layer_types"] = model.layer_types
    
    # Training loop
    all_step_logs = []
    epoch_logs = []
    best_val_ppl = float("inf")
    
    # Initial weight stats
    initial_weight_stats = get_weight_stats(model)
    
    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    train_start = time.time()
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*50}")
        
        epoch_stats, step_logs = train_one_epoch(
            model, train_dl, optimizer, scheduler, device,
            epoch=epoch,
            grad_clip=grad_clip,
            log_interval=log_interval,
            scaler=scaler,
        )
        
        # Validation
        val_stats = evaluate(model, val_dl, device)
        
        epoch_log = {**epoch_stats, **val_stats}
        epoch_logs.append(epoch_log)
        all_step_logs.extend(step_logs)
        
        logger.info(
            f"Epoch {epoch+1} Summary | "
            f"Train Loss={epoch_stats['avg_loss']:.4f} | "
            f"Train PPL={epoch_stats['avg_perplexity']:.1f} | "
            f"Val Loss={val_stats['val_loss']:.4f} | "
            f"Val PPL={val_stats['val_perplexity']:.1f} | "
            f"Time={epoch_stats['epoch_time_sec']:.1f}s | "
            f"Tok/s={epoch_stats['avg_tokens_per_sec']:.0f}"
        )
        
        # Save best model
        if val_stats["val_perplexity"] < best_val_ppl:
            best_val_ppl = val_stats["val_perplexity"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info(f"  ★ New best val PPL: {best_val_ppl:.2f}")
    
    total_train_time = time.time() - train_start
    
    # Final weight stats
    final_weight_stats = get_weight_stats(model)
    
    # GPU stats
    gpu_stats = get_gpu_stats()
    
    # Compile all metrics
    full_results = {
        "metadata": run_metadata,
        "epoch_logs": epoch_logs,
        "best_val_perplexity": best_val_ppl,
        "total_train_time_sec": total_train_time,
        "total_train_time_min": total_train_time / 60,
        "final_gpu_stats": gpu_stats,
        "initial_weight_stats": initial_weight_stats,
        "final_weight_stats": final_weight_stats,
        "end_time": datetime.now().isoformat(),
    }
    
    # Save everything
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    
    with open(os.path.join(output_dir, "step_logs.json"), "w") as f:
        json.dump(all_step_logs, f, indent=2)
    
    # Save last checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "best_val_ppl": best_val_ppl,
    }, os.path.join(output_dir, "last_checkpoint.pt"))
    
    logger.info(f"\nTraining complete for {arch}!")
    logger.info(f"  Best Val PPL: {best_val_ppl:.2f}")
    logger.info(f"  Total Time: {total_train_time/60:.1f} min")
    logger.info(f"  Results saved to {output_dir}")
    
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--arch", type=str, required=True,
                       choices=["transformer", "hrm", "mamba", "hybrid"],
                       help="Architecture to train")
    parser.add_argument("--vocab-size", type=int, default=3919,
                       help="Vocabulary size (default: 3919 for RL Amharic tokenizer)")
    parser.add_argument("--train-data", type=str, default="data/tokenized/train_tokens.npy")
    parser.add_argument("--val-data", type=str, default="data/tokenized/val_tokens.npy")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    
    args = parser.parse_args()
    
    out_dir = os.path.join(args.output_dir, args.arch)
    
    train_model(
        arch=args.arch,
        vocab_size=args.vocab_size,
        train_path=args.train_data,
        val_path=args.val_data,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()

"""
Unified, rigorous pretraining & benchmarking harness for 50M-class architectures.

Features:
  • Decoupled bfloat16 AMP with proper device-aware autocast
  • True CUDA synchronization for microsecond-accurate token/sec throughput
  • Per-step peak GPU memory tracking (reset_peak_memory_stats per step)
  • Gradient accumulation support (--grad-accum N)
  • Streaming JSONL step logging (zero unbounded memory growth)
  • Fair, mathematically comparable validation perplexity (pure base LM cross-entropy)
  • Separate logging for base cross-entropy, MTP auxiliary loss, and training objective
  • Support for warmup TBPTT horizon scheduling in HRM-Text models
  • Complete, reproducible checkpointing (model, optimizer, scheduler, scaler, RNG states)
  • Model configuration inspection & target parameter verification
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
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_model
from data.dataset import MemmapDataset, InfiniteDataLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


IGNORE_INDEX = -100


def seed_everything(seed: int = 42):
    """Set deterministic random seeds across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_metrics(device: torch.device) -> Dict[str, Any]:
    """Capture instantaneous GPU memory status."""
    if device.type != "cuda":
        return {}
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / 1e6, 2),
        "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(device) / 1e6, 2),
        "gpu_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 2),
        "gpu_max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1e6, 2),
    }


def get_weight_statistics(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Capture weight distribution statistics across model layers."""
    stats = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            data = p.detach().float()
            stats[name] = {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "norm": float(data.norm()),
                "numel": p.numel(),
            }
    return stats


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    grad_accum: int = 1,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: Optional[torch.amp.GradScaler] = None,
    log_interval: int = 10,
    step_log_file: Optional[Path] = None,
    total_steps: int = 1000,
    hrm_mode: str = "causal",
) -> Dict[str, Any]:
    """Train for one epoch with rigorous timing, step peak VRAM tracking, and streaming JSONL logging."""
    model.train()

    total_base_loss_sum = 0.0
    total_objective_loss_sum = 0.0
    total_valid_tokens = 0
    total_processed_tokens = 0

    epoch_start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(dataloader):
        global_step = epoch * len(dataloader) + step

        # Reset per-step peak memory tracking
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        step_start_time = time.perf_counter()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_tokens = x.numel()

        # Dynamic arguments for specialized models (e.g. HRM TBPTT warmup horizon)
        kwargs = {}
        if hasattr(model, "backward_horizon"):
            kwargs["bp_steps"] = model.backward_horizon(global_step, total_steps)
        if hasattr(model, "ARCH_NAME") and model.ARCH_NAME == "hrm":
            kwargs["attention_mode"] = hrm_mode

        # Forward pass under device-aware autocast
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits, loss = model(x, targets=y, **kwargs)

        # Scale loss for gradient accumulation
        loss_scaled = loss / grad_accum

        # Backward pass
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # Optimizer step upon accumulation boundary
        is_accum_boundary = ((step + 1) % grad_accum == 0) or ((step + 1) == len(dataloader))
        grad_norm = None

        if is_accum_boundary:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        # Precise throughput timing with CUDA synchronization
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_elapsed = time.perf_counter() - step_start_time
        tokens_per_sec = batch_tokens / max(1e-6, step_elapsed)
        samples_per_sec = x.shape[0] / max(1e-6, step_elapsed)

        # Retrieve fine-grained loss components
        base_loss_val = getattr(model, "last_base_loss", None)
        if base_loss_val is None:
            base_loss_val = loss.item()
        mtp_loss_val = getattr(model, "last_mtp_loss", 0.0) or 0.0

        valid_tokens_count = int((y != IGNORE_INDEX).sum().item())
        total_base_loss_sum += base_loss_val * valid_tokens_count
        total_objective_loss_sum += loss.item() * valid_tokens_count
        total_valid_tokens += valid_tokens_count
        total_processed_tokens += batch_tokens

        # Record step diagnostics
        step_log = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "objective_loss": round(loss.item(), 5),
            "base_loss": round(base_loss_val, 5),
            "mtp_loss": round(mtp_loss_val, 5),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": round(float(grad_norm), 4) if grad_norm is not None else None,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "samples_per_sec": round(samples_per_sec, 2),
            "step_time_ms": round(step_elapsed * 1000, 2),
            "bp_steps": kwargs.get("bp_steps", None),
        }

        if device.type == "cuda":
            step_log["gpu_peak_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 2)

        # Stream directly to JSONL file to prevent memory accumulation
        if step_log_file is not None:
            with open(step_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_log) + "\n")

        if step % log_interval == 0:
            gpu_str = f" | Peak VRAM: {step_log.get('gpu_peak_memory_mb', 0):.1f}MB" if device.type == "cuda" else ""
            mtp_str = f" | MTP: {mtp_loss_val:.4f}" if mtp_loss_val > 0 else ""
            bp_str = f" | BP: {kwargs['bp_steps']}" if "bp_steps" in kwargs else ""
            logger.info(
                f"Epoch {epoch:2d} | Step {step:4d}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} (Base: {base_loss_val:.4f}{mtp_str}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Speed: {tokens_per_sec:,.0f} tok/s{bp_str}{gpu_str}"
            )

    epoch_elapsed = time.perf_counter() - epoch_start_time
    avg_base_loss = total_base_loss_sum / max(1, total_valid_tokens)
    avg_objective_loss = total_objective_loss_sum / max(1, total_valid_tokens)

    return {
        "epoch": epoch,
        "avg_base_loss": avg_base_loss,
        "avg_objective_loss": avg_objective_loss,
        "total_valid_tokens": total_valid_tokens,
        "total_processed_tokens": total_processed_tokens,
        "epoch_time_seconds": epoch_elapsed,
        "avg_tokens_per_sec": total_processed_tokens / max(1e-6, epoch_elapsed),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    hrm_mode: str = "causal",
) -> Dict[str, float]:
    """
    Rigorously evaluate language model perplexity.
    Computes cross-entropy strictly on base next-token prediction logits.
    """
    model.eval()

    total_ce_sum = 0.0
    total_valid_tokens = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        kwargs = {}
        if hasattr(model, "ARCH_NAME") and model.ARCH_NAME == "hrm":
            kwargs["attention_mode"] = hrm_mode

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits, _ = model(x, **kwargs)

        flat_logits = logits.float().reshape(-1, logits.size(-1))
        flat_targets = y.reshape(-1)

        ce_sum = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        valid = (flat_targets != IGNORE_INDEX).sum().item()

        total_ce_sum += float(ce_sum)
        total_valid_tokens += int(valid)

    avg_loss = total_ce_sum / max(1, total_valid_tokens)
    perplexity = math.exp(min(20.0, avg_loss))  # clamp to avoid overflow

    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        "val_valid_tokens": total_valid_tokens,
    }


def train_model(
    arch: str,
    data_dir: str,
    output_dir: str,
    vocab_size: int = 3919,
    target_params: int = 50_000_000,
    seq_len: int = 512,
    batch_size: int = 32,
    grad_accum: int = 1,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    weight_decay: float = 0.1,
    warmup_steps: int = 200,
    epochs: int = 3,
    grad_clip: float = 1.0,
    log_interval: int = 10,
    seed: int = 42,
    mamba_backend: str = "auto",
    hrm_mode: str = "causal",
) -> Dict[str, Any]:
    """Complete, end-to-end reproducible training workflow."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(output_dir) / f"{arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    step_log_file = run_dir / "step_logs.jsonl"

    logger.info("=" * 80)
    logger.info(f"STARTING TRAINING: {arch.upper()} on {device} (Device Name: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"Target Parameters: {target_params:,} | Vocab Size: {vocab_size:,} | Batch: {batch_size} (accum={grad_accum})")
    logger.info("=" * 80)

    # 1. Instantiate Model Architecture
    model_kwargs = {"target_params": target_params}
    if arch in {"mamba", "hybrid"}:
        model_kwargs["mamba_backend"] = mamba_backend

    model = create_model(arch, vocab_size=vocab_size, **model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_error = (total_params - target_params) / target_params

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total Parameters:     {total_params:,} ({total_params/1e6:.2f}M) [Target Error: {param_error:+.2%}]")
    logger.info(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 2. Datasets & Loaders
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_bin):
        logger.warning(f"{train_bin} not found. Creating synthetic training tokens for verification...")
        synthetic_tokens = np.random.randint(0, vocab_size, size=(200_000,), dtype=np.uint16)
        synthetic_tokens.tofile(train_bin)
        synthetic_tokens[:20_000].tofile(val_bin)

    train_ds = MemmapDataset(train_bin, seq_len=seq_len)
    val_ds = MemmapDataset(val_bin, seq_len=seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Train Dataset: {len(train_ds):,} sequences ({len(train_ds) * seq_len:,} tokens)")
    logger.info(f"Val Dataset:   {len(val_ds):,} sequences ({len(val_ds) * seq_len:,} tokens)")

    # 3. Optimizer with parameter-grouping respecting _no_weight_decay
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
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))

    total_training_steps = (epochs * len(train_dl)) // grad_accum
    model._total_training_steps = total_training_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / lr) + (1.0 - min_lr / lr) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed Precision Setup (independent of GradScaler)
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.bfloat16
    scaler = None  # bfloat16 does not require gradient scaling

    epoch_logs = []
    val_history = []
    initial_weights = get_weight_statistics(model)

    # Initial Zero-Shot Validation Check
    initial_eval = evaluate(model, val_dl, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype, hrm_mode=hrm_mode)
    logger.info(f"Initial Zero-Shot Validation Loss: {initial_eval['val_loss']:.4f} | Perplexity: {initial_eval['perplexity']:.2f}")

    start_wall_clock = time.time()

    # 4. Training Loop
    for epoch in range(epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_result = train_one_epoch(
            model=model,
            dataloader=train_dl,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            grad_clip=grad_clip,
            grad_accum=grad_accum,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log_interval=log_interval,
            step_log_file=step_log_file,
            total_steps=total_training_steps,
            hrm_mode=hrm_mode,
        )

        eval_result = evaluate(
            model=model,
            dataloader=val_dl,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            hrm_mode=hrm_mode,
        )

        epoch_result.update(eval_result)
        epoch_logs.append(epoch_result)
        val_history.append(eval_result["perplexity"])

        logger.info(
            f"Epoch {epoch + 1} Complete | "
            f"Train Base Loss: {epoch_result['avg_base_loss']:.4f} | "
            f"Val Loss: {eval_result['val_loss']:.4f} | "
            f"Val PPL: {eval_result['perplexity']:.2f} | "
            f"Epoch Speed: {epoch_result['avg_tokens_per_sec']:,.0f} tok/s"
        )

        # Comprehensive, fully-resumable checkpoint
        checkpoint = {
            "epoch": epoch,
            "global_step": (epoch + 1) * len(train_dl),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "arch": arch,
            "vocab_size": vocab_size,
            "val_loss": eval_result["val_loss"],
            "perplexity": eval_result["perplexity"],
            "model_config": model_kwargs,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, run_dir / f"checkpoint_epoch_{epoch + 1}.pt")

    total_wall_clock = time.time() - start_wall_clock
    final_weights = get_weight_statistics(model)

    summary = {
        "arch": arch,
        "vocab_size": vocab_size,
        "target_params": target_params,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_error": round(param_error, 4),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "lr": lr,
        "min_lr": min_lr,
        "epochs": epochs,
        "total_steps": total_training_steps,
        "total_wall_clock_seconds": round(total_wall_clock, 2),
        "initial_val_ppl": round(initial_eval["perplexity"], 2),
        "best_val_ppl": round(min(val_history), 2),
        "final_val_ppl": round(val_history[-1], 2),
        "epoch_logs": epoch_logs,
        "gpu_info": get_gpu_metrics(device),
        "completed_at": datetime.now().isoformat(),
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING COMPLETE: {arch.upper()}")
    logger.info(f"Best Val PPL: {min(val_history):.2f} | Total Time: {total_wall_clock/60:.1f} min | Run Dir: {run_dir}")
    logger.info("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train 50M-Class Amharic Language Model Architectures")
    parser.add_argument("--arch", type=str, required=True, choices=["transformer", "hrm", "mamba", "hybrid"], help="Architecture to train")
    parser.add_argument("--data-dir", type=str, default="data/tokenized", help="Path to tokenized binary datasets")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save checkpoints and metrics")
    parser.add_argument("--vocab-size", type=int, default=3919, help="Vocabulary size")
    parser.add_argument("--target-params", type=int, default=50_000_000, help="Target parameter count (~50M)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mamba-backend", type=str, default="auto", choices=["auto", "official", "torch"], help="Mamba implementation backend")
    parser.add_argument("--hrm-mode", type=str, default="causal", choices=["causal", "prefix"], help="HRM attention mode")

    args = parser.parse_args()

    train_model(
        arch=args.arch,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        target_params=args.target_params,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        seed=args.seed,
        mamba_backend=args.mamba_backend,
        hrm_mode=args.hrm_mode,
    )


if __name__ == "__main__":
    main()

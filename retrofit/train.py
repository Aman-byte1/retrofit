"""
Training for the Retrofit Voice Cloner.

Trains ONLY the adapter layers using contrastive speaker learning:
- Same-speaker pairs → similar conditioning vectors
- Different-speaker pairs → dissimilar conditioning vectors

This is fast because we DON'T run the TTS model during training.
The adapter learns to project speaker embeddings into a good
conditioning space using only the frozen speaker encoder.

Usage:
    # Train on IWSLT data (French)
    python -m retrofit.train --language fr --epochs 100

    # Train on Common Voice
    python -m retrofit.train --data-source mozilla-foundation/common_voice_17_0 --language fr

    # Quick test run
    python -m retrofit.train --language fr --epochs 5 --max-train-samples 100
"""

import argparse
import logging
import sys
import time
import yaml
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("retrofit.train")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_speaker_pairs(dataset, speaker_encoder, device="cuda"):
    """
    Pre-compute speaker embeddings for all training samples.
    
    Returns a list of (embedding, speaker_id) tuples for contrastive training.
    """
    logger.info("Pre-computing speaker embeddings for training data...")
    
    embeddings = []
    
    for i in tqdm(range(len(dataset)), desc="Extracting speaker embeddings"):
        sample = dataset[i]
        audio = sample["audio"]
        
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        emb = speaker_encoder(audio, sr=24000)  # Returns [1, 192]
        embeddings.append(emb.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)  # [N, 192]
    
    # Auto-discover speaker groups by clustering embeddings
    # (since dataset metadata may not have reliable speaker IDs)
    speaker_labels = _cluster_speakers(embeddings, threshold=0.75)
    n_speakers = speaker_labels.max().item() + 1
    
    logger.info(f"Extracted {len(embeddings)} embeddings, discovered {n_speakers} speaker clusters")
    
    return embeddings, speaker_labels, n_speakers


def _cluster_speakers(
    embeddings: torch.Tensor,
    threshold: float = 0.75,
) -> torch.Tensor:
    """
    Greedy clustering of speaker embeddings by cosine similarity.
    
    Two embeddings are assigned the same speaker label if their
    cosine similarity exceeds the threshold.
    
    Args:
        embeddings: [N, dim] speaker embeddings
        threshold: cosine similarity threshold for same-speaker
        
    Returns:
        [N] integer speaker labels
    """
    import torch.nn.functional as F
    
    N = embeddings.shape[0]
    normed = F.normalize(embeddings, dim=-1)
    
    labels = torch.full((N,), -1, dtype=torch.long)
    cluster_centroids = []
    current_label = 0
    
    for i in range(N):
        if len(cluster_centroids) > 0:
            # Compare to existing cluster centroids
            centroids = torch.stack(cluster_centroids)  # [K, dim]
            sims = torch.matmul(normed[i:i+1], centroids.T).squeeze(0)  # [K]
            best_sim, best_idx = sims.max(dim=0)
            
            if best_sim.item() >= threshold:
                labels[i] = best_idx.item()
                # Update centroid (running average)
                cluster_centroids[best_idx.item()] = F.normalize(
                    cluster_centroids[best_idx.item()] + normed[i], dim=-1
                )
                continue
        
        # New cluster
        labels[i] = current_label
        cluster_centroids.append(normed[i].clone())
        current_label += 1
    
    return labels


def train_adapter(
    adapter: nn.Module,
    embeddings: torch.Tensor,
    speaker_labels: torch.Tensor,
    config: dict,
    output_dir: Path,
):
    """
    Train the adapter using contrastive speaker loss.
    
    Fast training — no TTS model in the loop.
    Only the adapter MLP + FiLM layers are optimized.
    """
    from .adapters import ContrastiveLoss, SpeakerConsistencyLoss
    
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 64)
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    device = config.get("model", {}).get("device", "cuda")
    
    # Move data to device
    embeddings = embeddings.to(device)
    speaker_labels = speaker_labels.to(device)
    adapter = adapter.to(device)
    adapter.train()
    
    n_samples = len(embeddings)
    
    # Losses
    contrastive_loss = ContrastiveLoss(temperature=0.07)
    consistency_loss = SpeakerConsistencyLoss(margin=0.2)
    
    # Optimizer (only adapter params)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    
    logger.info(f"Training adapter for {epochs} epochs...")
    logger.info(f"  Samples: {n_samples}, Batch size: {batch_size}")
    logger.info(f"  Adapter params: {sum(p.numel() for p in adapter.parameters()):,}")
    logger.info(f"  Learning rate: {lr}")
    
    best_loss = float("inf")
    history = []
    
    for epoch in range(1, epochs + 1):
        # Shuffle
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]
            
            batch_emb = embeddings[idx]       # [B, 192]
            batch_spk = speaker_labels[idx]    # [B]
            
            # Forward through adapter
            conditioning = adapter(batch_emb)  # [B, output_dim]
            
            # Contrastive loss: same speaker → similar, different → dissimilar
            loss_contra = contrastive_loss(conditioning, batch_spk)
            
            # Consistency loss: conditioning should be speaker-discriminative
            # Create random pairs within the batch
            if len(idx) > 1:
                idx_a = torch.randperm(len(idx))
                idx_b = torch.randperm(len(idx))
                cond_a = conditioning[idx_a]
                cond_b = conditioning[idx_b]
                same = batch_spk[idx_a] == batch_spk[idx_b]
                loss_consist = consistency_loss(cond_a, cond_b, same)
            else:
                loss_consist = torch.tensor(0.0, device=device)
            
            # Total loss
            loss = loss_contra + 0.5 * loss_consist
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)
        
        # Logging
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "adapter_state_dict": adapter.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, output_dir / "adapter_best.pt")
        
        # Periodic checkpoint
        save_every = train_cfg.get("save_every_n_epochs", 25)
        if epoch % save_every == 0:
            torch.save({
                "adapter_state_dict": adapter.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }, output_dir / f"adapter_epoch_{epoch}.pt")
    
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    
    # Save training history
    import json
    with open(output_dir / "training_history.json", "w") as f:
        json.dump({"loss": history, "best_loss": best_loss}, f)
    
    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Retrofit: Train Voice Cloning Adapter")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--language", default="fr", help="Language for TTS model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-source", default="iwslt")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--adapter-type", default="film", choices=["film", "additive"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-eval", action="store_true")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Apply overrides
    if args.epochs: config["training"]["epochs"] = args.epochs
    if args.lr: config["training"]["learning_rate"] = args.lr
    if args.batch_size: config["training"]["batch_size"] = args.batch_size
    if args.max_train_samples: config["data"]["max_train_samples"] = args.max_train_samples
    config["model"]["language"] = args.language
    config["adapter"] = config.get("adapter", {})
    config["adapter"]["type"] = args.adapter_type
    
    # Setup output
    exp_name = args.experiment_name or f"retrofit_{args.adapter_type}_{args.language}"
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File logging
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("Retrofit: Training Voice Cloning Adapter")
    logger.info("=" * 60)
    logger.info(f"Language: {args.language}")
    logger.info(f"Adapter type: {args.adapter_type}")
    logger.info(f"Output: {output_dir}")
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # ── Step 1: Build the retrofit model ──
    logger.info("\n[1/4] Building retrofit model...")
    from .model import RetrofitVoiceCloner
    
    model_config = {
        **config.get("model", {}),
        "language": args.language,
        "adapter": config.get("adapter", {}),
        "speaker_encoder": config.get("speaker_encoder", {}),
    }
    model = RetrofitVoiceCloner(model_config)
    
    # ── Step 2: Load training data ──
    logger.info("\n[2/4] Loading training data...")
    from .data import MultiSpeakerTrainDataset
    
    train_dataset = MultiSpeakerTrainDataset(
        source=args.data_source,
        language=args.language,
        target_sr=config.get("audio", {}).get("sample_rate", 24000),
        max_samples=config["data"].get("max_train_samples"),
        is_train=True,
    )
    
    # ── Step 3: Pre-compute speaker embeddings ──
    logger.info("\n[3/4] Pre-computing speaker embeddings...")
    embeddings, speaker_labels, unique_speakers = create_speaker_pairs(
        train_dataset,
        model.speaker_encoder,
        device=config["model"]["device"],
    )
    
    # ── Step 4: Train the adapter ──
    logger.info("\n[4/4] Training adapter...")
    start_time = time.time()
    
    best_loss = train_adapter(
        adapter=model.adapter,
        embeddings=embeddings,
        speaker_labels=speaker_labels,
        config=config,
        output_dir=output_dir,
    )
    
    train_time = time.time() - start_time
    logger.info(f"\nTraining time: {train_time/60:.1f} minutes")
    
    # Save final adapter from the model
    model.save_adapter(str(output_dir / "adapter_final.pt"))
    
    # ── Step 5: Post-training evaluation ──
    if not args.no_eval:
        logger.info("\nRunning post-training evaluation...")
        model.load_adapter(str(output_dir / "adapter_best.pt"))
        
        from .evaluate import run_evaluation
        results = run_evaluation(
            model=model,
            config=config,
            output_dir=output_dir,
            method_name=f"retrofit_{args.adapter_type}",
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        for lang, stats in results.items():
            logger.info(
                f"  {lang}: CER={stats['avg_cer']:.3f} | "
                f"SpkSim={stats['avg_speaker_sim']:.3f} | "
                f"Combined={stats['avg_combined']:.3f}"
            )
        
        # Compare against baseline
        logger.info(f"\n  Training time: {train_time/60:.1f} min")
        logger.info(f"  Adapter size: {sum(p.numel() for p in model.adapter.parameters()):,} params")
    
    logger.info("\nDone! ✓")


if __name__ == "__main__":
    main()

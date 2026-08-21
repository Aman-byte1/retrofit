"""
Training entry point for Retrofit voice cloning experiments.

Usage:
    # Zero-shot baseline (no training, just evaluate)
    python -m retrofit.train --method zero_shot

    # Uniform LoRA fine-tuning
    python -m retrofit.train --method uniform_lora --epochs 50

    # Targeted LoRA (after running layer analysis)
    python -m retrofit.train --method targeted_lora --target-layers 0 1 2 5 8

    # Full fine-tuning baseline (expensive)
    python -m retrofit.train --method full_finetune --epochs 20
"""

import argparse
import logging
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("retrofit.train")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Retrofit: Efficient Voice Cloning Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--method", type=str, default="uniform_lora",
                       choices=["zero_shot", "uniform_lora", "targeted_lora", "full_finetune"],
                       help="Adaptation method")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--lora-rank", type=int, default=None,
                       help="Override LoRA rank")
    parser.add_argument("--target-layers", type=int, nargs="+", default=None,
                       help="Layer indices for targeted LoRA")
    parser.add_argument("--data-source", type=str, default="iwslt",
                       help="Training data source: 'iwslt', HF dataset name, or 'local:/path'")
    parser.add_argument("--language", type=str, default="fr",
                       help="Language for training data")
    parser.add_argument("--max-train-samples", type=int, default=None,
                       help="Max training samples")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override output directory")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name for logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no-eval", action="store_true",
                       help="Skip post-training evaluation")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply CLI overrides
    config["adaptation_method"] = args.method
    
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.lora_rank is not None:
        config["lora"]["rank"] = args.lora_rank
    if args.target_layers is not None:
        config["lora"]["target_layers"] = args.target_layers
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = args.output_dir
    if args.experiment_name is not None:
        config["experiment"]["name"] = args.experiment_name
    if args.max_train_samples is not None:
        config["data"]["max_train_samples"] = args.max_train_samples
    
    # Set output directory based on method
    exp_name = args.experiment_name or f"{args.method}_r{config['lora']['rank']}"
    output_dir = Path(config["experiment"]["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config["experiment"]["output_dir"] = str(output_dir)
    
    # Add file logging
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    # Set seed
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("Retrofit: Efficient Voice Cloning Training")
    logger.info("=" * 60)
    logger.info(f"Method: {args.method}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {config['model']['device']}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # ==========================================
    # 1. Load Model
    # ==========================================
    logger.info("Loading model...")
    
    model_config = {**config["model"], "lora": config["lora"], "adaptation_method": args.method}
    
    from .model import RetrofitModel, FlowMatchingTrainer
    model = RetrofitModel(model_config)
    
    if args.method == "zero_shot":
        logger.info("Zero-shot mode — skipping training, proceeding to evaluation")
    else:
        # ==========================================
        # 2. Load Training Data
        # ==========================================
        logger.info("Loading training data...")
        from .data import create_train_dataloader
        
        train_loader = create_train_dataloader(
            source=args.data_source,
            language=args.language,
            batch_size=config["training"]["batch_size"],
            max_samples=config["data"].get("max_train_samples"),
            target_sr=config["audio"]["sample_rate"],
            num_workers=config["training"]["num_workers"],
            is_train=True,
        )
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        
        # ==========================================
        # 3. Train
        # ==========================================
        trainer = FlowMatchingTrainer(model, config)
        best_loss = trainer.train(train_loader)
        
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    
    # ==========================================
    # 4. Post-Training Evaluation
    # ==========================================
    if not args.no_eval:
        logger.info("Running post-training evaluation...")
        from .evaluate import run_evaluation
        
        results = run_evaluation(
            model=model,
            config=config,
            output_dir=output_dir,
            method_name=args.method,
        )
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        for lang, stats in results.items():
            logger.info(
                f"  {lang}: CER={stats['avg_cer']:.3f} | "
                f"SpeakerSim={stats['avg_speaker_sim']:.3f} | "
                f"Combined={stats['avg_combined']:.3f}"
            )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

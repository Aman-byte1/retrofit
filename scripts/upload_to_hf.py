"""
Hugging Face Hub Uploader for 50M Amharic Language Model Benchmark.

Uploads:
  • Trained model checkpoints (best_model.pt for each architecture)
  • Model definitions, tokenizers, and configs
  • Benchmark plots, JSON metrics, and LaTeX analysis tables
  • Comprehensive Hugging Face Model Card (README.md)
"""

import argparse
import os
import shutil
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_REPO_NAME = "amharic-50m-architecture-benchmark"


def create_model_card(output_path: Path, repo_id: str, results_dir: Path):
    """Generate a clean, professional Hugging Face Model Card."""
    readme_content = f"""---
language:
- am
license: apache-2.0
tags:
- amharic
- language-modeling
- qwen
- hrm
- mamba
- hybrid
- ssm
- transformers
- 50m
datasets:
- wikimedia/wikipedia
pipeline_tag: text-generation
---

# 50M-Class Comparative Architecture Benchmark (Amharic)

This repository contains trained checkpoints, benchmark logs, and evaluation reports for **4 distinct ~50M parameter language model architectures** trained from scratch on the Amharic Wikipedia corpus using an RL-trained subword tokenizer (3,919 tokens).

## 📊 Models & Architectures

| Architecture | Model Class | Parameters | Distinct Features |
|---|---|---|---|
| **Qwen3.5 (Transformer)** | `TransformerLM` | **50.79M** | 3:1 Gated DeltaNet / Gated Attention, Zero-Centered RMSNorm, Partial RoPE (`0.25`), Multi-Token Prediction (MTP) Head |
| **HRM-Text** | `HRMLM` | **49.02M** | Dual-timescale `H2L3` Recurrence, MagicNorm parameterless RMSNorm, PrefixLM masking, Warmup TBPTT ($2 \\rightarrow 5$) |
| **Mamba SSM** | `MambaLM` | **49.47M** | Selective State Space Model, `dt_rank=32`, S4D $A_{{\\text{{log}}}}$, specialized log-uniform $\\Delta$ initialization |
| **Hybrid Mamba-Transformer** | `HybridMambaTransformerLM` | **52.04M** | Interleaved 2:1 Mamba to Qwen3.5 Attention, Unified SwiGLU FFN, Idempotent Depth-Scaling |

## 📁 Repository Structure

```
├── models/                     # PyTorch architecture implementations
│   ├── transformer_lm.py       # Qwen3.5 3:1 DeltaNet + Attention + MTP
│   ├── hrm_lm.py               # HRM-Text H2L3 recurrence
│   ├── mamba_lm.py             # Mamba selective SSM
│   └── hybrid_lm.py            # Hybrid Mamba + Qwen attention
├── tokenizer/                  # Amharic subword tokenizer
│   ├── vocab.txt
│   └── config.json
├── checkpoints/                # Best model weights for each architecture
│   ├── transformer/best_model.pt
│   ├── hrm/best_model.pt
│   ├── mamba/best_model.pt
│   └── hybrid/best_model.pt
└── analysis/                   # Comparative benchmark results & plots
    ├── report.md
    ├── results_table.tex
    ├── loss_curves.png
    ├── throughput_scaling.png
    └── pareto_frontier.png
```

## 🚀 How to Load and Use

```python
import torch
from models import create_model

# Load model architecture
model = create_model("transformer", vocab_size=3919)

# Load checkpoint
checkpoint = torch.load("checkpoints/transformer/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state"], strict=False)
model.eval()

# Generate tokens
tokens = torch.tensor([[2, 45, 128, 902]], dtype=torch.long)
with torch.no_grad():
    logits, _ = model(tokens)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
print("Next token ID:", next_token.item())
```

## 📜 Citation & Attribution

If you use these models or comparative benchmarks in your research, please cite:
- **Qwen3.5**: Alibaba Qwen Team (2025/2026)
- **HRM-Text**: Sapient Intelligence (Wang et al., 2026)
- **Mamba**: Gu & Dao (2023)
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(readme_content)


def upload_to_huggingface(
    token: Optional[str] = None,
    username: str = "amanuelbyte",
    repo_name: str = DEFAULT_REPO_NAME,
    results_dir: str = "results",
    models_dir: str = "models",
    tokenizer_dir: str = "tokenizer",
):
    """Package and upload repository artifacts to Hugging Face Hub."""
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: No Hugging Face token provided via --token or HF_TOKEN environment variable.")
        print("Please set HF_TOKEN or pass --token to upload.")
        return

    api = HfApi(token=hf_token)
    repo_id = f"{username}/{repo_name}"

    print(f"============================================================")
    print(f"  Uploading 50M Amharic Benchmark to Hugging Face Hub")
    print(f"  Target Repository: https://huggingface.co/{repo_id}")
    print(f"============================================================")

    # 1. Create Model Repository on Hugging Face
    print(f"\n[1/4] Creating or verifying repo: {repo_id}...")
    create_repo(
        repo_id=repo_id,
        token=hf_token,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    # 2. Stage export package
    staging_dir = Path("hf_staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/4] Staging files for upload in {staging_dir}...")

    # Copy code & architecture modules
    shutil.copytree(models_dir, staging_dir / "models")
    shutil.copytree(tokenizer_dir, staging_dir / "tokenizer")
    for script in ["train.py", "benchmark.py", "analyze.py"]:
        if os.path.exists(script):
            shutil.copy2(script, staging_dir / script)

    # Copy checkpoints
    checkpoints_dir = staging_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    res_path = Path(results_dir)
    if res_path.exists():
        for arch in ["transformer", "hrm", "mamba", "hybrid"]:
            arch_ckpt_dir = checkpoints_dir / arch
            arch_ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Search in canonical or timestamped folders
            candidates = [
                res_path / arch / "best_model.pt",
                res_path / arch / "checkpoint_epoch_1.pt",
            ]
            # Search timestamped dirs
            for sub in sorted(res_path.glob(f"{arch}_*")):
                candidates.extend([
                    sub / "best_model.pt",
                    sub / "checkpoint_epoch_1.pt",
                    sub / "summary.json",
                ])

            for cand in candidates:
                if cand.exists():
                    shutil.copy2(cand, arch_ckpt_dir / cand.name)
                    print(f"  • Found {arch} checkpoint: {cand}")
                    break

        # Copy analysis results & plots
        analysis_src = res_path / "analysis"
        if analysis_src.exists():
            shutil.copytree(analysis_src, staging_dir / "analysis")
            print(f"  • Included analysis reports and plots from {analysis_src}")

    # Create README.md model card
    create_model_card(staging_dir / "README.md", repo_id, res_path)

    # 3. Upload folder to Hugging Face Hub
    print(f"\n[3/4] Uploading folder to Hugging Face Hub ({repo_id})...")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload 50M Amharic LM architectures, checkpoints, and benchmark analysis",
    )

    print(f"\n[4/4] Upload complete!")
    print(f"============================================================")
    print(f"  View your repository on Hugging Face Hub:")
    print(f"  👉 https://huggingface.co/{repo_id}")
    print(f"============================================================")


def main():
    parser = argparse.ArgumentParser(description="Upload models & benchmark to Hugging Face")
    parser.add_argument("--token", default=None, help="Hugging Face User Access Token (or set HF_TOKEN env var)")
    parser.add_argument("--username", default="amanuelbyte", help="Hugging Face username")
    parser.add_argument("--repo-name", default=DEFAULT_REPO_NAME, help="Repository name")
    parser.add_argument("--results-dir", default="results", help="Path to results directory")
    args = parser.parse_args()

    upload_to_huggingface(
        token=args.token,
        username=args.username,
        repo_name=args.repo_name,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()

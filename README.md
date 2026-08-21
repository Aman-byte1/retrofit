# Architecture Comparison for Amharic Language Modeling

**Comparing modern sequence architectures on low-resource Amharic text.**

## Architectures

| # | Architecture | Mechanism | Complexity | Source |
|---|---|---|---|---|
| 1 | **Transformer** | RoPE + SwiGLU + RMSNorm + SDPA | O(N²) | Llama-style |
| 2 | **HRM** | Dual-Timescale H/L Modules + Cross-Attention | O(N) | [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text) |
| 3 | **Mamba** | Selective State Space Model (SSM) | O(N) | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| 4 | **Hybrid** | Interleaved Mamba SSM + Transformer Attention | O(N·K) | Jamba/Zamba-inspired |

All models are ~50M parameters for fair comparison.

## Tokenizer

RL-trained Amharic subword tokenizer (3,919 tokens) from `tokenizer_amh`.

## Dataset

Amharic Wikipedia (`wikimedia/wikipedia`, `20231101.am`) from HuggingFace.

## Quick Start

```bash
# On GPU server:
git pull
bash scripts/setup.sh            # Install deps + download/tokenize data
bash scripts/run_experiments.sh   # Train all 4 → benchmark → analyze
```

## Results

After running, find results in `results/analysis/`:
- `report.md` — Full research report
- `results_table.tex` — LaTeX table for papers
- `loss_curves.png` — Training convergence comparison
- `throughput_scaling.png` — Inference speed vs sequence length
- `pareto_frontier.png` — Quality vs efficiency Pareto

## Project Structure

```
├── data/
│   ├── prepare_amharic.py    # Download + tokenize Amharic Wikipedia
│   └── dataset.py            # PyTorch Dataset for tokenized data
├── models/
│   ├── transformer_lm.py     # Modern Transformer (Llama-style)
│   ├── hrm_lm.py             # HRM (Hierarchical Recurrent Memory)
│   ├── mamba_lm.py            # Mamba (Selective SSM)
│   └── hybrid_lm.py          # Hybrid Mamba-Transformer
├── tokenizer/
│   ├── vocab.txt             # RL tokenizer vocabulary (3,919 tokens)
│   └── config.json           # Tokenizer configuration
├── train.py                  # Unified training script
├── benchmark.py              # Throughput + VRAM + latency benchmarks
├── analyze.py                # Plots, LaTeX tables, markdown report
└── scripts/
    ├── setup.sh              # Environment setup
    └── run_experiments.sh    # End-to-end experiment runner
```

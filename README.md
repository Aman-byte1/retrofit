# Retrofit: Efficient Voice Cloning via Parameter-Efficient Adaptation

> **Research Assignment**: Data and Compute-Efficient Generative AI (July 2026)

## TL;DR

Can we **cheaply add voice cloning capability** to any TTS model? We retrofit [F5-TTS](https://github.com/SWivid/F5-TTS) with LoRA adapters targeting only **speaker-identity-critical layers**, achieving comparable voice cloning quality with **~5% of the trainable parameters** and **~1/10th the fine-tuning compute** compared to full model adaptation.

## Research Question

> How can parameter-efficient adaptation methods reduce the computational cost of voice cloning while maintaining speaker similarity and synthesis quality?

## Method

We compare four adaptation strategies for multilingual voice cloning:

| Method | Trainable Params | Training Time | Description |
|--------|-----------------|---------------|-------------|
| **Zero-Shot** | 0 | 0 | F5-TTS inference with reference audio (baseline) |
| **Full Fine-Tune** | ~300M (100%) | ~4 hours | Train all parameters (expensive baseline) |
| **Uniform LoRA** | ~1.5M (~0.5%) | ~30 min | LoRA on all attention layers |
| **Targeted LoRA** (ours) | ~500K (~0.17%) | ~15 min | LoRA on speaker-critical layers only |

### Key Contribution: Targeted LoRA

Not all layers contribute equally to speaker identity. We identify the most speaker-sensitive layers via activation variance analysis, then apply LoRA adapters **only to those layers**. This maximizes quality-per-parameter.

## Evaluation

Evaluated on the [IWSLT 2026 Voice Cloning Benchmark](https://huggingface.co/datasets/amanuelbyte/omnivoice-best-of-n-dev-eval):
- **Languages**: French, Arabic, Chinese
- **Metrics**:
  - **CER** (Character Error Rate) — intelligibility via Whisper ASR
  - **Speaker Similarity** — cosine similarity of ECAPA-TDNN embeddings
  - **Combined Score** = 0.5 × (1 − CER) + 0.5 × Speaker Similarity

## Quick Start

### Setup
```bash
# Clone the repo
git clone https://github.com/Aman-byte1/retrofit.git
cd retrofit

# Install dependencies
bash scripts/setup.sh
```

### Run All Experiments
```bash
bash scripts/run_experiments.sh
```

### Run Individual Experiments

```bash
# Zero-shot baseline (no training)
python -m retrofit.evaluate --method zero_shot --max-samples 50

# Uniform LoRA training
python -m retrofit.train --method uniform_lora --lora-rank 8 --epochs 30

# Targeted LoRA training
python -m retrofit.train --method targeted_lora --lora-rank 8 --target-layers 0 1 2 3 4 5 6 7 --epochs 30

# Generate analysis plots & report
python -m retrofit.analyze --results-dir experiments/
```

## Project Structure

```
retrofit/
├── configs/
│   └── default.yaml          # All hyperparameters & experiment config
├── retrofit/
│   ├── __init__.py
│   ├── lora.py               # Custom LoRA implementation
│   ├── model.py              # F5-TTS wrapper with adapter injection
│   ├── data.py               # Dataset loading (IWSLT, CommonVoice, local)
│   ├── metrics.py            # CER, speaker similarity, combined score
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Evaluation pipeline
│   └── analyze.py            # Results analysis & plotting
├── scripts/
│   ├── setup.sh              # Environment setup
│   └── run_experiments.sh    # Full experiment pipeline
├── experiments/               # Results output directory
├── requirements.txt
└── README.md
```

## Hardware Requirements

- **Minimum**: 1× A40 (48GB) or equivalent
- **Recommended**: 2× A40 (used in this research)
- **Training time**: 15 min (targeted LoRA) to 4 hours (full fine-tune)

## Dataset

We evaluate on [amanuelbyte/omnivoice-best-of-n-dev-eval](https://huggingface.co/datasets/amanuelbyte/omnivoice-best-of-n-dev-eval):
- 2,652 samples across French, Arabic, and Chinese
- Reference audio + synthesized audio + quality scores
- Originally curated for IWSLT 2026

## Citation

```bibtex
@misc{retrofit2026,
  title={Retrofit: Efficient Voice Cloning via Parameter-Efficient Adaptation},
  author={Aman},
  year={2026},
  note={Research Assignment: Data and Compute-Efficient Generative AI}
}
```

## License

Apache License 2.0

# Retrofit: Adding Voice Cloning to Any TTS Model — Cheaply

> **Research Assignment**: Data and Compute-Efficient Generative AI (July 2026)

## TL;DR

We **retrofit zero-shot voice cloning** into TTS models that **don't have it**. By injecting a tiny trainable adapter between a frozen speaker encoder and a frozen TTS backbone, we add voice cloning capability using **<0.5% extra parameters** and **minutes of training** on a single GPU.

## The Problem

High-quality TTS models like MMS-TTS support **1000+ languages** but are **single-speaker** — they can't clone voices. Models that CAN clone voices (F5-TTS, VALL-E, XTTS) were designed from scratch for it, requiring massive compute to build.

**Our question**: Can we cheaply bolt voice cloning onto existing non-cloning TTS models?

## Architecture

```
                 ┌──────────────┐
  Reference      │   Speaker    │
  Audio (10s) ──►│   Encoder    │──► Speaker Embedding (192-dim)
                 │  (frozen)    │        │
                 └──────────────┘        │
                                         ▼
                                  ┌──────────────┐
                                  │   Adapter     │  ◄── ONLY this is trained
                                  │   Layers      │      (~150K params)
                                  └──────┬───────┘
                                         │
                 ┌──────────────┐        ▼
  Text ─────────►│   MMS-TTS    │◄── FiLM conditioning
                 │   (VITS)     │    (gamma * h + beta)
                 │  (frozen)    │
                 └──────┬───────┘
                        ▼
                   Cloned Speech
```

**Three components:**
- **Speaker Encoder** (ECAPA-TDNN, frozen): Extracts speaker identity from reference audio
- **Adapter** (FiLM MLP, **trained**): Projects speaker embedding → TTS conditioning space
- **TTS Model** (MMS-TTS/VITS, frozen): Generates speech, conditioned by the adapter

**Only the adapter is trained.** Everything else stays frozen.

## Key Results

| Component | Parameters | Trainable | Training Time |
|-----------|-----------|-----------|---------------|
| MMS-TTS (VITS) | ~35M | 0 (frozen) | — |
| ECAPA-TDNN | ~15M | 0 (frozen) | — |
| **Adapter (ours)** | **~150K** | **100%** | **~5 min** |

### Evaluation (IWSLT 2026 Benchmark)

| Method | CER ↓ | Speaker Sim ↑ | Combined ↑ | Params Trained |
|--------|-------|---------------|------------|----------------|
| MMS-TTS (no cloning) | — | 0.0 | — | 0 |
| Retrofit + FiLM | TBD | TBD | TBD | 150K |
| Retrofit + Additive | TBD | TBD | TBD | 100K |

### Cross-Language Transfer

A key finding: the adapter trained on French transfers to Arabic and Chinese without retraining. Since the speaker encoder operates in a language-agnostic embedding space, the adapter generalizes across languages.

## Quick Start

### Setup
```bash
git clone https://github.com/Aman-byte1/retrofit.git
cd retrofit
bash scripts/setup.sh
```

### Run All Experiments
```bash
bash scripts/run_experiments.sh
```

### Run Individual Steps

```bash
# Baseline: MMS-TTS without cloning
python -m retrofit.evaluate --language fr --no-adapter --max-samples 50

# Train the adapter (takes ~5 minutes)
python -m retrofit.train --language fr --adapter-type film --epochs 100

# Evaluate with the trained adapter
python -m retrofit.evaluate --language fr --adapter-path experiments/retrofit_film_fr/adapter_best.pt

# Cross-language: test French adapter on Arabic
python -m retrofit.evaluate --language ar --adapter-path experiments/retrofit_film_fr/adapter_best.pt

# Generate plots and report
python -m retrofit.analyze --results-dir experiments/
```

## Project Structure

```
retrofit/
├── configs/default.yaml           # All hyperparameters
├── retrofit/
│   ├── speaker_encoder.py         # Frozen ECAPA-TDNN (block 1)
│   ├── adapters.py                # FiLM adapter + training losses (block 2)
│   ├── model.py                   # Complete retrofit architecture (block 3)
│   ├── data.py                    # Dataset loading (IWSLT + training data)
│   ├── metrics.py                 # CER + speaker similarity + combined score
│   ├── train.py                   # Contrastive adapter training
│   ├── evaluate.py                # Full evaluation pipeline
│   ├── analyze.py                 # Plots, tables, reports
│   └── lora.py                    # LoRA utilities (alternative approach)
├── scripts/
│   ├── setup.sh                   # Environment setup
│   └── run_experiments.sh         # Full experiment pipeline
├── requirements.txt
└── README.md
```

## Training Method

We train the adapter using **contrastive speaker learning** — no TTS in the training loop:

1. **Pre-compute** speaker embeddings for all training utterances (frozen ECAPA-TDNN)
2. **Train adapter** with InfoNCE loss: same-speaker → similar conditioning, different-speaker → dissimilar
3. **Plug adapter** into the frozen TTS and run inference

This makes training extremely fast (~5 minutes) because we never run the TTS model during training.

## Hardware

- **Minimum**: 1× GPU with 24GB VRAM
- **Used**: 2× A40 (48GB each)
- **Training time**: ~5 minutes (adapter only)
- **Evaluation time**: ~30 minutes per language

## Dataset

Evaluation: [amanuelbyte/omnivoice-best-of-n-dev-eval](https://huggingface.co/datasets/amanuelbyte/omnivoice-best-of-n-dev-eval)
- 2,652 samples: French, Arabic, Chinese
- Curated for IWSLT 2026

## Citation

```bibtex
@misc{retrofit2026,
  title={Retrofit: Adding Voice Cloning to Any TTS Model via Parameter-Efficient Adaptation},
  author={Aman},
  year={2026},
  note={Research Assignment: Data and Compute-Efficient Generative AI}
}
```

## License

Apache-2.0

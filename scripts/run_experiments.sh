#!/bin/bash
# ============================================================
# Retrofit: Full Experiment Pipeline
# Run all experiments on 2x A40 GPUs
# ============================================================
set -euo pipefail

echo "============================================================"
echo "  Retrofit: Efficient Voice Cloning Experiments"
echo "  $(date)"
echo "============================================================"

# Configuration
CONFIG="configs/default.yaml"
OUTPUT_DIR="experiments"
MAX_EVAL_SAMPLES=100  # Reduce for quick testing, remove for full eval

# ============================================================
# Experiment 1: Zero-Shot Baseline (no training)
# ============================================================
echo ""
echo "[1/4] Running zero-shot baseline..."
python -m retrofit.evaluate \
    --config $CONFIG \
    --method zero_shot \
    --max-samples $MAX_EVAL_SAMPLES \
    --output-dir $OUTPUT_DIR/zero_shot

# ============================================================
# Experiment 2: Uniform LoRA (rank 8, all layers)
# ============================================================
echo ""
echo "[2/4] Training with Uniform LoRA (rank=8)..."
python -m retrofit.train \
    --config $CONFIG \
    --method uniform_lora \
    --lora-rank 8 \
    --epochs 30 \
    --data-source iwslt \
    --language fr \
    --experiment-name uniform_lora_r8 \
    --output-dir $OUTPUT_DIR

# ============================================================
# Experiment 3: Uniform LoRA (rank 4, fewer params)
# ============================================================
echo ""
echo "[3/4] Training with Uniform LoRA (rank=4)..."
python -m retrofit.train \
    --config $CONFIG \
    --method uniform_lora \
    --lora-rank 4 \
    --epochs 30 \
    --data-source iwslt \
    --language fr \
    --experiment-name uniform_lora_r4 \
    --output-dir $OUTPUT_DIR

# ============================================================
# Experiment 4: Targeted LoRA (top layers only)
# ============================================================
echo ""
echo "[4/4] Training with Targeted LoRA (layers 0-7)..."
python -m retrofit.train \
    --config $CONFIG \
    --method targeted_lora \
    --lora-rank 8 \
    --target-layers 0 1 2 3 4 5 6 7 \
    --epochs 30 \
    --data-source iwslt \
    --language fr \
    --experiment-name targeted_lora_r8_top8 \
    --output-dir $OUTPUT_DIR

# ============================================================
# Analysis: Generate plots and report
# ============================================================
echo ""
echo "Generating analysis..."
python -m retrofit.analyze \
    --results-dir $OUTPUT_DIR \
    --output-dir $OUTPUT_DIR/analysis

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Results: $OUTPUT_DIR/analysis/"
echo "============================================================"

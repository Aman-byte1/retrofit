#!/bin/bash
# ============================================================
# Retrofit: Full Experiment Pipeline (2x A40 GPUs)
#
# Runs the complete research experiment:
# 1. Baseline: TTS without voice cloning
# 2. Retrofit with FiLM adapter
# 3. Retrofit with Additive adapter (ablation)
# 4. Cross-language transfer test
# 5. Analysis & report generation
# ============================================================
set -euo pipefail

echo "============================================================"
echo "  Retrofit: Efficient Voice Cloning Experiments"
echo "  Architecture: Speaker Encoder → Adapter → Frozen TTS"
echo "  $(date)"
echo "============================================================"

CONFIG="configs/default.yaml"
OUTPUT_DIR="experiments"
MAX_EVAL=50  # Samples per language for eval. Remove limit for full run.

# ============================================================
# Experiment 1: Baseline (TTS only, no voice cloning)
# ============================================================
echo ""
echo "[1/5] Baseline: MMS-TTS without voice cloning..."
python -m retrofit.evaluate \
    --config $CONFIG \
    --language fr \
    --no-adapter \
    --max-samples $MAX_EVAL \
    --output-dir $OUTPUT_DIR

# ============================================================
# Experiment 2: Retrofit with FiLM Adapter (main method)
# ============================================================
echo ""
echo "[2/5] Training FiLM adapter on French data..."
python -m retrofit.train \
    --config $CONFIG \
    --language fr \
    --adapter-type film \
    --epochs 100 \
    --data-source iwslt \
    --experiment-name retrofit_film_fr \
    --output-dir $OUTPUT_DIR

# ============================================================
# Experiment 3: Retrofit with Additive Adapter (ablation)
# ============================================================
echo ""
echo "[3/5] Training Additive adapter (ablation)..."
python -m retrofit.train \
    --config $CONFIG \
    --language fr \
    --adapter-type additive \
    --epochs 100 \
    --data-source iwslt \
    --experiment-name retrofit_additive_fr \
    --output-dir $OUTPUT_DIR

# ============================================================
# Experiment 4: Cross-Language Transfer
# ============================================================
echo ""
echo "[4/5] Testing cross-language transfer (adapter trained on FR, tested on AR & ZH)..."
# Use the French-trained FiLM adapter on Arabic and Chinese
python -m retrofit.evaluate \
    --config $CONFIG \
    --language ar zh \
    --adapter-path $OUTPUT_DIR/retrofit_film_fr/adapter_best.pt \
    --max-samples $MAX_EVAL \
    --output-dir $OUTPUT_DIR/cross_language_transfer

# ============================================================
# Analysis
# ============================================================
echo ""
echo "[5/5] Generating analysis..."
python -m retrofit.analyze \
    --results-dir $OUTPUT_DIR \
    --output-dir $OUTPUT_DIR/analysis

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Results: $OUTPUT_DIR/analysis/"
echo "============================================================"

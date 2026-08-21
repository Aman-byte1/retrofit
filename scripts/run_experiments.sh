#!/bin/bash
# ============================================================
# Full Architecture Comparison Benchmark
#
# Trains 4 architectures on Amharic Wikipedia, benchmarks all,
# and generates analysis report.
# ============================================================
set -euo pipefail

EPOCHS=5
BATCH_SIZE=32
SEQ_LEN=512
LR=3e-4
VOCAB_SIZE=3919
RESULTS_DIR="results"

echo "============================================================"
echo "  Amharic LM Architecture Comparison"
echo "  Architectures: Transformer, HRM, Mamba, Hybrid"
echo "  Tokenizer: RL-trained Amharic (${VOCAB_SIZE} tokens)"
echo "  Target: ~50M parameters each"
echo "  $(date)"
echo "============================================================"

# ============================================================
# Step 0: Setup (if not already done)
# ============================================================
if [ ! -f "data/tokenized/train.bin" ] && [ ! -f "data/tokenized/train_tokens.npy" ]; then
    echo ""
    echo "[0/6] Running setup..."
    bash scripts/setup.sh
fi

# ============================================================
# Step 1: Train Transformer (Qwen3.5 3:1 DeltaNet:Attention + MTP)
# ============================================================
echo ""
echo "============================================================"
echo "[1/6] Training: TRANSFORMER (Qwen3.5 DeltaNet + Gated Attention + MTP)"
echo "============================================================"
python train.py \
    --arch transformer \
    --vocab-size $VOCAB_SIZE \
    --data-dir data/tokenized \
    --output-dir $RESULTS_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --grad-accum 1 \
    --seq-len $SEQ_LEN \
    --lr $LR

# ============================================================
# Step 2: Train HRM-Text (Dual-timescale Recurrence)
# ============================================================
echo ""
echo "============================================================"
echo "[2/6] Training: HRM-Text (Hierarchical Reasoning Model)"
echo "============================================================"
python train.py \
    --arch hrm \
    --vocab-size $VOCAB_SIZE \
    --data-dir data/tokenized \
    --output-dir $RESULTS_DIR \
    --epochs $EPOCHS \
    --batch-size 16 \
    --grad-accum 2 \
    --seq-len $SEQ_LEN \
    --lr $LR

# ============================================================
# Step 3: Train Mamba (Selective SSM)
# ============================================================
echo ""
echo "============================================================"
echo "[3/6] Training: MAMBA (Selective State Space Model)"
echo "============================================================"
python train.py \
    --arch mamba \
    --vocab-size $VOCAB_SIZE \
    --data-dir data/tokenized \
    --output-dir $RESULTS_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --grad-accum 1 \
    --seq-len $SEQ_LEN \
    --lr $LR \
    --mamba-backend auto

# ============================================================
# Step 4: Train Hybrid (Mamba + Qwen3.5 Attention)
# ============================================================
echo ""
echo "============================================================"
echo "[4/6] Training: HYBRID (Mamba SSM + Qwen3.5 Attention)"
echo "============================================================"
python train.py \
    --arch hybrid \
    --vocab-size $VOCAB_SIZE \
    --data-dir data/tokenized \
    --output-dir $RESULTS_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --grad-accum 1 \
    --seq-len $SEQ_LEN \
    --lr $LR \
    --mamba-backend auto

# ============================================================
# Step 5: Benchmark all models
# ============================================================
echo ""
echo "============================================================"
echo "[5/6] Benchmarking: Throughput, VRAM, and Generation Latency"
echo "============================================================"

for ARCH in transformer hrm mamba hybrid; do
    echo ""
    echo "--- Benchmarking ${ARCH} ---"
    python benchmark.py \
        --arch $ARCH \
        --vocab-size $VOCAB_SIZE \
        --checkpoint $RESULTS_DIR/$ARCH/best_model.pt \
        --output-dir $RESULTS_DIR \
        --seq-lengths 128 256 512 1024 2048
done

# ============================================================
# Step 6: Analysis
# ============================================================
echo ""
echo "============================================================"
echo "[6/6] Generating analysis report and plots"
echo "============================================================"
python analyze.py \
    --results-dir $RESULTS_DIR \
    --output-dir $RESULTS_DIR/analysis

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE!"
echo "  Results:  ${RESULTS_DIR}/analysis/"
echo "  Report:   ${RESULTS_DIR}/analysis/report.md"
echo "  LaTeX:    ${RESULTS_DIR}/analysis/results_table.tex"
echo "  Plots:    ${RESULTS_DIR}/analysis/*.png"
echo "  $(date)"
echo "============================================================"

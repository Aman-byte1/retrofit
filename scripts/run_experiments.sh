#!/bin/bash
# ============================================================
# Full Architecture Comparison Benchmark (50M Parameters)
#
# Trains 4 architectures on Amharic Wikipedia, benchmarks all,
# and generates comprehensive comparative report and plots.
# ============================================================
set -euo pipefail

EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-64}
SEQ_LEN=${SEQ_LEN:-512}
LR=${LR:-3e-4}
WARMUP_STEPS=${WARMUP_STEPS:-50}
VOCAB_SIZE=3919
RESULTS_DIR="results"
FORCE_RETRAIN=${FORCE_RETRAIN:-0}

echo "============================================================"
echo "  Amharic LM Architecture Comparison"
echo "  Architectures: Transformer, HRM, Mamba, Hybrid"
echo "  Tokenizer: RL-trained Amharic (${VOCAB_SIZE} tokens)"
echo "  Target: ~50M parameters each"
echo "  Epochs: ${EPOCHS} | Batch Size: ${BATCH_SIZE} | Seq Len: ${SEQ_LEN}"
echo "  $(date)"
echo "============================================================"

# Helper to check if model already has a checkpoint
has_checkpoint() {
    local arch=$1
    if [ "$FORCE_RETRAIN" = "1" ]; then
        return 1
    fi
    if [ -f "$RESULTS_DIR/$arch/best_model.pt" ]; then
        return 0
    fi
    # Check timestamped directories
    for d in "$RESULTS_DIR"/${arch}_*; do
        if [ -d "$d" ] && [ -f "$d/best_model.pt" ]; then
            return 0
        fi
    done
    return 1
}

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
if has_checkpoint "transformer"; then
    echo "  ✓ Transformer already trained! Skipping to next architecture."
else
    python train.py \
        --arch transformer \
        --vocab-size $VOCAB_SIZE \
        --data-dir data/tokenized \
        --output-dir $RESULTS_DIR \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --grad-accum 1 \
        --warmup-steps $WARMUP_STEPS \
        --seq-len $SEQ_LEN \
        --lr $LR
fi

# ============================================================
# Step 2: Train HRM-Text (Dual-timescale Recurrence)
# ============================================================
echo ""
echo "============================================================"
echo "[2/6] Training: HRM-Text (Hierarchical Reasoning Model)"
echo "============================================================"
if has_checkpoint "hrm"; then
    echo "  ✓ HRM already trained! Skipping to next architecture."
else
    python train.py \
        --arch hrm \
        --vocab-size $VOCAB_SIZE \
        --data-dir data/tokenized \
        --output-dir $RESULTS_DIR \
        --epochs $EPOCHS \
        --batch-size 32 \
        --grad-accum 2 \
        --warmup-steps $WARMUP_STEPS \
        --seq-len $SEQ_LEN \
        --lr $LR
fi

# ============================================================
# Step 3: Train Mamba (Selective SSM)
# ============================================================
echo ""
echo "============================================================"
echo "[3/6] Training: MAMBA (Selective State Space Model)"
echo "============================================================"
if has_checkpoint "mamba"; then
    echo "  ✓ Mamba already trained! Skipping to next architecture."
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python train.py \
        --arch mamba \
        --vocab-size $VOCAB_SIZE \
        --data-dir data/tokenized \
        --output-dir $RESULTS_DIR \
        --epochs $EPOCHS \
        --batch-size 16 \
        --grad-accum 4 \
        --warmup-steps $WARMUP_STEPS \
        --seq-len $SEQ_LEN \
        --lr $LR \
        --mamba-backend auto
fi

# ============================================================
# Step 4: Train Hybrid (Mamba + Qwen3.5 Attention)
# ============================================================
echo ""
echo "============================================================"
echo "[4/6] Training: HYBRID (Mamba SSM + Qwen3.5 Attention)"
echo "============================================================"
if has_checkpoint "hybrid"; then
    echo "  ✓ Hybrid already trained! Skipping to next architecture."
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python train.py \
        --arch hybrid \
        --vocab-size $VOCAB_SIZE \
        --data-dir data/tokenized \
        --output-dir $RESULTS_DIR \
        --epochs $EPOCHS \
        --batch-size 16 \
        --grad-accum 4 \
        --warmup-steps $WARMUP_STEPS \
        --seq-len $SEQ_LEN \
        --lr $LR \
        --mamba-backend auto
fi

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
    
    # Locate best checkpoint
    CKPT_PATH="$RESULTS_DIR/$ARCH/best_model.pt"
    if [ ! -f "$CKPT_PATH" ]; then
        for d in "$RESULTS_DIR"/${ARCH}_*; do
            if [ -d "$d" ] && [ -f "$d/best_model.pt" ]; then
                CKPT_PATH="$d/best_model.pt"
                break
            fi
        done
    fi

    python benchmark.py \
        --arch $ARCH \
        --vocab-size $VOCAB_SIZE \
        --checkpoint "$CKPT_PATH" \
        --output-dir $RESULTS_DIR \
        --seq-lengths 128 256 512 1024 2048
done

# ============================================================
# Step 6: Analysis
# ============================================================
echo ""
echo "============================================================"
echo "[6/7] Generating analysis report and plots"
echo "============================================================"
python analyze.py \
    --results-dir $RESULTS_DIR \
    --output-dir $RESULTS_DIR/analysis

# ============================================================
# Step 7: Upload to Hugging Face Hub
# ============================================================
echo ""
echo "============================================================"
echo "[7/7] Uploading Models & Benchmark to Hugging Face Hub"
echo "============================================================"
python scripts/upload_to_hf.py \
    --username amanuelbyte \
    --repo-name amharic-50m-architecture-benchmark \
    --results-dir $RESULTS_DIR || echo "  HF upload skipped or failed (can run manually: python scripts/upload_to_hf.py)"

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE!"
echo "  Results:  ${RESULTS_DIR}/analysis/"
echo "  Report:   ${RESULTS_DIR}/analysis/report.md"
echo "  LaTeX:    ${RESULTS_DIR}/analysis/results_table.tex"
echo "  Plots:    ${RESULTS_DIR}/analysis/*.png"
echo "  Hugging Face: https://huggingface.co/amanuelbyte/amharic-50m-architecture-benchmark"
echo "  $(date)"
echo "============================================================"

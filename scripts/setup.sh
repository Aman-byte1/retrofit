#!/bin/bash
# ============================================================
# Setup: Install dependencies for architecture comparison
# ============================================================
set -euo pipefail

echo "============================================================"
echo "  Setting up Amharic LM Architecture Comparison"
echo "  $(date)"
echo "============================================================"

# 1. Install base dependencies
echo "[1/3] Installing base dependencies..."
pip install -q datasets numpy matplotlib seaborn pandas

# 2. Try to install mamba-ssm (CUDA fused kernels)
echo "[2/3] Installing Mamba SSM (CUDA fused kernels)..."
pip install --no-build-isolation causal-conv1d>=1.4.0 mamba-ssm>=2.2.2 2>/dev/null || \
pip install causal-conv1d mamba-ssm 2>/dev/null || \
echo "  Note: mamba-ssm build skipped (will use pure PyTorch fallback with full dt_rank=32)"

# 3. Prepare data with ultra-fast parallel tokenizer
echo "[3/3] Downloading and tokenizing Amharic Wikipedia..."
python -m data.prepare_amharic \
    --tokenizer-dir tokenizer \
    --output-dir data/tokenized \
    --cache-dir data/raw \
    --num-workers 4

echo ""
echo "============================================================"
echo "  Setup complete! Ready to train."
echo "============================================================"

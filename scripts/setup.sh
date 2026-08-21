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

# 2. Try to install mamba-ssm (needs CUDA build tools)
echo "[2/3] Installing Mamba SSM (optional, falls back to pure PyTorch)..."
pip install -q causal-conv1d>=1.2.0 2>/dev/null || echo "  causal-conv1d install failed (will use fallback)"
pip install -q mamba-ssm 2>/dev/null || echo "  mamba-ssm install failed (will use pure PyTorch SSM)"

# 3. Prepare data
echo "[3/3] Downloading and tokenizing Amharic Wikipedia..."
python -m data.prepare_amharic \
    --tokenizer-dir tokenizer \
    --output-dir data/tokenized \
    --cache-dir data/raw

echo ""
echo "============================================================"
echo "  Setup complete! Ready to train."
echo "============================================================"

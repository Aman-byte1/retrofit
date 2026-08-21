#!/bin/bash
# ============================================================
# Environment Setup for Retrofit
# Run this on each GPU node before experiments
# ============================================================
set -euo pipefail

echo "Setting up Retrofit environment..."

# Check Python version
python3 --version || { echo "Python 3 required"; exit 1; }

# Check CUDA
nvidia-smi || { echo "NVIDIA GPU not found"; exit 1; }

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical imports
echo ""
echo "Verifying imports..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')

import torchaudio
print(f'Torchaudio: {torchaudio.__version__}')

import datasets
print(f'Datasets: {datasets.__version__}')

# Test F5-TTS import
try:
    from f5_tts.api import F5TTS
    print('F5-TTS: OK')
except ImportError as e:
    print(f'F5-TTS: FAILED ({e})')

# Test SpeechBrain import
try:
    import speechbrain
    print(f'SpeechBrain: {speechbrain.__version__}')
except ImportError as e:
    print(f'SpeechBrain: FAILED ({e})')

# Test Whisper import
try:
    from faster_whisper import WhisperModel
    print('Faster-Whisper: OK')
except ImportError as e:
    print(f'Faster-Whisper: FAILED ({e})')
"

echo ""
echo "Setup complete! Run experiments with:"
echo "  bash scripts/run_experiments.sh"

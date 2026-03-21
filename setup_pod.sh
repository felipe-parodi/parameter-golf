#!/bin/bash
# Parameter Golf pod setup script
# Usage: bash setup_pod.sh
set -e

cd /workspace
git clone https://github.com/felipe-parodi/parameter-golf.git
cd parameter-golf
git checkout feat/competitive-submission

# Data download
python3 data/cached_challenge_fineweb.py --variant sp1024

# Deps (template should have most, but just in case)
pip install sentencepiece zstandard huggingface_hub 2>/dev/null

# FA3 Hopper build (background)
echo "Starting FA3 build in background..."
(cd /tmp && git clone https://github.com/Dao-AILab/flash-attention.git 2>/dev/null; cd /tmp/flash-attention/hopper && pip install . --no-build-isolation) &
FA3_PID=$!

# Verify
python3 -c "from flash_attn import flash_attn_func; print('FA2 OK')" 2>/dev/null || echo "WARNING: No flash_attn installed"
echo ""
echo "========================================="
echo "SETUP DONE. Data ready, FA3 building (pid $FA3_PID)"
echo "To check FA3: python3 -c \"from flash_attn_interface import flash_attn_func; print('FA3 OK')\""
echo "========================================="

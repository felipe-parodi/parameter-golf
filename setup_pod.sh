#!/bin/bash
# Parameter Golf pod setup script
# Run from anywhere: curl + bash, or clone then bash setup_pod.sh
set -e

REPO_DIR=/workspace/parameter-golf

# Clone or update repo
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists, pulling latest..."
    cd "$REPO_DIR" && git fetch && git checkout feat/competitive-submission && git pull
else
    cd /workspace
    git clone https://github.com/felipe-parodi/parameter-golf.git
    cd parameter-golf
    git checkout feat/competitive-submission
fi

# Deps
pip install -q sentencepiece zstandard huggingface_hub 2>/dev/null || true
pip install flash-attn --no-build-isolation --no-cache-dir 2>/dev/null || echo "WARNING: flash-attn install failed, will retry"

# Data download
python3 data/cached_challenge_fineweb.py --variant sp1024

# FA3 Hopper build (background)
echo "Starting FA3 build in background..."
(cd /tmp && rm -rf flash-attention && git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention/hopper && pip install . --no-build-isolation) &
FA3_PID=$!

# Verify
python3 -c "from flash_attn import flash_attn_func; print('FA2 OK')" 2>/dev/null || echo "WARNING: No flash_attn"
echo ""
echo "========================================="
echo "SETUP DONE — cd $REPO_DIR"
echo "FA3 building in background (pid $FA3_PID)"
echo "Check: python3 -c \"from flash_attn_interface import flash_attn_func; print('FA3 OK')\""
echo "========================================="

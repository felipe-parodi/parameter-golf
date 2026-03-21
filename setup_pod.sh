#!/bin/bash
# Parameter Golf pod setup — one command, any fresh pod
# Usage: curl -sL https://raw.githubusercontent.com/felipe-parodi/parameter-golf/feat/competitive-submission/setup_pod.sh | bash
# Or: cd /workspace/parameter-golf && bash setup_pod.sh
set -e

REPO_DIR=/workspace/parameter-golf
BRANCH=feat/competitive-submission

# Clone or update repo
if [ -d "$REPO_DIR/.git" ]; then
    echo ">>> Repo exists, pulling latest..."
    cd "$REPO_DIR" && git fetch && git checkout "$BRANCH" && git pull
else
    echo ">>> Cloning repo..."
    cd /workspace
    git clone https://github.com/felipe-parodi/parameter-golf.git
    cd parameter-golf
    git checkout "$BRANCH"
fi

# Deps
echo ">>> Installing dependencies..."
pip install -q sentencepiece zstandard huggingface_hub 2>/dev/null || true
pip install flash-attn --no-build-isolation --no-cache-dir 2>/dev/null || echo "FA2 install failed (non-fatal)"

# FA3: pre-built wheel (seconds)
echo ">>> Installing FA3 (pre-built wheel)..."
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>/dev/null \
  && echo ">>> FA3 OK" \
  || echo ">>> FA3 wheel failed — will fall back to FA2"

# Data
echo ">>> Downloading data..."
python3 data/cached_challenge_fineweb.py --variant sp1024

# Verify
echo ""
echo "========================================="
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3: OK')" 2>/dev/null || \
python3 -c "from flash_attn import flash_attn_func; print('FA2: OK')" 2>/dev/null || \
echo "WARNING: No flash attention installed!"
echo "Setup complete. cd $REPO_DIR"
echo "========================================="

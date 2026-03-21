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

# FA3 Hopper: try pre-built wheel first, then minimal source build
echo "Installing FA3..."
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>/dev/null \
  && echo "FA3 wheel installed" \
  || (echo "FA3 wheel failed, doing minimal source build..." && \
      cd /tmp && rm -rf flash-attention && git clone --depth=1 https://github.com/Dao-AILab/flash-attention.git && \
      cd flash-attention/hopper && \
      MAX_JOBS=$(nproc) FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
      FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
      FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE \
      FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
      FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
      FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM128=TRUE \
      FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
      TORCH_CUDA_ARCH_LIST="9.0a" python setup.py install)

# Verify
python3 -c "from flash_attn import flash_attn_func; print('FA2 OK')" 2>/dev/null || echo "WARNING: No flash_attn"
echo ""
echo "========================================="
echo "SETUP DONE — cd $REPO_DIR"
echo "FA3 building in background (pid $FA3_PID)"
echo "Check: python3 -c \"from flash_attn_interface import flash_attn_func; print('FA3 OK')\""
echo "========================================="

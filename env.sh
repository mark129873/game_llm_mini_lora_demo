#!/usr/bin/env bash
# Source this file from Bash before running the test.
LORA_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$LORA_PROJECT/cache/huggingface"
export XDG_CACHE_HOME="$LORA_PROJECT/cache"
export UV_CACHE_DIR="$LORA_PROJECT/cache/uv"
export PIP_CACHE_DIR="$LORA_PROJECT/cache/pip"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false
export HSA_ENABLE_DXG_DETECTION=1
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$LORA_PROJECT/.venv/bin/activate"

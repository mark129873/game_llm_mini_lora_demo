#!/usr/bin/env bash
# 用途：为当前 Bash 会话设置 uv 路径、ROCm 显卡识别和项目缓存位置。
# 用法：进入项目后 source env.sh；随后运行 uv run lora_demo.py。
# uv 自动使用 .venv，本文件不激活或创建 Python 虚拟环境。

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请使用 source env.sh"
    exit 1
fi
LORA_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# uv 是独立工具；安装到用户目录后，即使重建 .venv 也仍可调用。
export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"

# 下载缓存和 GPU 编译缓存集中放在项目内，便于清理。
export UV_CACHE_DIR="$LORA_PROJECT/cache/uv"
export HF_HOME="$LORA_PROJECT/cache/huggingface"
export XDG_CACHE_HOME="$LORA_PROJECT/cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false

# 启用 WSL 的 DXG 显卡发现，并提供 ROCm 与 Windows 透传库的搜索路径。
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

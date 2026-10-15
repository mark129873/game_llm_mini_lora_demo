#!/usr/bin/env bash
# 用途：在当前 Bash 会话激活项目的 Python 虚拟环境，配置 ROCm GPU 与缓存路径。
# 用法：cd /home/nnnnnn/lora && source env.sh
# 前提：install_rocm.sh 和 install_python.sh 已完成；也由 Python 安装脚本调用。
# 生效范围：当前终端及其子进程；新开终端时需要再次 source。
# 路径：.venv、Hugging Face、pip、uv 及 GPU 编译缓存都定位到本项目。
# env.sh 必须被 source，直接执行子进程无法修改父终端的环境。

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请使用 source env.sh 激活当前终端"
    exit 1
fi
LORA_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$LORA_PROJECT/.venv/bin/activate" ]]; then
    echo "缺少 .venv，请先运行 bash install_python.sh"
    return 1
fi

# 将模型下载缓存和安装缓存集中放在项目内，方便定位和清理。
export HF_HOME="$LORA_PROJECT/cache/huggingface"
export XDG_CACHE_HOME="$LORA_PROJECT/cache"
export UV_CACHE_DIR="$LORA_PROJECT/cache/uv"
export PIP_CACHE_DIR="$LORA_PROJECT/cache/pip"

# 使用普通 HTTP 下载，增加读取超时；关闭分词器后台线程的重复并行。
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false

# 启用 WSL DXG 设备发现，并让程序找到 ROCm 命令、HIP 及 Windows 透传库。
export HSA_ENABLE_DXG_DETECTION=1
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 避免在源码目录生成 __pycache__；Python 会直接在内存中编译模块。
export PYTHONDONTWRITEBYTECODE=1
source "$LORA_PROJECT/.venv/bin/activate"

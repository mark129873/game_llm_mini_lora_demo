#!/usr/bin/env bash
# 用途：安装 uv，按锁文件同步 Python 环境，然后下载固定版本的基础模型。
# 执行：在 Ubuntu-24.04 普通用户终端运行 bash install_python.sh。
# 前提：先由 root 执行 install_rocm.sh。
# 版本：.python-version 选择 Python，pyproject.toml 声明依赖，uv.lock 锁定版本与校验值。
# 目录：uv 管理 .venv；安装缓存放在 cache/，原始模型放在 models/。

set -euo pipefail
[[ "$EUID" -ne 0 ]] || { echo "请使用普通 WSL 用户执行此脚本"; exit 1; }
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"
source env.sh

# uv 安装在用户目录，独立于项目环境；安装器不修改 shell 配置文件。
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/0.13.0/install.sh \
        | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi

# 自动选择 .python-version 指定的 Python；找不到时由 uv 下载。
# --locked 保证按已提交的锁文件安装；AMD wheel 来源及哈希也由 uv 管理。
uv sync --locked

# 固定模型提交，下载原始 safetensors、分词器与配置；已有文件可复用。
uv run --locked python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    local_dir="models/Qwen3.5-0.8B",
    allow_patterns=["*.json", "*.jinja", "*.safetensors"],
)
PY

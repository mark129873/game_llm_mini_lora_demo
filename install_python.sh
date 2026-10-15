#!/usr/bin/env bash
# 用途：为 LoRA demo 创建 Python 3.12 虚拟环境，安装 AMD PyTorch 并下载基础模型。
# 执行：在 Ubuntu-24.04 普通用户终端运行 bash install_python.sh。
# 前提：先以 root 执行 install_rocm.sh；本脚本不修改系统 Python。
# 产物：.venv/ 为运行环境，downloads/ 为安装包，cache/ 为缓存，models/ 为模型。
# 版本：requirements.txt 固定 Python 包；Torch/Triton 使用配套的 AMD cp312 wheel。
# 下载：安装包支持断点续传并校验 SHA256；模型固定到指定 Hugging Face revision。

set -euo pipefail
[[ "$EUID" -ne 0 ]] || { echo "请使用普通 WSL 用户执行此脚本"; exit 1; }
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "需要 Python 3.12"'

# 先创建隔离环境，再加载 env.sh；之后的 pip/uv 只操作本项目 .venv。
mkdir -p cache downloads models
python3 -m venv .venv
source env.sh
python -m pip install pip==26.2.1 uv==0.13.0

# 校验已下载的完整文件；否则续传 .part，校验成功后再改为最终文件名。
# 安装包与运行库分开：删除下载包不影响已经安装到 .venv 的模块。
download_wheel() {
    local url="$1" target="$2" expected_sha="$3"
    if [[ -f "$target" ]] && printf '%s  %s\n' "$expected_sha" "$target" \
        | sha256sum --check --status; then
        return
    fi
    if [[ -f "$target.part" ]] && printf '%s  %s\n' "$expected_sha" "$target.part" \
        | sha256sum --check --status; then
        mv -- "$target.part" "$target"
        return
    fi
    curl -fL --retry 3 --continue-at - -o "$target.part" "$url"
    printf '%s  %s\n' "$expected_sha" "$target.part" | sha256sum --check
    mv -- "$target.part" "$target"
}

# 两个 wheel 必须一起提供给解析器；通用 PyPI 源没有这一版 AMD Triton。
torch_whl=downloads/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl
triton_whl=downloads/triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
download_wheel \
    'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
    "$torch_whl" 'fb45ace0a27e9f0d0e3c4c6efd8932162743f8376f2aa4752a4d31ef5a1bd3d7'
download_wheel \
    'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl' \
    "$triton_whl" '07787af1d28c273852f897bfeaa7bca29f2fa4a13ca0f28f535832b240ce7016'
uv pip install "$torch_whl" "$triton_whl" -r requirements.txt
python -m pip check

# 下载官方原始 safetensors 和 tokenizer；demo 直接读取这个本地目录。
# 固定提交以保持可复现，下载工作交给 Hugging Face 的缓存/校验机制。
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    local_dir="models/Qwen3.5-0.8B",
    allow_patterns=["*.json", "*.jinja", "*.safetensors"],
)
PY

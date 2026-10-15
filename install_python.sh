#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"
[[ "$EUID" -ne 0 ]] || { echo "Run as the normal WSL user, not root."; exit 1; }
mkdir -p cache downloads logs models outputs
python3 -m venv .venv
source env.sh
python -m pip install pip==26.2.1 uv==0.13.0
torch_whl=downloads/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl
triton_whl=downloads/triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
python -u download_file.py 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' "$torch_whl" --workers 8
python -u download_file.py 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl' "$triton_whl" --workers 8
if [[ -f downloads/SHA256SUMS ]]; then sha256sum -c downloads/SHA256SUMS; fi
requirements=requirements.txt
if [[ -f requirements-lock.txt ]]; then requirements=requirements-lock.txt; fi
uv pip install "$torch_whl" "$triton_whl" -r "$requirements"
python -m pip check

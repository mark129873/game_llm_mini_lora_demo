#!/usr/bin/env bash
# 用途：在 WSL2 Ubuntu 24.04 中安装 RX 7700 XT 需要的 ROCm 系统运行库。
# 执行：wsl -d Ubuntu-24.04 -u root --exec bash /home/nnnnnn/lora/install_rocm.sh
# 权限：必须由 root 运行；Python 虚拟环境由 install_python.sh 单独配置。
# 前提：Windows 已安装可供 WSL 使用的 AMD 显卡驱动，WSL 中存在 /dev/dxg。
# 配置：添加 AMD APT 软件源、ROCm 7.2.1、ROCDXG 1.2.2 和动态链接器搜索路径。
# 文件：系统库放在 /opt/rocm 等系统目录，下载的 deb 放在本项目 downloads/。
# 本脚本使用 WSL 用户态驱动接口，不安装 Linux amdgpu DKMS 内核驱动。

set -euo pipefail
[[ "$EUID" -eq 0 ]] || { echo "请使用 WSL root 用户执行此脚本"; exit 1; }
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_USER="$(stat -c '%U' "$PROJECT")"
PROJECT_GROUP="$(stat -c '%G' "$PROJECT")"
export DEBIAN_FRONTEND=noninteractive

# curl/GnuPG 用于下载和验证软件源；libnuma1 是 HIP 的系统依赖。
apt-get update -qq
apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg libnuma1

# 固定使用 ROCm 7.2.1 的 Ubuntu noble 仓库，并为该源配置独立签名密钥。
install -d /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --batch --yes --dearmor -o /etc/apt/keyrings/rocm.gpg
chmod 644 /etc/apt/keyrings/rocm.gpg
printf '%s\n' \
    'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.1 noble main' \
    > /etc/apt/sources.list.d/rocm-7.2.1.list

# 优先选择 AMD 源，避免 Ubuntu 自带的 rocminfo 与固定版本运行库依赖冲突。
printf '%s\n' 'Package: *' 'Pin: origin repo.radeon.com' 'Pin-Priority: 600' \
    > /etc/apt/preferences.d/rocm-radeon-pin-600
apt-get update -qq

# HIP 提供设备运行时；BLAS 提供矩阵计算；MIOpen 提供神经网络算子；
# FFT、随机数、稀疏矩阵、求解器和 RCCL 是这个 AMD PyTorch wheel 的动态库依赖。
# roctracer 提供 libroctx64/libroctracer64；仅装 HIP 会导致 import torch 缺库。
apt-get install -y --no-install-recommends \
    rocm-hip-runtime roctracer miopen-hip \
    rocblas hipblas hipblaslt hipfft hiprand \
    hipsparse hipsolver rocsolver hipsparselt rccl

# ROCDXG 将 ROCm 设备访问接到 WSL 的 /dev/dxg；校验固定 deb 后再安装。
install -d -o "$PROJECT_USER" -g "$PROJECT_GROUP" "$PROJECT/downloads"
ROCDXG_DEB="$PROJECT/downloads/rocdxg-roct_1.2.2_amd64.deb"
curl -fL --retry 3 -o "$ROCDXG_DEB" \
    https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
printf '%s  %s\n' \
    '28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541' "$ROCDXG_DEB" \
    | sha256sum --check
apt-get install -y "$ROCDXG_DEB"
chown "$PROJECT_USER:$PROJECT_GROUP" "$ROCDXG_DEB"

# 将 ROCm 和 Windows 注入 WSL 的 GPU 库加入动态链接器搜索路径。
printf '%s\n' '/opt/rocm/lib' '/usr/lib/wsl/lib' > /etc/ld.so.conf.d/rocm-wsl.conf
ldconfig

# 在终端验证 GPU 是否可见；不额外生成安装日志或报告文件。
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo

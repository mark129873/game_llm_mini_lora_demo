#!/usr/bin/env bash
# Run as root in Ubuntu-24.04. Project files remain owned by nnnnnn.
set -euo pipefail
[[ "$EUID" -eq 0 ]] || { echo "Run as root (see INSTALL.md)."; exit 1; }
PROJECT=/home/nnnnnn/lora
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends python3-venv python3-pip curl ca-certificates gnupg libnuma1 pciutils
install -d /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --batch --yes --dearmor -o /etc/apt/keyrings/rocm.gpg
chmod 644 /etc/apt/keyrings/rocm.gpg
printf '%s\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.1 noble main' > /etc/apt/sources.list.d/rocm-7.2.1.list
printf '%s\n' 'Package: *' 'Pin: origin repo.radeon.com' 'Pin-Priority: 600' > /etc/apt/preferences.d/rocm-radeon-pin-600
apt-get update -qq
apt-get install -y --no-install-recommends rocm-hip-runtime
apt-get install -y --no-install-recommends miopen-hip hipblas hipfft hiprand hipsparse hipsolver rocsolver hipsparselt rccl rocblas hipblaslt roctracer
install -d -o nnnnnn -g nnnnnn "$PROJECT/downloads" "$PROJECT/logs"
curl -fL --retry 3 -o "$PROJECT/downloads/rocdxg-roct_1.2.2_amd64.deb" https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
apt-get install -y "$PROJECT/downloads/rocdxg-roct_1.2.2_amd64.deb"
printf '%s\n' '/opt/rocm/lib' '/usr/lib/wsl/lib' > /etc/ld.so.conf.d/rocm-wsl.conf
ldconfig
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo > "$PROJECT/logs/rocminfo.txt" 2>&1
chown nnnnnn:nnnnnn "$PROJECT/downloads/rocdxg-roct_1.2.2_amd64.deb" "$PROJECT/logs/rocminfo.txt"
cat "$PROJECT/logs/rocminfo.txt"

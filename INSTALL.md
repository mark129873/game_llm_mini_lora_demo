# Qwen3.5-0.8B LoRA：本机安装记录与重装步骤

记录日期：2026-10-13。项目目录：`/home/nnnnnn/lora`。所有项目文件、模型、Python 虚拟环境、缓存、训练结果与本文均在 Ubuntu-24.04 的 Linux 文件系统里。ROCm 系统库由 APT 安装到同一个 WSL 发行版的 `/opt/rocm*` 等系统目录。

本文把实际执行的命令整理为可重复的安装顺序，重复的状态查询做了合并。失败尝试及修复见后文。脚本源文件也保留在本目录。**本次已经通过训练、保存、重载及 WSL 重启后的独立推理验证。结果见 TEST_RESULTS.md 与 outputs/smoke-test/report.json。**

## 1. 本机与版本选择

| 项目 | 本次配置 |
|---|---|
| Windows | Windows 11 Pro，build 26200 |
| CPU / 物理内存 | i9-9880H，8 核 16 线程 / 16 GB |
| 用于训练的显卡 | AMD Radeon RX 7700 XT，12 GB，gfx1101 |
| Windows 显卡驱动 | 32.0.31007.5012，2026-05-12；本次保留现有驱动 |
| WSL 发行版 | 新装 Ubuntu-24.04；镜像实际为 24.04.5 LTS |
| WSL 配额 | 内存 8 GB、6 个逻辑处理器、swap 4 GB |
| Linux Python | 3.12；独立 .venv |
| ROCm / ROCDXG | 7.2.1 / 1.2.2 |
| PyTorch | 2.9.1+rocm7.2.1.lw.gitff65f5bc，cp312 AMD wheel |
| Triton | 3.5.1+rocm7.2.1.gita272dfa8，cp312 AMD wheel |
| Transformers / PEFT | 5.19.0 / 0.21.2 |
| NumPy | 1.26.4 |
| 模型 | Qwen/Qwen3.5-0.8B，官方 safetensors |
| 固定模型 revision | 2fc06364715b967f1860aea9cf38778875588b17 |

本机已有 Ubuntu-26.04 和 docker-desktop。此次单独安装 Ubuntu-24.04，没有卸载旧发行版，也没有改默认发行版。每条 Windows 命令显式指定 `-d Ubuntu-24.04`。

桌面的 `C:\Users\nnnnnn\Desktop\model\Qwen3.5-0.8B-Q4_0.gguf` 保留原样。这次训练直接使用官方原始权重，保存 PEFT LoRA adapter。**没有将 Q4_0 GGUF 反量化来训练，也没有执行 GGUF 转换或合并。** 迁移目录不会改变模型格式；adapter 与 GGUF 的后续导出属于单独步骤。

## 2. Windows：检查、调整 WSL、安装 Ubuntu

以下在 Windows PowerShell 执行。

### 2.1 检查配置（只读）

```powershell
wsl.exe --version
wsl.exe --status
wsl.exe --list --verbose
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,BuildNumber,TotalVisibleMemorySize,FreePhysicalMemory
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate
Get-PSDrive C
Get-Content -LiteralPath "$env:USERPROFILE\.wslconfig"
Get-Item -LiteralPath 'C:\Users\nnnnnn\Desktop\model\Qwen3.5-0.8B-Q4_0.gguf'
```

显存以 GPU 实际查询为准；Win32_VideoController.AdapterRAM 对大显存可能不准确。后面的 Python GPU 检查会记录真实可见显存。

### 2.2 备份并修改全局 WSL 配额

本次原配置为 4 GB / 2 CPU / 2 GB swap，备份是：
`C:\Users\nnnnnn\.wslconfig.before-lora-20261013-230659`。

下面命令适用于此机器原来只包含这些基础选项的配置；若以后文件增加了其他设置，应保留其他段落，只修改这三个值。

```powershell
$loraWslConfig = Join-Path $env:USERPROFILE '.wslconfig'
$loraBackup = "$loraWslConfig.before-lora-$(Get-Date -Format yyyyMMdd-HHmmss)"
if (Test-Path -LiteralPath $loraWslConfig) {
    Copy-Item -LiteralPath $loraWslConfig -Destination $loraBackup
}
@'
[wsl2]
memory=8GB
processors=6
swap=4GB
'@ | Set-Content -LiteralPath $loraWslConfig -Encoding ascii
wsl.exe --shutdown
wsl.exe --install -d Ubuntu-24.04 --no-launch --web-download
wsl.exe --list --verbose
```

`--shutdown` 会停止所有 WSL 发行版，保存其他 WSL 工作后再执行。配额作用于 WSL2 虚拟机，不是 Ubuntu-24.04 的独享物理内存。本次调整前其他发行版处于停止状态。若安装提示需要 Windows 重启，重启后继续，不要重复导入同名发行版。

### 2.3 首次创建 Linux 用户

```powershell
wsl.exe -d Ubuntu-24.04 -u root
```

进入 Ubuntu 的 root shell 后执行：

```bash
set -e
id nnnnnn >/dev/null 2>&1 || useradd --create-home --shell /bin/bash nnnnnn
for grp in sudo video render; do
    if getent group "$grp" >/dev/null; then usermod -a -G "$grp" nnnnnn; fi
done
install -d -o nnnnnn -g nnnnnn /home/nnnnnn/lora{,/logs,/cache,/downloads,/models,/outputs}
```

在 `/etc/wsl.conf` 保留已有配置并确保包含以下内容（本次保留了原来的 systemd 设置，并追加 user 段）：

```ini
[boot]
systemd=true

[user]
default=nnnnnn
```

新安装、原文件尚无 `[user]` 时，本次追加配置的命令相当于：

```bash
printf '\n[user]\ndefault=nnnnnn\n' >> /etc/wsl.conf
exit
```

已有 `[user]` 时应编辑其 default，避免重复追加。本次没有设置 Linux 密码，也没有创建免密 sudo 规则。系统维护直接使用 `wsl.exe -d Ubuntu-24.04 -u root`；普通工作用 `-u nnnnnn`。默认用户设置在该发行版下一次启动生效。

## 3. 恢复项目脚本

重装前保留本目录的 `.py`、`.sh`、`.md`、`requirements*.txt`、`downloads/SHA256SUMS` 以及需要保留的 `outputs/`。若重装整个发行版，也必须先把备份复制到发行版之外。

恢复这些文件到 `/home/nnnnnn/lora` 后，执行以下命令检查：

```powershell
wsl.exe -d Ubuntu-24.04 -u nnnnnn
```

```bash
cd /home/nnnnnn/lora
ls -la
chmod +x env.sh run.sh install_rocm.sh install_python.sh
bash -n env.sh run.sh install_rocm.sh install_python.sh
```

脚本职责：

| 文件 | 作用 |
|---|---|
| install_rocm.sh | 安装 Linux 系统依赖、ROCm 运行时和数学库、ROCDXG |
| install_python.sh | 建虚拟环境，下载并安装指定 AMD wheel 和锁定依赖 |
| env.sh | 激活虚拟环境，并设置 GPU 和缓存变量 |
| download_file.py | 支持断点记录与多连接 Range 下载 |
| prefetch_rocm.py | 可选：根据 APT 元数据并行下载数学库包并校验 SHA256 |
| download_model.py | 下载固定版本官方模型，并校验 Git blob / LFS SHA256 |
| smoke_test_lora.py | GPU 检查、LoRA 训练、保存、新模型重新加载及一致性检查 |
| run.sh | 训练脚本入口 |
| requirements-lock.txt | 实测环境的完整 Python 版本锁定 |
| INSTALL.md / TEST_RESULTS.md | 安装记录 / 实际测试结果 |

## 4. 安装 ROCm 与完整运行库

在 Windows PowerShell 以 WSL root 执行安装脚本。脚本的完整安装命令见本文附录 A。

```powershell
wsl.exe -d Ubuntu-24.04 -u root --exec bash /home/nnnnnn/lora/install_rocm.sh
```

这里只需要 WSL 用户态 ROCm 和 ROCDXG；本次没有安装 Linux amdgpu 内核 DKMS 驱动，也没有修改 Windows 驱动。

数学库约需下载 4.0 GB，解压后约 16.1 GB；加上 HIP/LLVM、Python 包和模型，总空间需留出余量。WSL 的 `df` 会显示虚拟磁盘上限，还要用 Windows 的 `Get-PSDrive C` 看物理盘实际剩余空间。

本次为了加速，先用脚本并行下载 APT 已解析好的包，然后仍交给 APT 校验、安装。普通重装直接执行 install_rocm.sh 即可；如果 APT 下载很慢，可在软件源配置完成后，以 root 执行：

```bash
cd /home/nnnnnn/lora
set -o pipefail
python3 -u prefetch_rocm.py 2>&1 | tee logs/rocm-libraries-download.log
apt-get install -y --no-install-recommends miopen-hip hipblas hipfft hiprand hipsparse hipsolver rocsolver hipsparselt rccl rocblas hipblaslt roctracer 2>&1 | tee logs/rocm-libraries-install.log
ldconfig
chown nnnnnn:nnnnnn logs/rocm-*
```

验证设备：

```bash
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo
ls -l /dev/dxg
```

本机已经在 rocminfo 中识别到 `AMD Radeon RX 7700 XT`、`gfx1101`。完整输出保存在 `logs/rocminfo.txt`。

## 5. 安装 Python 环境

用普通 Linux 用户执行，不能在 Windows Python 或其他 Ubuntu 的环境里执行：

```powershell
wsl.exe -d Ubuntu-24.04 -u nnnnnn
```

```bash
cd /home/nnnnnn/lora
set -o pipefail
./install_python.sh 2>&1 | tee logs/python-install.log
source env.sh
python --version
python -m pip check
```

完整命令见附录 B。AMD Torch 和 Triton 需要作为本地 wheel 一起安装。不要直接单独 `pip install torch`，也不要在安装 AMD wheel 之前仅安装 requirements.txt，让包管理器选择其他 Torch 构建。

`requirements-lock.txt` 如存在，安装脚本优先使用它；`requirements.txt` 只定义最初的直接依赖。安装包保存在 downloads，缓存保存在 cache。

## 6. 下载原始模型

```bash
cd /home/nnnnnn/lora
source env.sh
set -o pipefail
python -u download_model.py 2>&1 | tee logs/model-verification.log
```

下载路径：`models/Qwen3.5-0.8B`。固定文件清单、revision、大小与官方 LFS SHA256 保存在 `models/Qwen3.5-0.8B/source.json`。

权重文件：

```text
model.safetensors-00001-of-00001.safetensors
size: 1746942600 bytes
SHA256: 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
```

本次已通过这个 SHA256。下载脚本可重新运行以核对已有文件。Range 下载的中途文件可能已经预分配到完整大小，应以 `.download.json` 和哈希校验结果判断完成状态，不能仅看文件大小。保留状态文件即可断点续传。

## 7. 运行 GPU、训练、保存与重新加载测试

在普通用户 WSL shell：

```bash
cd /home/nnnnnn/lora
source env.sh
set -o pipefail
./run.sh --check-only --output-dir outputs/gpu-check 2>&1 | tee logs/gpu-check.log
./run.sh --steps 20 --output-dir outputs/smoke-test 2>&1 | tee logs/smoke-test.log
cat outputs/smoke-test/report.json
```

`outputs/smoke-test` 是本次验证目录；里面存在 adapter 时脚本拒绝覆盖。日常再次测试直接执行下面命令，会自动创建时间戳目录：

```bash
cd /home/nnnnnn/lora
./run.sh --steps 20
```

Windows 一行启动：

```powershell
wsl.exe -d Ubuntu-24.04 -u nnnnnn --exec bash /home/nnnnnn/lora/run.sh --steps 20
```

测试设置：FP32 普通 LoRA，rank=8、alpha=16、dropout=0、学习率 2e-4；micro batch=1、梯度累积=4、20 个 optimizer steps，16 条内置演示问答；只对 assistant 回答计算 loss。训练原始权重被冻结，只有 LoRA 可训练。使用纯文本模型类，不训练视觉部分。

脚本逐项检查：

1. 显卡为 RX 7700 XT，GPU 矩阵乘法与 CPU 结果一致，反向传播有效。
2. 基础文本模型的必需权重完整加载。
3. loss 和梯度有限，梯度非零，LoRA 参数实际更新。
4. 保存 adapter_model.safetensors、adapter_config.json、tokenizer 和测试报告。
5. 释放旧模型及 optimizer，重新加载全新的基础模型，再加载刚才保存的 adapter。
6. 比较重载前后 logits（容差 1e-4），并要求贪心生成 token 完全一致。

这属于流程冒烟测试，16 条演示数据没有独立验证集；loss 下降和背下演示答案不能代表真实任务的泛化质量。

## 8. 本次排查记录与修复

### ROCm 软件源版本优先级

首次 `apt-get install -y --no-install-recommends rocm-hip-runtime` 依赖解析失败。Ubuntu 自带 rocminfo 的版本号高于 AMD 仓库对应包，APT 选错来源。添加以下优先级后成功：

```bash
printf '%s\n' 'Package: *' 'Pin: origin repo.radeon.com' 'Pin-Priority: 600' > /etc/apt/preferences.d/rocm-radeon-pin-600
apt-cache policy rocminfo rocm-hip-runtime
apt-get install -y --no-install-recommends rocm-hip-runtime
```

这个修复已提前放进安装脚本。

### Torch 缺少配套 Triton

第一次仅把 Torch wheel 与 requirements.txt 一起传给 `uv pip install`，解析提示缺少：
`triton==3.5.1+rocm7.2.1.gita272dfa8`。
从 AMD 同一目录下载匹配的 Triton wheel，与 Torch wheel 一起安装后通过 `pip check`。精确下载地址见附录 B。

### Torch 导入缺少 libroctx64.so.4 等系统库

首次 GPU 检查报 `ImportError: libroctx64.so.4`。用以下命令检查后发现还缺少 MIOpen、rocBLAS、hipBLAS、RCCL 等依赖：

```bash
cd /home/nnnnnn/lora
source env.sh
ldd .venv/lib/python3.12/site-packages/torch/lib/libtorch_hip.so
apt-cache search roctx
apt-cache show roctracer rocm-ml-libraries rocm-libs
apt-get --simulate --no-install-recommends install miopen-hip hipblas hipfft hiprand hipsparse hipsolver rocsolver hipsparselt rccl rocblas hipblaslt roctracer
apt-get --print-uris -y --no-install-recommends install miopen-hip hipblas hipfft hiprand hipsparse hipsolver rocsolver hipsparselt rccl rocblas hipblaslt roctracer
```

AMD 的此版 `lw` wheel 需要系统库；不能把 `rocm-hip-runtime` 当作完整 PyTorch 依赖。完整安装列表已加入附录 A，后续应按修正后的脚本安装。

### 下载慢与 WSL 安装等待

WSL 安装命令起初长时间无新输出，但最后正常完成。等待过程中曾启动备用官方 Ubuntu 镜像下载；发现原安装成功后就停止并删除了该临时下载，没有执行 wsl --import，也没有创建额外发行版。

Torch 和模型的初始单连接下载较慢，因此停止了这两个下载进程，保留已有部分文件，再用 download_file.py 的 8 个 Range 连接继续。最后模型通过官方 SHA256。重装直接用最终脚本即可，不需要重演中断过程。

## 9. 环境记录与备份

测试通过后使用以下命令留下版本和依赖检查结果：

```bash
cd /home/nnnnnn/lora
source env.sh
python -m pip list --format=freeze > requirements-lock.txt
python -m pip check
python -m torch.utils.collect_env > logs/torch-environment.txt 2>&1
dpkg-query -W > logs/dpkg-versions.txt
uname -a > logs/kernel.txt
sha256sum downloads/*.whl downloads/*.deb > downloads/SHA256SUMS
```

本项目不需要把基础模型合并进 adapter；推理时保留 models 和 outputs/对应训练目录/adapter 即可。若以后移动基础模型目录，在加载脚本中显式传入新的基础模型路径。

## 10. 官方参考

- [AMD WSL 安装说明](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)
- [AMD PyTorch wheel 安装说明](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
- [AMD WSL 兼容表](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
- [ROCDXG 1.2.2](https://github.com/ROCm/librocdxg/releases/tag/v1.2.2)
- [Qwen 官方模型](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)
- [Microsoft WSL 文件系统说明](https://learn.microsoft.com/en-us/windows/wsl/filesystems)

以下固定的是本次实装版本。未来重装若要升级，应重新核对相应版本的兼容矩阵。

## 附录 A：完整 Linux 系统安装命令

以下即 install_rocm.sh 内容，在 Ubuntu-24.04 的 root shell 执行：

```bash
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
```

## 附录 B：完整 Python 安装命令

以下即 install_python.sh 内容，在 nnnnnn 用户下执行：

```bash
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
```

## 附录 C：训练结束后单独加载 adapter 问答

`infer_lora.py` 会重新读取原始模型和保存的 adapter，执行一次贪心生成：

```bash
cd /home/nnnnnn/lora
source env.sh
python infer_lora.py --adapter outputs/smoke-test/adapter --prompt '项目青禾的用途是什么？'
```

提示词可以替换；测试模型仅用少量内置样本训练，所以先使用 demo_data.json 中的问答验证保存结果。推理仍需保留基础模型。

Transformers 5 的 apply_chat_template 默认返回字典；脚本已显式指定 return_dict=False，以取得 token 列表或 Tensor，并正确构造 assistant-only 标签。

## 附录 D：本次最后执行的验证与日志归档

训练通过后，在 Windows PowerShell 停止并重新启动本次新装的发行版，以验证默认用户及环境持久化。执行 terminate 前训练已经退出；不要在训练过程中执行：

```powershell
wsl.exe --terminate Ubuntu-24.04
wsl.exe -d Ubuntu-24.04 --exec whoami
```

返回 `nnnnnn`。随后在新启动的 WSL 中执行：

```bash
cd /home/nnnnnn/lora
source env.sh
set -o pipefail
id -un > logs/default-user-after-restart.txt
python -u infer_lora.py --adapter outputs/smoke-test/adapter --prompt '项目青禾的用途是什么？' 2>&1 | tee logs/restart-inference.log
```

返回“离线文档检索”，已自动比对预期答案并记录到 `logs/restart-inference.json`。

另外执行过的依赖、语法和模型校验命令：

```bash
cd /home/nnnnnn/lora
source env.sh
uv pip install --dry-run 'downloads/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' 'downloads/triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl' -r requirements-lock.txt
bash -n env.sh run.sh install_rocm.sh install_python.sh
python download_model.py
python - <<'PY'
import ast
from pathlib import Path
from transformers import AutoTokenizer
from smoke_test_lora import make_data
for path in Path(".").glob("*.py"):
    ast.parse(path.read_text())
tokenizer = AutoTokenizer.from_pretrained("models/Qwen3.5-0.8B")
rows = make_data(tokenizer, 256)
print(len(rows), min(len(r["input_ids"]) for r in rows), max(len(r["input_ids"]) for r in rows))
PY
```

依赖检查通过，数据检查输出 `16 43 44`。这些短样本没有填充到 256 tokens；256 是脚本允许的长度上限。

APT 原始安装记录也已归档。以下命令在 WSL root 下执行：

```bash
cp /var/log/apt/history.log /home/nnnnnn/lora/logs/apt-history.txt
cp /var/log/apt/term.log /home/nnnnnn/lora/logs/apt-terminal.txt
cp /etc/wsl.conf /home/nnnnnn/lora/logs/wsl.conf
chown nnnnnn:nnnnnn /home/nnnnnn/lora/logs/apt-history.txt /home/nnnnnn/lora/logs/apt-terminal.txt /home/nnnnnn/lora/logs/wsl.conf
```

最初创建虚拟环境时执行的是 `python -m pip install --upgrade pip uv`，实际获得 pip 26.2.1、uv 0.13.0；附录 B 已固定为这两个版本，供重装复现。

Torch wheel 的发行包版本包含 `.lw`，但运行时 `torch.__version__` 返回 `2.9.1+rocm7.2.1.gitff65f5bc`；这是同一个已安装 wheel 的两种版本表示。

### env.sh 完整内容

`run.sh` 自动加载以下环境。手动运行 Python 前也应 `source env.sh`：


```bash
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
```

## 2026-10-16 项目清理与 Git 初始化

已清理安装完成后的 Torch / Triton wheel、pip / uv 缓存、项目顶层 Python 字节码和旧缺库报错日志。downloads/SHA256SUMS 保留；install_python.sh 会在重新安装时先下载对应 wheel，再核对校验清单。ROCDXG deb 仍保留，完整重装时 install_rocm.sh 也会重新下载它。

全部源码脚本仍用于运行或安装复现，均已保留。详见 CLEANUP.md，其中记录了本次清理、Git 初始化以及后续验证的命令。

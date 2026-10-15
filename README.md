# Qwen3.5 LoRA Demo

RX 7700 XT / WSL2 Ubuntu 24.04。uv 管理 Python 版本、依赖和 `.venv`。

## 首次安装

项目放在 `/home/nnnnnn/lora`。Windows PowerShell 安装系统依赖：

```powershell
wsl -d Ubuntu-24.04 -u root --exec bash /home/nnnnnn/lora/install_rocm.sh
```

WSL 普通用户安装 Python 依赖并下载模型：

```bash
bash /home/nnnnnn/lora/install_python.sh
```

## 运行

```bash
cd /home/nnnnnn/lora
source env.sh
uv run lora_demo.py
```

demo 每次从原始模型开始，依次训练、保存 LoRA、重新加载并回答一个问题。参数和示例数据直接修改 `lora_demo.py` 顶部常量。原始模型在 `models/Qwen3.5-0.8B`；每次运行覆盖 `outputs/adapter` 中的演示结果。

## 文件

```text
lora_demo.py        # 最小训练、保存和重载问答演示
pyproject.toml     # 项目依赖与 AMD wheel 来源
uv.lock            # 依赖版本和校验值
.python-version    # Python 版本
install_rocm.sh    # ROCm 系统库
install_python.sh  # uv 环境同步与模型下载
env.sh             # uv 路径、显卡和缓存配置
```

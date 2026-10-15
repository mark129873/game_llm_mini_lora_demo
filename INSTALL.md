# 安装

环境：Windows 11、RX 7700 XT、WSL2 Ubuntu 24.04、Python 3.12。以下使用 Linux 用户 `nnnnnn`，项目放在 `/home/nnnnnn/lora`。

## 1. WSL

Windows PowerShell：

```powershell
wsl --install -d Ubuntu-24.04
```

完成 Ubuntu 用户初始化。Windows 的 `%USERPROFILE%\.wslconfig` 中设置：

```ini
[wsl2]
memory=8GB
processors=6
swap=4GB
```

## 2. 系统依赖

项目文件放好后，在 PowerShell 执行（`--shutdown` 会停止所有 WSL 任务）：

```powershell
wsl --shutdown
wsl -d Ubuntu-24.04 -u root --exec bash /home/nnnnnn/lora/install_rocm.sh
```

## 3. Python 与模型

```powershell
wsl -d Ubuntu-24.04 -u nnnnnn --exec bash /home/nnnnnn/lora/install_python.sh
```

脚本创建 `.venv`，安装固定版本依赖并下载模型。ROCm 7.2.1、ROCDXG 1.2.2；详细配置见脚本注释。运行方式见 [README.md](README.md)。

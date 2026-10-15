# 本机 Qwen3.5 LoRA 测试

项目位于 `/home/nnnnnn/lora`。Ubuntu 24.04 + ROCm + RX 7700 XT 已实际跑通训练、保存和重载。

- [完整安装命令、版本及排错记录](INSTALL.md)
- [实测结果与资源占用](TEST_RESULTS.md)
- [原始训练报告](outputs/smoke-test/report.json)

## 重新跑一遍

在 Ubuntu-24.04 终端执行：

```bash
cd /home/nnnnnn/lora
./run.sh --steps 20
```

每次默认生成独立的时间戳输出目录。首次通过的结果保存在 `outputs/smoke-test`。

也可直接在 Windows PowerShell 执行：

```powershell
wsl.exe -d Ubuntu-24.04 -u nnnnnn --exec bash /home/nnnnnn/lora/run.sh --steps 20
```

## 只加载已保存的 LoRA 测试问答

```bash
cd /home/nnnnnn/lora
source env.sh
python infer_lora.py --adapter outputs/smoke-test/adapter --prompt '项目青禾的用途是什么？'
```

实测回答：离线文档检索。

训练使用官方原始 safetensors，产物是 adapter。桌面原有的 Q4_0 GGUF 保留原样。演示数据在 `smoke_test_lora.py` 的 `make_data()` 中，16 条问答用于验证流程；替换为实际任务数据后需要单独评估效果。

重装时保留本目录的脚本、Markdown、requirements 文件及需要的模型与训练结果，按 INSTALL.md 执行。

## Git 与清理记录

本目录已于 2026-10-16 初始化为本地 Git 仓库，分支为 main。源码、文档、依赖清单和安装包校验清单纳入版本管理；虚拟环境、缓存、下载包、模型、训练产物与机器日志由 .gitignore 排除。

[本次清理详情与执行命令](CLEANUP.md)。完整模型和 adapter 需要另行备份；Git 保存复现所需的代码和说明。

```bash
cd /home/nnnnnn/lora
git status
```

WSL 和 Windows 均尚未配置 Git 用户名、邮箱，因此本次只初始化仓库并暂存文件。首次提交前，请把下面的署名替换为自己的信息：

```bash
git config --local user.name "你的名字"
git config --local user.email "你的邮箱"
git commit -m "chore: initialize LoRA project"
```

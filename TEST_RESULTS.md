# 本机 LoRA 实测结果

测试日期：2026-10-13。结论：**通过**。Ubuntu 24.04、RX 7700 XT 已完成真实 GPU 训练、保存、重新加载；WSL 重启后的独立推理也通过。

## 训练与资源

| 项目 | 实测值 |
|---|---|
| GPU | AMD Radeon RX 7700 XT / gfx1101；可见显存 11.8 GiB |
| 训练步数 / 数据 | 20 optimizer steps；16 条，43～44 tokens/条 |
| 精度 / LoRA | FP32；rank 8；alpha 16；5,411,328 个可训练参数 |
| Batch / 梯度累积 | 1 / 4 |
| 训练循环及参数更新检查耗时 | 55.068 秒 |
| PyTorch 峰值 allocated 显存 | 3.073 GiB |
| PyTorch 峰值 reserved 显存 | 3.115 GiB |
| 第一 / 最后一步平均训练 loss | 5.080179 → 0.079799 |
| 同一训练样本的前后 probe loss | 4.226623 → 0.038145 |
| Adapter 权重大小 | 21,694,976 bytes（约 20.69 MiB） |

耗时不包含基础模型加载、训练前后生成、保存和重载。显存数字来自 PyTorch allocator，不是 Windows 任务管理器的全进程显存峰值。本次短样本并未 padding 到 256；这些数字不能直接代表长上下文、大 batch 或真实数据集的消耗。

## 保存和重载验证

- 只有 LoRA 参数参与训练，参数实际发生更新：True。
- 释放旧模型和 optimizer，重新读取原始基础模型及 adapter；logits 最大绝对差：0.0。
- 三组贪心生成的 token 完全相同：True。
- 终止并重新启动 Ubuntu-24.04 后，默认用户：nnnnnn。
- 新进程独立加载 adapter 回答：离线文档检索。

## 生成结果

| 提问 | 训练后及重载后的回答 |
|---|---|
| 项目青禾的用途是什么？ | 离线文档检索 |
| 项目星桥的用途是什么？ | 会议纪要整理 |
| 项目海棠的用途是什么？ | 设备状态巡检 |

这些是训练集中的演示问题，没有独立验证集；此次验证了训练与保存流程，不代表实际任务准确率。

## 实际环境和产物

- 基础模型：models/Qwen3.5-0.8B；revision 为 2fc06364715b967f1860aea9cf38778875588b17。
- 权重来自 Qwen 官方 safetensors，已校验 SHA256。桌面原 GGUF 未改动，未执行 GGUF 导出。
- LoRA：outputs/smoke-test/adapter/adapter_model.safetensors。
- 配置：outputs/smoke-test/adapter/adapter_config.json。
- 原始机器可读报告：outputs/smoke-test/report.json。
- 训练日志：logs/smoke-test.log。
- 重启后推理：logs/restart-inference.log、logs/restart-inference.json。
- 完整版本：requirements-lock.txt、logs/torch-environment.txt、logs/dpkg-versions.txt。
- 安装命令、失败原因及修复：INSTALL.md。

当前使用 PyTorch 参考实现的 causal convolution 和 gated delta rule；相关优化内核未安装，日志中的 fallback 提示在这次测试中没有导致错误。未来若追求更长序列或训练吞吐，再单独验证适用于这张 AMD 显卡的优化实现。

项目目录约 8.0 GiB，另有 WSL 中的 ROCm 系统库与 APT 缓存。完成安装时 Windows C 盘剩余约 46.9 GiB。

# Qwen3.5 LoRA Demo

RX 7700 XT / WSL2 Ubuntu 24.04。首次配置见 [INSTALL.md](INSTALL.md)。

[文件树及各文件用途](tree.md)。

## 训练、保存并验证重载

在 WSL 中执行：

```bash
cd /home/nnnnnn/lora
source env.sh
python lora_demo.py
```

默认用 16 条内置问答训练 20 步，FP32、LoRA rank=8；结果保存到 `outputs/adapter`。终端显示训练前后回答，并检查重新加载后的 logits 和生成 token 是否一致。

修改训练步数或输出目录：

```bash
python lora_demo.py --steps 20 --output-dir outputs/another-adapter
```

## 加载已保存的 LoRA 问答

```bash
python lora_demo.py --adapter outputs/adapter --prompt '项目青禾的用途是什么？'
```

演示数据在 `lora_demo.py` 的 `make_data()` 中；基础模型位于 `models/Qwen3.5-0.8B`。输出目录已有 adapter 时，请指定其他 `--output-dir`。

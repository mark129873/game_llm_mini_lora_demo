#!/usr/bin/env python3
"""Qwen3.5 LoRA 演示：训练、保存 adapter、重新加载并验证生成结果。"""

import argparse
import gc
from pathlib import Path
import random
import time

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = "你是内部项目查询助手。只回答项目用途，不添加解释。"
GRADIENT_ACCUMULATION = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-0.8B",
                        help="本地原始模型目录")
    parser.add_argument("--steps", type=int, default=20, help="优化器更新次数")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/adapter",
                        help="训练后保存 adapter 和 tokenizer 的目录")
    parser.add_argument("--adapter", type=Path, help="仅加载这个 adapter 进行问答，跳过训练")
    parser.add_argument("--prompt", default="项目青禾的用途是什么？", help="仅推理时使用的提问")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps 必须大于 0")
    if not (args.model / "config.json").is_file():
        parser.error("找不到本地模型，请先运行 bash install_python.sh")
    if args.adapter:
        if not (args.adapter / "adapter_config.json").is_file():
            parser.error("--adapter 必须指向已保存的 LoRA 目录")
    elif (args.output_dir / "adapter_config.json").exists():
        parser.error("输出目录已有 adapter，请使用 --output-dir 指定其他目录")
    return args


def select_device():
    """ROCm 也使用 torch.cuda 接口；选中本机的 7700 XT 并检查真实 GPU 计算。"""
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU 不可用，请先 source env.sh 并检查 rocminfo")
    matches = [i for i in range(torch.cuda.device_count())
               if "7700 XT" in torch.cuda.get_device_name(i)]
    if not matches:
        raise RuntimeError("没有找到 RX 7700 XT")
    device = torch.device("cuda", matches[0])
    torch.cuda.set_device(device)

    # 对照 CPU 计算结果，再验证反向传播，避免只检测到设备却无法执行计算。
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    gpu_a = a.to(device).requires_grad_(True)
    result = gpu_a @ b.to(device)
    torch.testing.assert_close(result.detach().cpu(), a @ b, rtol=1e-4, atol=1e-4)
    result.square().mean().backward()
    if not torch.isfinite(gpu_a.grad).all() or gpu_a.grad.norm().item() == 0:
        raise RuntimeError("GPU 反向传播检查失败")
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    return device


def load_model(model_dir, device):
    """只加载 Qwen3.5 的文本模型，FP32 配合标准 PyTorch 算子运行。"""
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager",
        device_map={"": device.index}, local_files_only=True, output_loading_info=True,
    )
    # 多模态检查点中的视觉权重由文本模型忽略；必需的文本权重不能缺失。
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"基础模型权重加载不完整：{info}")
    model.config.use_cache = False  # 训练时关闭 KV cache，节省显存。
    return model


def messages(prompt):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]


def make_data(tokenizer):
    """8 个虚构项目，各用两种问法，构成 16 条便于观察变化的演示数据。"""
    facts = [
        ("青禾", "离线文档检索"), ("星桥", "会议纪要整理"),
        ("海棠", "设备状态巡检"), ("白鹭", "图片标签管理"),
        ("云杉", "库存数量核对"), ("远帆", "客户反馈分类"),
        ("石竹", "日报格式检查"), ("晨露", "本地日志分析"),
    ]
    data = []
    for name, answer in facts:
        for prompt in (f"项目{name}的用途是什么？", f"请查询{name}项目的用途。"):
            conversation = messages(prompt)
            prefix = tokenizer.apply_chat_template(
                conversation, tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            full = tokenizer.apply_chat_template(
                conversation + [{"role": "assistant", "content": answer}],
                tokenize=True, return_dict=False,
                add_generation_prompt=False, enable_thinking=False,
            )
            # system/user 和思考标记不参与 loss，只学习 assistant 的回答及结束符。
            # 显式指定 return_dict=False，兼容 Transformers 5 的默认返回类型。
            if full[:len(prefix)] != prefix or len(full) <= len(prefix):
                raise RuntimeError("聊天模板前缀不匹配，无法构造正确的训练标签")
            data.append({"prompt": prompt, "expected": answer, "input_ids": full,
                         "labels": [-100] * len(prefix) + full[len(prefix):]})
    return data


def batch(row, device):
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def predict(model, tokenizer, prompt, device):
    """固定为贪心生成，方便逐 token 比较保存前后的结果。"""
    ids = tokenizer.apply_chat_template(
        messages(prompt), tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            do_sample=False, max_new_tokens=32, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    tokens = output[0, ids.shape[1]:].cpu().tolist()
    return tokens, tokenizer.decode(tokens, skip_special_tokens=True).strip()


def train_demo(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = make_data(tokenizer)
    probes = [data[0], data[2], data[4]]
    model = load_model(args.model, device)
    model.eval()
    before = [predict(model, tokenizer, row["prompt"], device)[1] for row in probes]

    # PEFT 冻结基础模型，只给文本模型的线性层添加可训练的低秩矩阵。
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        lora_dropout=0.0, target_modules="all-linear", bias="none",
    ))
    # 用重算中间激活换取更低显存；非重入方式可配合冻结的基础模型。
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("训练参数范围不正确，应只有 LoRA 参数")
    initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=2e-4, weight_decay=0)
    order = list(range(len(data)))
    random.Random(42).shuffle(order)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    print(f"训练 {args.steps} 步，可训练参数 {sum(p.numel() for p in trainable.values()):,}",
          flush=True)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        # 每次只处理一条样本，累计 4 次梯度后更新一次，等效 batch size 为 4。
        for micro in range(GRADIENT_ACCUMULATION):
            index = order[(step * GRADIENT_ACCUMULATION + micro) % len(order)]
            loss = model(**batch(data[index], device)).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"第 {step + 1} 步 loss 非有限值")
            step_loss += loss.item() / GRADIENT_ACCUMULATION
            (loss / GRADIENT_ACCUMULATION).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            list(trainable.values()), 1.0, error_if_nonfinite=True)
        if norm.item() == 0:
            raise RuntimeError("LoRA 梯度为零")
        optimizer.step()
        torch.cuda.synchronize(device)
        print(f"step {step + 1:02d}/{args.steps} loss={step_loss:.6f}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    if not any(not torch.equal(initial[name], p.detach().cpu())
               for name, p in trainable.items()):
        raise RuntimeError("LoRA 参数没有发生更新")
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"训练耗时 {elapsed:.1f} 秒，PyTorch 峰值显存 {peak:.2f} GiB", flush=True)

    # 保存前先记录同一输入的 logits 与贪心生成结果，供重载后对照。
    model.eval()
    with torch.inference_mode():
        reference = model(**batch(data[0], device)).logits[:, -1].float().cpu()
    after = [predict(model, tokenizer, row["prompt"], device) for row in probes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # 必须释放旧模型和优化器，再创建新的基础模型，才能验证磁盘中的 adapter。
    del model, optimizer, trainable, initial, loss
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(
        load_model(args.model, device), args.output_dir, is_trainable=False)
    reloaded.eval()
    with torch.inference_mode():
        restored = reloaded(**batch(data[0], device)).logits[:, -1].float().cpu()
    torch.testing.assert_close(restored, reference, rtol=1e-4, atol=1e-4)
    for row, old_answer, trained_answer in zip(probes, before, after):
        restored_answer = predict(reloaded, tokenizer, row["prompt"], device)
        if restored_answer[0] != trained_answer[0]:
            raise RuntimeError("重载后的生成 token 与保存前不一致")
        print(f"\n问题：{row['prompt']}\n训练前：{old_answer}\n训练后：{restored_answer[1]}")
    difference = (restored - reference).abs().max().item()
    print(f"\n通过：训练、保存和重载验证；logits 最大差值 {difference:.6g}")
    print(f"LoRA 保存于：{args.output_dir.resolve()}")


def main():
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = select_device()
    if args.adapter:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
        model = PeftModel.from_pretrained(
            load_model(args.model, device), args.adapter, is_trainable=False)
        model.eval()
        print(predict(model, tokenizer, args.prompt, device)[1])
    else:
        train_demo(args, device)


if __name__ == "__main__":
    main()

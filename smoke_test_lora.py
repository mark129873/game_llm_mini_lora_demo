#!/usr/bin/env python3
"""A small, GPU-only Qwen3.5 LoRA training/save/reload smoke test."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Test the AMD GPU without loading a model.")
    parser.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    parser.add_argument("--revision", default="2fc06364715b967f1860aea9cf38778875588b17")
    parser.add_argument("--steps", type=int, default=20, help="Optimizer steps.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.max_length < 32:
        parser.error("--steps must be positive and --max-length must be at least 32")
    args.output_dir = args.output_dir or ROOT / "outputs" / time.strftime("%Y%m%d-%H%M%S")
    return args

def save_report(path, report):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

def gpu_check(torch):
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("A working ROCm GPU is required. No CPU fallback. Check env.sh and rocminfo.")
    devices = [
        {"index": i, "name": torch.cuda.get_device_name(i)}
        for i in range(torch.cuda.device_count())
    ]
    matches = [d for d in devices if "7700 XT" in d["name"]]
    if not matches:
        raise RuntimeError(f"RX 7700 XT was not found: {devices}")
    device = torch.device("cuda", matches[0]["index"])
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    a_cpu = torch.randn(64, 64, dtype=torch.float32)
    b_cpu = torch.randn(64, 64, dtype=torch.float32)
    a = a_cpu.to(device).requires_grad_(True)
    b = b_cpu.to(device)
    result = a @ b
    torch.testing.assert_close(result.detach().cpu(), a_cpu @ b_cpu, rtol=1e-4, atol=1e-4)
    loss = result.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(a.grad).all() or a.grad.norm().item() == 0:
        raise RuntimeError("GPU backward check failed.")
    info = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "devices": devices, "selected_device": str(device), "name": props.name,
        "architecture": getattr(props, "gcnArchName", ""),
        "vram_gib": round(props.total_memory / 2**30, 2),
        "matmul_matches_cpu": True, "backward_passed": True,
    }
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    return device, info

def load_base(torch, model_cls, args, device):
    kwargs = {"dtype": torch.float32, "attn_implementation": "eager",
              "device_map": {"": device.index}, "output_loading_info": True}
    if not Path(args.model).is_dir():
        kwargs["revision"] = args.revision
    model, info = model_cls.from_pretrained(args.model, **kwargs)
    missing = info.get("missing_keys", [])
    mismatched = info.get("mismatched_keys", [])
    if missing or mismatched:
        raise RuntimeError(f"Checkpoint did not fully load: missing={missing}, mismatched={mismatched}")
    model.config.use_cache = False
    return model

def make_data(tokenizer, max_length):
    facts = [
        ("青禾", "离线文档检索"),
        ("星桥", "会议纪要整理"),
        ("海棠", "设备状态巡检"),
        ("白鹭", "图片标签管理"),
        ("云杉", "库存数量核对"),
        ("远帆", "客户反馈分类"),
        ("石竹", "日报格式检查"),
        ("晨露", "本地日志分析"),
    ]
    rows = []
    for name, answer in facts:
        for prompt in [f"项目{name}的用途是什么？", f"请查询{name}项目的用途。"]:
            messages = [{"role": "system", "content": "你是内部项目查询助手。只回答项目用途，不添加解释。"},
                        {"role": "user", "content": prompt}]
            prefix = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                       add_generation_prompt=True, enable_thinking=False)
            full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                     tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False)
            if full[:len(prefix)] != prefix:
                raise RuntimeError("Chat template prefix mismatch; assistant-only loss mask would be wrong.")
            if len(full) > max_length:
                raise RuntimeError(f"Demo needs {len(full)} tokens; increase --max-length.")
            rows.append({"messages": messages, "expected": answer, "input_ids": full,
                         "labels": [-100] * len(prefix) + full[len(prefix):]})
    return rows

def batch(torch, row, device):
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}

def generate(torch, model, tokenizer, messages, device):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                                       enable_thinking=False, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             do_sample=False, max_new_tokens=32, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    tokens = out[0, ids.shape[1]:].cpu().tolist()
    return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "token_ids": tokens}

def main(args, report):
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device, report["gpu"] = gpu_check(torch)
    report["versions"] = {"transformers": transformers.__version__, "peft": peft.__version__}
    if args.check_only:
        report["status"] = "passed"
        return
    if (args.output_dir / "adapter").exists():
        raise RuntimeError("Output already contains an adapter. Select a new --output-dir.")

    tok_kwargs = {} if Path(args.model).is_dir() else {"revision": args.revision}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = make_data(tokenizer, args.max_length)
    (args.output_dir / "demo_data.json").write_text(
        json.dumps([{"messages": r["messages"], "expected": r["expected"]} for r in data],
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Loading the original text model in FP32...", flush=True)
    model = load_base(torch, Qwen3_5ForCausalLM, args, device)
    model.eval()
    probes = [data[0], data[2], data[4]]
    before = [generate(torch, model, tokenizer, row["messages"], device) for row in probes]
    with torch.inference_mode():
        initial_loss = model(**batch(torch, data[0], device)).loss.item()

    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                        lora_dropout=0.0, target_modules="all-linear", bias="none")
    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n for n in trainable):
        raise RuntimeError("Only LoRA parameters should be trainable.")
    initial_adapter = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    report["training"] = {
        "steps": args.steps, "seed": 42, "dtype": "float32", "rank": 8, "alpha": 16,
        "learning_rate": 2e-4, "batch_size": 1, "gradient_accumulation": 4,
        "max_length": args.max_length, "examples": len(data),
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "initial_probe_loss": initial_loss,
    }
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=2e-4, weight_decay=0)
    rng = random.Random(42)
    order = list(range(len(data)))
    rng.shuffle(order)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = []
    model.train()
    print(f"Training {args.steps} steps; {report['training']['trainable_parameters']:,} trainable parameters", flush=True)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(4):
            index = order[(step * 4 + micro) % len(order)]
            loss = model(**batch(torch, data[index], device)).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step + 1}.")
            step_loss += loss.item() / 4
            (loss / 4).backward()
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
        if norm.item() == 0:
            raise RuntimeError(f"Zero LoRA gradients at step {step + 1}.")
        optimizer.step()
        torch.cuda.synchronize(device)
        losses.append(step_loss)
        print(f"step {step + 1:02d}/{args.steps} loss={step_loss:.6f} grad_norm={norm.item():.5f}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    changed = any(not torch.equal(initial_adapter[n], p.detach().cpu()) for n, p in trainable.items())
    if not changed:
        raise RuntimeError("No LoRA parameter was updated.")
    report["training"].update(
        seconds=round(time.perf_counter() - started, 3), losses=losses,
        adapter_parameters_changed=changed,
        peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
    )
    model.eval()
    with torch.inference_mode():
        final_loss = model(**batch(torch, data[0], device)).loss.item()
        reference_logits = model(**batch(torch, data[0], device)).logits[:, -1].float().cpu()
    after = [generate(torch, model, tokenizer, row["messages"], device) for row in probes]
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    report["training"]["final_probe_loss"] = final_loss
    report["training"]["probe_loss_decreased"] = final_loss < initial_loss
    report["comparisons"] = [
        {"prompt": row["messages"][-1]["content"], "expected": row["expected"],
         "before": b["text"], "after": a["text"]}
        for row, b, a in zip(probes, before, after)
    ]
    print("Adapter saved. Releasing model and optimizer before a fresh load...", flush=True)
    del optimizer, model, trainable, initial_adapter, loss
    gc.collect()
    torch.cuda.empty_cache()
    base = load_base(torch, Qwen3_5ForCausalLM, args, device)
    reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    reloaded.eval()
    with torch.inference_mode():
        reloaded_logits = reloaded(**batch(torch, data[0], device)).logits[:, -1].float().cpu()
    max_diff = (reference_logits - reloaded_logits).abs().max().item()
    torch.testing.assert_close(reloaded_logits, reference_logits, rtol=1e-4, atol=1e-4)
    reloaded_answers = [generate(torch, reloaded, tokenizer, row["messages"], device) for row in probes]
    token_match = all(a["token_ids"] == b["token_ids"] for a, b in zip(after, reloaded_answers))
    if not token_match:
        raise RuntimeError("Greedy generated tokens changed after reloading.")
    report["reload"] = {"passed": True, "max_abs_logit_difference": max_diff,
                        "greedy_tokens_identical": token_match, "adapter_dir": str(adapter_dir)}
    report["status"] = "passed"
    for comparison in report["comparisons"]:
        print(json.dumps(comparison, ensure_ascii=False), flush=True)
    print(f"PASS: training, adapter save, and fresh reload. Report: {args.output_dir / 'report.json'}", flush=True)

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report = {"status": "running", "model": args.model, "revision": args.revision}
    save_report(report_path, report)
    try:
        main(args, report)
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)
    finally:
        save_report(report_path, report)

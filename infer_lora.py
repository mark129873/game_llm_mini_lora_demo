#!/usr/bin/env python3
"""Reload a saved adapter and answer one prompt on the RX 7700 XT."""
import argparse
from pathlib import Path
from smoke_test_lora import ROOT, gpu_check, load_base, generate
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from peft import PeftModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--adapter", type=Path, required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
parser.add_argument("--revision", default="2fc06364715b967f1860aea9cf38778875588b17")
args = parser.parse_args()
if not (args.adapter / "adapter_config.json").is_file():
    parser.error("--adapter must point to the saved adapter directory")
torch.manual_seed(42)
torch.set_num_threads(4)
device, _ = gpu_check(torch)
tokenizer = AutoTokenizer.from_pretrained(args.adapter)
base = load_base(torch, Qwen3_5ForCausalLM, args, device)
model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
model.eval()
messages = [
    {"role": "system", "content": "\u4f60\u662f\u5185\u90e8\u9879\u76ee\u67e5\u8be2\u52a9\u624b\u3002\u53ea\u56de\u7b54\u9879\u76ee\u7528\u9014\uff0c\u4e0d\u6dfb\u52a0\u89e3\u91ca\u3002"},
    {"role": "user", "content": args.prompt},
]
print(generate(torch, model, tokenizer, messages, device)["text"])

"""最小 LoRA 演示：训练 → 保存 → 重新加载 → 回答一个问题。"""

import gc
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

# 修改这里的路径、训练参数和示例数据即可。
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models/Qwen3.5-0.8B"
OUTPUT_DIR = ROOT / "outputs/adapter"
TRAIN_STEPS = 20
LEARNING_RATE = 2e-4
LORA_RANK = 8
EXAMPLES = [
    ("项目青禾的用途是什么？", "离线文档检索"),
    ("项目星桥的用途是什么？", "会议纪要整理"),
    ("项目海棠的用途是什么？", "设备状态巡检"),
]
QUESTION = EXAMPLES[0][0]


def load_model():
    # ROCm 沿用 PyTorch 的 cuda 接口；只加载原始检查点中的文本模型。
    return Qwen3_5ForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        device_map={"": 0}, local_files_only=True,
    )


def messages(question):
    return [
        {"role": "system", "content": "你是内部项目查询助手。只回答项目用途，不添加解释。"},
        {"role": "user", "content": question},
    ]


# 1. 加载原始模型，添加 LoRA。PEFT 冻结基础权重，只训练新增的低秩矩阵。
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = get_peft_model(load_model(), LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_RANK * 2,
    target_modules="all-linear",
))
model.config.use_cache = False

# 2. 按模型的聊天模板编码，只对 assistant 回答计算 loss。
batches = []
for question, answer in EXAMPLES:
    conversation = messages(question)
    prefix = tokenizer.apply_chat_template(
        conversation, tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer.apply_chat_template(
        conversation + [{"role": "assistant", "content": answer}],
        tokenize=True, return_dict=False, return_tensors="pt", enable_thinking=False,
    ).to("cuda")
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    batches.append({"input_ids": ids, "labels": labels})

# 3. 每步轮流学习一条示例。
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
model.train()
for step in range(TRAIN_STEPS):
    optimizer.zero_grad()
    loss = model(**batches[step % len(batches)]).loss
    loss.backward()
    optimizer.step()
    print(f"step {step + 1}/{TRAIN_STEPS} loss={loss.item():.4f}", flush=True)

# 4. 保存 LoRA 和分词器，再释放训练时的模型及优化器。
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
del model, optimizer, loss
gc.collect()
torch.cuda.empty_cache()

# 5. 从原始模型和刚保存的 LoRA 重新加载，生成一条回答。
model = PeftModel.from_pretrained(load_model(), OUTPUT_DIR)
model.eval()
prompt = tokenizer.apply_chat_template(
    messages(QUESTION), tokenize=True, return_dict=False, return_tensors="pt",
    add_generation_prompt=True, enable_thinking=False,
).to("cuda")
with torch.inference_mode():
    result = model.generate(
        input_ids=prompt, attention_mask=torch.ones_like(prompt),
        max_new_tokens=32, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
print(tokenizer.decode(result[0, prompt.shape[1]:], skip_special_tokens=True).strip())
print(f"LoRA 已保存到 {OUTPUT_DIR}")

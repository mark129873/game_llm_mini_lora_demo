"""最小 LoRA 演示：训练 → 保存 → 重新加载 → 回答一个问题。"""

# 可以把原始模型想成一本大书，把 LoRA 想成附在书旁边的小笔记。
# 训练只修改小笔记，原始模型文件不会被改动。
# 每次运行都从原始模型开始，不会接着上一次的 LoRA 继续训练。

import gc

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


def load_model():
    # 两次加载都用同样的设置，所以把这几行放进一个小函数。
    # 这是相对路径，请按 README 先进入项目目录，再运行本脚本。
    # Qwen3_5ForCausalLM 只使用原始模型中的文本部分。
    return Qwen3_5ForCausalLM.from_pretrained(
        "models/Qwen3.5-0.8B",
        dtype=torch.float32,           # 用 32 位浮点数加载权重。
        attn_implementation="eager",   # 使用普通的注意力实现。
        device_map={"": 0},            # 把整个模型放到第 0 张显卡。
        local_files_only=True,        # 只读本地模型。
    )


def messages(question):
    # 聊天模型需要知道谁在说话：system 规定任务，user 提出问题。
    # 训练和测试都使用同样的说话格式。
    return [
        {"role": "system", "content": "你是内部项目查询助手。只回答项目用途，不添加解释。"},
        {"role": "user", "content": question},
    ]


# 1. 加载原始模型，给它加上需要训练的 LoRA。
torch.manual_seed(42)  # 固定随机种子，便于重复实验时比较结果。

# 模型不直接读汉字。分词器把文字切成小块，再把每一块换成数字编号。
# 这种小块叫 token，一个 token 不一定等于一个汉字。
tokenizer = AutoTokenizer.from_pretrained("models/Qwen3.5-0.8B", local_files_only=True)
if tokenizer.pad_token_id is None:
    # 没有专用的填充标记时，借用结束标记，供后面的生成函数使用。
    tokenizer.pad_token = tokenizer.eos_token

# PEFT 会冻结原始权重：它们参与计算，但训练时不更新。
# 新加的 LoRA 参数可以更新，相当于只修改“小笔记”。
model = get_peft_model(load_model(), LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 根据前面的文字，学习预测下一个 token。
    r=8,                        # LoRA 的大小；越大，需要训练的参数越多。
    lora_alpha=16,               # 控制 LoRA 对原模型计算结果的影响强度。
    target_modules="all-linear",  # 在线性层上添加 LoRA。
))
model.config.use_cache = False  # 训练时关闭生成缓存，减少不必要的显存占用。


# 2. 把问答变成数字，并告诉模型“只给回答部分打分”。
# 本机使用 ROCm；PyTorch 仍把显卡接口叫作 cuda，这里实际使用的是 AMD 显卡。
# 三条虚构问答就是全部教材，放在使用它们的循环旁边。
examples = [
    ("项目青禾的用途是什么？", "离线文档检索"),
    ("项目星桥的用途是什么？", "会议纪要整理"),
    ("项目海棠的用途是什么？", "设备状态巡检"),
]
batches = []
for question, answer in examples:
    conversation = messages(question)

    # prefix 只有系统要求、问题和“轮到助手回答”的开头，没有正确答案。
    # 聊天模板负责添加模型认识的角色标记；关闭思考模式，直接学习回答。
    prefix = tokenizer.apply_chat_template(
        conversation, tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False,
    )

    # ids 是完整问答的编号。return_tensors="pt" 把编号装成 PyTorch 张量，
    # 可以理解为一张数字表；to("cuda") 把这张表放进显卡。
    ids = tokenizer.apply_chat_template(
        conversation + [{"role": "assistant", "content": answer}],
        tokenize=True, return_dict=False, return_tensors="pt", enable_thinking=False,
    ).to("cuda")

    # labels 是用来打分的正确答案，先复制一份，避免改动模型的输入。
    # -100 是 PyTorch 约定的“这一格不打分”，所以问题部分不会计入 loss。
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    batches.append({"input_ids": ids, "labels": labels})


# 3. 轮流学习三条问答，每一步都调整一次 LoRA。
# 优化器负责调整参数；这里只把允许训练的 LoRA 参数交给它。
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4,  # 学习率是 0.0002，控制每次调整 LoRA 的幅度。
)
model.train()  # 切换到训练模式；真正的学习发生在下面的循环里。

for step in range(20):  # 更新 20 次，每次学习一条问答。
    optimizer.zero_grad()  # 清空上一步的梯度，避免把上一步的调整方向重复算进去。

    # % 是取余数：依次选第 0、1、2、0、1、2……条问答。
    # loss 是“模型回答和示例答案有多不一样”的分数，通常越小越好。
    loss = model(**batches[step % len(batches)]).loss
    loss.backward()   # 算出各个参数应该往哪个方向调整，这些结果叫梯度。
    optimizer.step()  # 按照梯度和学习率，更新一次 LoRA 参数。
    print(f"step {step + 1}/20 loss={loss.item():.4f}", flush=True)


# 4. 全部训练完成后保存一次；保存的是 LoRA，不会复制一整份原始模型。
# 固定使用 outputs/adapter，因此下次运行会覆盖这个目录中的同名结果文件。
model.save_pretrained("outputs/adapter")
tokenizer.save_pretrained("outputs/adapter")  # 一并保存分词器，方便以后使用。

# 先释放训练模型和优化器，给重新加载腾出内存和显存。
# del 只移除内存中的引用，不会删除硬盘上的模型或训练结果。
del model, optimizer, loss
gc.collect()
torch.cuda.empty_cache()


# 5. 重新读取原始模型，再装上刚保存的 LoRA，验证保存的结果能用。
model = PeftModel.from_pretrained(load_model(), "outputs/adapter")
model.eval()  # 切换到回答问题的模式。

# 测试输入只有问题，没有答案；让模型自己接着往后写。
prompt = tokenizer.apply_chat_template(
    messages("项目青禾的用途是什么？"), tokenize=True, return_dict=False, return_tensors="pt",
    add_generation_prompt=True, enable_thinking=False,
).to("cuda")

with torch.inference_mode():  # 只生成回答，不记录训练所需的梯度。
    result = model.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),  # 输入没有填充，每个位置都是真实内容。
        max_new_tokens=32,            # 最多新生成 32 个 token，不是 32 个汉字。
        do_sample=False,              # 每次选概率最高的下一个 token，不随机抽取。
        use_cache=True,               # 回答时缓存之前的计算，减少重复计算。
        pad_token_id=tokenizer.pad_token_id,
    )

# 生成结果包含“原来的问题 + 新回答”。切掉问题，只把新回答转回文字。
# skip_special_tokens=True 会隐藏角色标记、结束标记等内部符号。
print(tokenizer.decode(result[0, prompt.shape[1]:], skip_special_tokens=True).strip())
print("LoRA 已保存到 outputs/adapter")

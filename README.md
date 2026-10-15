# Qwen3.5 LoRA Demo

RX 7700 XT / WSL2 Ubuntu 24.04。


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

# 文件树

```text
lora/
├── lora_demo.py                 # 主程序：LoRA 训练、保存、重载验证和推理
├── env.sh                       # 激活虚拟环境，配置 ROCm 和缓存路径
├── install_rocm.sh              # 安装 AMD GPU 所需的 ROCm 系统运行库
├── install_python.sh            # 创建虚拟环境，用 uv 安装依赖并下载模型
├── requirements.txt             # 固定 Python 依赖版本，供 uv 安装使用
├── README.md                    # 项目简介和运行命令
├── INSTALL.md                   # WSL、系统依赖和 Python 环境的安装步骤
├── tree.md                      # 文件树及各文件用途
├── .gitignore                   # 排除环境、缓存、模型和训练产物等本地文件
├── .gitattributes               # 统一文本文件的换行格式
├── models/                      # 本地基础模型
│   └── Qwen3.5-0.8B/
│       ├── config.json          # 模型结构和基础配置
│       ├── model.safetensors-00001-of-00001.safetensors # 原始模型权重
│       ├── model.safetensors.index.json # 参数名称到权重文件的索引
│       ├── tokenizer.json       # 分词器数据和编码规则
│       ├── tokenizer_config.json # 分词器设置和特殊 token 配置
│       ├── vocab.json           # 分词器词表
│       ├── chat_template.jinja  # system、user、assistant 消息的拼接模板
│       ├── preprocessor_config.json # 图像预处理配置，当前文本训练不使用
│       ├── video_preprocessor_config.json # 视频预处理配置，当前文本训练不使用
│       └── source.json          # 旧下载流程留下的模型来源记录
├── cache/                       # 下载、包管理和 GPU 编译缓存
│   └── comgr/                   # AMD GPU 编译缓存
├── .venv/                       # Python 虚拟环境及已安装的第三方包
└── .git/                        # Git 仓库的版本历史和内部数据
```

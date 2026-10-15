# 2026-10-16 项目清理与 Git 初始化

项目路径：/home/nnnnnn/lora。

## 文件检查结论

现有源码脚本均由安装、训练、下载或文档中的复现流程引用。未发现可以直接废弃的源码脚本。

模型原始文件、训练后的 adapter、Python 虚拟环境、GPU 编译缓存 cache/comgr 和历史安装记录保留。

清理范围：

- 安装完成后的 Torch、Triton wheel（可以按固定 URL、版本及 SHA256 重新下载）。
- pip 缓存和 uv 缓存（加速再次安装使用）。
- 项目顶层 __pycache__（Python 会自动重建）。
- logs/gpu-check.missing-libraries.log（旧缺库报错；原因与修复已写入 INSTALL.md）。

清理前本项目实际分配空间：8,498,761,728 bytes。
清理后：6,518,870,016 bytes。
共释放：1,979,891,712 bytes，约 1.844 GiB。

uv 缓存中约 4.2 GiB 文件与 .venv 共享硬链接；uv cache clean 清除缓存入口后，运行环境仍持有这些文件，因此这一部分不计入释放空间。

这是 WSL 文件系统内部回收的空间；本次没有压缩 ext4.vhdx，Windows C 盘可用空间不一定立即同步增长。

## 实际执行的清理及 Git 初始化命令

以下为 Ubuntu-24.04 普通用户下执行的命令；删除前逐个验证了目标属于项目目录，且不是符号链接。

```bash
set -euo pipefail
cd /home/nnnnnn/lora
source env.sh
python - <<'PY'
from pathlib import Path
import json,subprocess
root=Path('/home/nnnnnn/lora').resolve(strict=True)
assert Path.cwd().resolve()==root
before=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0])
(root/'logs/cleanup-20261016.json').write_text(json.dumps({'before_allocated_bytes':before,'date':'2026-10-16'},indent=2)+'\n')
print('Project allocated before:',before)
PY
uv cache clean
python - <<'PY'
from pathlib import Path
import json,shutil,subprocess
root=Path('/home/nnnnnn/lora').resolve(strict=True)
names=[
    'cache/pip',
    '__pycache__',
    'downloads/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl',
    'downloads/triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl',
    'logs/gpu-check.missing-libraries.log',
]
for name in names:
    target=root/name
    if target.is_symlink():
        raise RuntimeError('Refusing symlink cleanup target: '+str(target))
    target.resolve().relative_to(root)
    assert target!=root and name not in ('.venv','models','outputs','cache/comgr')
    if target.is_dir(): shutil.rmtree(target)
    else: target.unlink(missing_ok=True)
    print('Removed:',name)
report_path=root/'logs/cleanup-20261016.json'
report=json.loads(report_path.read_text())
after=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0])
report.update(removed=names,uv_cache_cleaned=True,after_allocated_bytes=after,released_bytes=report['before_allocated_bytes']-after)
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
git init --initial-branch=main
git config --local core.autocrlf false
git config --local core.filemode true
git config --local user.useConfigOnly true
```

## Git 配置

已添加 .gitignore 和 .gitattributes。Git 保存源码、文档和固定版本 / 校验清单；.venv、cache、models、outputs、logs、下载的安装包被忽略。

当前无 Git 署名，因此仅初始化 main 分支并暂存初始文件，未创建提交。未设置远程仓库。

```bash
cd /home/nnnnnn/lora
git add -- .
git diff --cached --check
git status --short
git check-ignore .venv/bin/python models/Qwen3.5-0.8B/config.json outputs/smoke-test/adapter/adapter_model.safetensors logs/smoke-test.log cache/uv/example downloads/example.whl
```

## 清理后的验证命令

```bash
cd /home/nnnnnn/lora
source env.sh
python -m pip check
bash -n env.sh run.sh install_python.sh install_rocm.sh
PYTHONDONTWRITEBYTECODE=1 python -u infer_lora.py --adapter outputs/smoke-test/adapter --prompt '项目青禾的用途是什么？' 2>&1 | tee logs/cleanup-inference-20261016.log
```

验证结果：通过。pip check 无缺失依赖，Bash 语法检查通过；清理后重新加载基础模型和 adapter，仍回答“离线文档检索”，GPU 前向 / 反向检查也通过。

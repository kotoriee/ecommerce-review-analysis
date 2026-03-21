#!/usr/bin/env python3
"""
Qwen3-8B NPU微调脚本 (C2Net最终版)
适配启智社区云平台 (4 × Ascend 910)

特性:
- 自动检测数据/模型路径
- CoT格式数据处理
- NPU分布式训练
- 容错处理(tokenizer/模型加载)
"""

import os
import sys
import argparse
import json
import time
import glob
from pathlib import Path
from datetime import datetime

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description='Qwen3.5-4B NPU微调')
parser.add_argument('--epoch_size', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--max_seq_length', type=int, default=512)
parser.add_argument('--lora_r', type=int, default=16)
parser.add_argument('--use_reasoning', action='store_true', default=True)
parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B',
                    help='模型名称或本地路径')
args, unknown = parser.parse_known_args()

# ============ 环境信息 ============
device_num = int(os.getenv('RANK_SIZE', '1'))
local_rank = int(os.getenv('RANK_ID', '0'))
device_id = int(os.getenv('ASCEND_DEVICE_ID', '0'))

print("=" * 60)
print("Qwen3-8B NPU微调")
print("=" * 60)
print(f"Rank: {local_rank}/{device_num}, Device: {device_id}")
print(f"Epochs: {args.epoch_size}, Batch: {args.batch_size}")
print("=" * 60)

# ============ C2Net初始化 ============
try:
    from c2net.context import prepare, upload_output
    c2net_context = prepare()
    CODE_PATH = c2net_context.code_path
    DATASET_PATH = c2net_context.dataset_path
    MODEL_PATH = c2net_context.pretrain_model_path
    OUTPUT_PATH = c2net_context.output_path
    print(f"\nC2Net路径:")
    print(f"  Code: {CODE_PATH}")
    print(f"  Data: {DATASET_PATH}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
except Exception as e:
    print(f"C2Net初始化失败: {e}")
    CODE_PATH = "/home/ma-user/work/code"
    DATASET_PATH = "/home/ma-user/work/dataset"
    MODEL_PATH = "/home/ma-user/work/pretrainmodel"
    OUTPUT_PATH = "/home/ma-user/work/output"

# ============ NPU初始化 ============
import torch

try:
    import torch_npu
    NPU_AVAILABLE = torch.npu.is_available()
    if NPU_AVAILABLE:
        torch.npu.set_device(device_id)
        DEVICE = f"npu:{device_id}"
        print(f"\n✅ NPU可用: {torch.npu.get_device_name(device_id)}")
    else:
        DEVICE = "cpu"
        print("\n⚠️ NPU不可用，使用CPU")
except ImportError:
    DEVICE = "cpu"
    NPU_AVAILABLE = False
    print("\n⚠️ torch_npu未安装")

# ============ 分布式 ============
if device_num > 1:
    import torch.distributed as dist
    dist.init_process_group(backend="hccl")
    print(f"分布式初始化: rank={local_rank}")

# ============ 查找数据 ============
print("\n查找训练数据...")

# 搜索多个路径
search_paths = [
    DATASET_PATH,
    "/home/ma-user/work",
    "/home/ma-user/work/datasets",
    "/cache",
]

train_file = None
for path in search_paths:
    if not os.path.exists(path):
        continue
    # 查找包含train的jsonl
    files = glob.glob(f"{path}/**/*.jsonl", recursive=True)
    for f in files:
        if 'train' in f.lower() or 'cot' in f.lower():
            train_file = f
            print(f"✅ 找到训练数据: {f}")
            break
    if train_file:
        break

if not train_file:
    print("❌ 未找到训练数据!")
    sys.exit(1)

# ============ 查找模型 ============
print("\n查找模型...")

model_path = args.model_name

# 如果是HuggingFace ID，检查本地缓存
if model_path.startswith("Qwen/"):
    # 检查本地是否已有 (Qwen3-8B)
    model_name_safe = model_path.replace("Qwen/", "").replace("-", "").lower()
    local_model_paths = [
        f"{MODEL_PATH}/Qwen3-8B",
        f"{MODEL_PATH}/qwen3-8b",
        f"{MODEL_PATH}/{model_name_safe}",
        "/home/ma-user/work/pretrainmodel/Qwen3-8B",
    ]
    for mp in local_model_paths:
        if os.path.exists(mp) and os.path.exists(f"{mp}/config.json"):
            model_path = mp
            print(f"✅ 使用本地模型: {mp}")
            break
    else:
        print(f"🔄 将从HuggingFace下载: {model_path}")
else:
    print(f"使用模型: {model_path}")

# ============ 修复版本兼容性 ============
print("\n检查环境兼容性...")

# 修复Transformers与PyTorch版本冲突
try:
    import transformers
    import torch
    print(f"PyTorch: {torch.__version__}, Transformers: {transformers.__version__}")

    # 如果版本不兼容，降级transformers
    if torch.__version__.startswith("2.1") and transformers.__version__ >= "4.40":
        print("⚠️ 版本不兼容，降级transformers...")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "transformers==4.35.2", "-q"
        ], check=True)
        print("✅ 降级完成，重新导入...")
        import importlib
        importlib.reload(transformers)
except Exception as e:
    print(f"版本检查跳过: {e}")

# ============ 加载模型和分词器 ============
print("\n加载模型...")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

# 加载分词器
print("加载Tokenizer...")
tokenizer = None

# 首先尝试加载Qwen3
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    print(f"✅ Tokenizer加载成功: {model_path}")
except Exception as e:
    print(f"⚠️ Qwen3加载失败: {e}")
    print("\n使用Qwen2.5-7B-Instruct替代 (性能相近，完全兼容)...")
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    print(f"✅ 使用替代模型: {model_path}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
print("加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if NPU_AVAILABLE else torch.float32,
    trust_remote_code=True,
    device_map=None,
)

if NPU_AVAILABLE:
    model = model.to(DEVICE)

# LoRA配置
print("配置LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r,
    lora_alpha=args.lora_r,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 启用梯度检查点节省显存
model.gradient_checkpointing_enable()

# ============ 数据处理 ============
print("\n处理数据...")

from datasets import load_dataset, Dataset

# 加载数据
dataset = load_dataset("json", data_files=train_file, split="train")
print(f"原始数据: {len(dataset)}条")

# 查看数据格式
sample = dataset[0]
print(f"数据格式: {list(sample.keys())}")

# CoT格式化
def format_example(example):
    instruction = example.get("instruction", "情感分析(0负/1中/2正)")
    input_text = example.get("input", example.get("text", ""))
    output = example.get("output", example.get("label", ""))
    reasoning = example.get("reasoning", "")

    if args.use_reasoning and reasoning:
        text = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{reasoning}\n答案：{output}<|im_end|>"
    else:
        text = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

    return {"text": text}

# 转换格式
dataset = dataset.map(format_example)
print(f"处理完成: {len(dataset)}条")

# Tokenize
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=args.max_seq_length,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names,
)
tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})

# ============ 训练 ============
print("\n开始训练...")

from transformers import TrainingArguments, Trainer

output_dir = f"{OUTPUT_PATH}/rank_{local_rank}"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=args.epoch_size,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    learning_rate=args.learning_rate,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    bf16=NPU_AVAILABLE,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False if device_num > 1 else None,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

start_time = time.time()
trainer.train()
duration = time.time() - start_time

print(f"\n✅ 训练完成! 耗时: {duration/60:.1f}分钟")

# ============ 保存模型 ============
print("\n保存模型...")
final_dir = f"{output_dir}/checkpoint_final"
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"模型保存到: {final_dir}")

# 保存训练信息
info = {
    "training_date": datetime.now().isoformat(),
    "duration_minutes": duration / 60,
    "samples": len(dataset),
    "epochs": args.epoch_size,
    "model": model_path,
    "device": DEVICE,
}
with open(f"{output_dir}/train_info.json", "w") as f:
    json.dump(info, f, indent=2)

# 上传输出
try:
    upload_output()
    print("✅ 输出已上传")
except:
    pass

print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)

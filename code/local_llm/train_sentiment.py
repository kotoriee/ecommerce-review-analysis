#!/usr/bin/env python3
"""
Qwen3-4B 情感分析微调脚本（Unsloth QLoRA）

基于 lib/qwen3_(14b)_reasoning_conversational.py 官方教程改编
适配：2000条英文电商评论 + Qwen3-4B + 本地 GPU

Usage:
    # 正式训练
    python train_sentiment.py

    # 指定数据路径和输出路径
    python train_sentiment.py --data data/processed/train.json --output models/qwen3-4b-sentiment-lora

    # 测试模式 (30步，验证流程)
    python train_sentiment.py --test
"""

import json
import argparse
from pathlib import Path
import torch


# ============== 配置 ==============

DEFAULT_MODEL   = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
DEFAULT_DATA    = "data/processed/train.json"
DEFAULT_OUTPUT  = "models/qwen3-4b-sentiment-lora"
MAX_SEQ_LENGTH  = 256   # 电商短评论，256足够；节省显存（OOM修复）
LORA_RANK       = 16    # 8GB显存用16；r=32会OOM
RANDOM_STATE    = 3407


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-4B 情感分析 QLoRA 微调")
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data",    type=str, default=DEFAULT_DATA)
    parser.add_argument("--output",  type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs",  type=int, default=3,    help="情感分析任务用3轮")
    parser.add_argument("--batch",   type=int, default=1,   help="8GB显存用1")
    parser.add_argument("--grad-acc",type=int, default=16,  help="batch=1×16=等效16")
    parser.add_argument("--lr",      type=float, default=2e-5, help="情感分析专用：保护预训练知识")
    parser.add_argument("--test",    action="store_true", help="测试模式：只跑30步")
    return parser.parse_args()


# ============== 主训练流程 ==============

def main():
    args = parse_args()

    # 1. 检查依赖
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install unsloth trl datasets")
        return

    # 2. 加载数据
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 训练数据不存在: {data_path}")
        print("请先运行:")
        print("  python cloud_agent/generate_cot_data.py --from-hf --count 2000")
        print("  python local_llm/data_formatter.py")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"加载训练数据: {len(raw_data)} 条")

    # 3. 加载模型（4-bit QLoRA）
    print(f"\n加载模型: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # 4. 添加 LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,   # alpha = rank
        lora_dropout=0,          # 0 已优化
        bias="none",
        use_gradient_checkpointing="unsloth",  # 节省 30% VRAM
        random_state=RANDOM_STATE,
        use_rslora=False,
        loftq_config=None,
    )

    # 5. 构建 HuggingFace Dataset
    dataset = Dataset.from_list(raw_data)  # raw_data 已是 [{"text": "..."}]

    # 6. 显示内存状态
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        max_mem   = round(gpu_stats.total_memory / 1024 ** 3, 2)
        print(f"\nGPU: {gpu_stats.name} | 总显存: {max_mem} GB | 已用: {start_mem} GB")

    # 7. 训练配置（来自 lib/qwen3_(14b)_reasoning_conversational.py）
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        warmup_ratio=0.05,          # 博客3：大数据集不需要长warmup
        num_train_epochs=args.epochs if not args.test else 1,
        max_steps=30 if args.test else -1,
        learning_rate=args.lr,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,           # 博客3：0.01防过拟合（原0.001）
        lr_scheduler_type="linear",
        seed=RANDOM_STATE,
        output_dir=args.output + "_checkpoints",
        report_to="none",
        bf16=True,
        fp16=False,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,                # 短评论打包，减少padding浪费
        dataloader_num_workers=0,    # 避免worker进程OOM（exit 137）
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    print(f"\n开始训练:")
    print(f"  Epochs: {args.epochs if not args.test else '1 (test: 30 steps)'}")
    print(f"  Batch: {args.batch} × grad_acc {args.grad_acc} = 等效 batch {args.batch * args.grad_acc}")
    print(f"  LR: {args.lr}")
    print("=" * 60)

    trainer_stats = trainer.train()

    # 8. 显示训练统计
    if torch.cuda.is_available():
        used_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        print(f"\n训练时间: {trainer_stats.metrics['train_runtime']:.0f}s "
              f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
        print(f"峰值显存: {used_mem} GB")

    # 9. 保存 LoRA adapters
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nLoRA adapters 已保存: {output_dir}")

    # 10. 导出合并版本（用于 vLLM / Ollama）
    merged_dir = str(output_dir) + "_merged_16bit"
    print(f"\n导出合并模型（16bit）→ {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    gguf_dir = str(output_dir) + "_gguf"
    print(f"导出 GGUF（q4_k_m）→ {gguf_dir}")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")

    print("\n" + "=" * 60)
    print("微调完成！")
    print(f"  LoRA adapters:  {output_dir}")
    print(f"  合并 16bit:     {merged_dir}")
    print(f"  GGUF (Ollama):  {gguf_dir}.Q4_K_M.gguf")
    print("=" * 60)
    print("\n下一步：运行推理验证")
    print("  ollama create qwen3-sentiment -f Modelfile")
    print("  python local_llm/inference/predictor.py")


if __name__ == "__main__":
    main()

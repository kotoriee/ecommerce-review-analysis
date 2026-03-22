"""
Google Colab 专用 - Qwen3-4B Curriculum 评估脚本
在 Colab T4 GPU 上批量推理，分流本地计算压力

## 使用步骤:
1. 上传此文件到 Colab
2. 安装依赖: !pip install vllm transformers -q
3. 从 GitHub 下载模型或挂载 Google Drive
4. 运行: !python curriculum_eval_colab.py --stage s1

## 或者使用 notebook 单元格:
```python
!git clone https://github.com/kotoriee/ecommerce-review-analysis.git
%cd ecommerce-review-analysis
!pip install vllm transformers -q
!python colab/curriculum_eval_colab.py --stage s5 --samples 1000
```
"""

import json
import argparse
import re
import time
from pathlib import Path
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Colab vLLM Batch Evaluation")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["s1", "s2", "s3", "s4", "s5"],
                        help="Which stage to evaluate: s1(600), s2(1200), s3(2400), s4(4800), s5(8500)")
    parser.add_argument("--model-dir", type=str, default="./models/curriculum",
                        help="Directory containing LoRA adapters")
    parser.add_argument("--data-dir", type=str, default="./data/curriculum",
                        help="Directory containing validation data")
    parser.add_argument("--base-model", type=str,
                        default="unsloth/Qwen3-4B-unsloth-bnb-4bit",
                        help="Base model name (Hugging Face ID)")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of validation samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="vLLM batch size (Colab T4 can handle 64)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: eval_{stage}_colab.json)")
    return parser.parse_args()


def get_model_and_data(stage: str, model_dir: str, data_dir: str) -> Tuple[str, str, str]:
    """根据 stage 获取模型路径和数据路径"""
    stage_map = {
        "s1": ("lora_s1_600", "600"),
        "s2": ("lora_s2_1200", "1200"),
        "s3": ("lora_s3_2400", "2400"),
        "s4": ("lora_s4_4800", "4800"),
        "s5": ("lora_s5_full", "8500"),
    }

    model_name, data_suffix = stage_map[stage]
    lora_path = f"{model_dir}/{model_name}"

    # 检查模型是否存在
    if not Path(lora_path).exists():
        print(f"⚠️  模型不存在: {lora_path}")
        print(f"   请确保从 GitHub 下载了模型或使用 Google Drive 挂载")

    return lora_path, f"{data_dir}/val_fixed.json", model_name


def create_prompt(review_text: str) -> str:
    """构建推理 prompt - 与训练格式统一"""
    return f"""<|im_start|>system
You are a sentiment analysis expert for e-commerce reviews.
Analyze the review and output ONLY a JSON object with these exact fields:
- sentiment: 0 (negative), 1 (neutral), or 2 (positive)
- confidence: float between 0 and 1
- rationale: brief explanation in English

Output format: {{"sentiment": X, "confidence": Y.YZ, "rationale": "explanation"}}<|im_end|>
<|im_start|>user
Review: {review_text}<|im_end|>
<|im_start|>assistant
<thinking>
"""


def load_validation_data(data_path: str, max_samples: int) -> List[dict]:
    """加载验证数据 - 支持 Alpaca 格式"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data[:max_samples]:
        # Alpaca format
        if "input" in item and "output" in item:
            review = item.get("input", "").strip()
            try:
                label = int(item.get("output", "-1").strip())
            except ValueError:
                continue
            if review and label in [0, 1, 2]:
                samples.append({
                    "review": review[:500],
                    "label": label
                })

    return samples


def extract_sentiment(text: str) -> int:
    """从输出中提取情感标签"""
    text_lower = text.lower()

    # JSON 格式匹配
    match = re.search(r'"sentiment"\s*[:=]\s*([0-2])', text_lower)
    if match:
        return int(match.group(1))

    # 关键词匹配
    negative_words = ['negative', 'disappointed', 'terrible', 'bad', 'worst', 'hate', 'poor', 'awful']
    positive_words = ['positive', 'excellent', 'great', 'good', 'amazing', 'love', 'perfect', 'wonderful']

    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)

    if neg_count > pos_count:
        return 0
    elif pos_count > neg_count:
        return 2
    else:
        return 1


def evaluate_with_unsloth(base_model: str, lora_path: str, samples: List[dict], batch_size: int, max_tokens: int):
    """使用 Unsloth 批量评估 (Colab T4 优化版)"""
    from unsloth import FastLanguageModel
    import torch

    print(f"🚀 加载 Unsloth 模型")
    print(f"   Base: {base_model}")
    print(f"   LoRA: {lora_path}")
    print(f"   样本数: {len(samples)}, Batch: {batch_size}")

    # 加载基础模型 (4-bit 节省显存)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=512,
        load_in_4bit=True,
    )

    # 加载 LoRA
    model = FastLanguageModel.get_peft_model(model)
    model.load_adapter(lora_path, adapter_name="default")
    model.set_adapter("default")
    FastLanguageModel.for_inference(model)

    print(f"   模型已加载到: {next(model.parameters()).device}")

    # 准备 prompts
    prompts = [create_prompt(s["review"]) for s in samples]
    true_labels = [s["label"] for s in samples]

    # 批量推理
    print(f"\n⏱️  开始批量推理...")
    start_time = time.time()

    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode (skip input tokens)
        input_len = inputs['input_ids'].shape[1]
        for output in outputs:
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            all_outputs.append(response)

        # 进度显示
        current_batch = i // batch_size + 1
        if current_batch % 5 == 0 or current_batch == 1:
            progress = min((i + batch_size), len(prompts))
            print(f"   进度: {progress}/{len(prompts)} ({current_batch}/{total_batches} 批次)")

    infer_time = time.time() - start_time
    speed = len(all_outputs) / infer_time

    print(f"\n✅ 推理完成: {infer_time:.1f}s ({speed:.2f} 条/秒, {speed*60:.1f} 条/分钟)")

    # 解析结果
    results = []
    correct = 0
    class_stats = {0: {"total": 0, "correct": 0}, 1: {"total": 0, "correct": 0}, 2: {"total": 0, "correct": 0}}

    for i, pred_text in enumerate(all_outputs):
        pred_label = extract_sentiment(pred_text)
        true_label = true_labels[i]

        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1

        class_stats[true_label]["total"] += 1
        if is_correct:
            class_stats[true_label]["correct"] += 1

        results.append({
            "true": true_label,
            "pred": pred_label,
            "correct": is_correct,
            "review": samples[i]["review"][:100],
            "raw": pred_text[:150],
        })

    accuracy = correct / len(results) * 100

    return {
        "accuracy": accuracy,
        "total": len(results),
        "correct": correct,
        "time": infer_time,
        "speed": speed,
        "class_stats": class_stats,
        "results": results,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print(f"🎓 Qwen3-4B Curriculum 评估 - {args.stage.upper()}")
    print("=" * 60)

    # 获取路径
    lora_path, data_path, model_name = get_model_and_data(args.stage, args.model_dir, args.data_dir)

    # 检查文件
    if not Path(data_path).exists():
        print(f"❌ 数据文件不存在: {data_path}")
        print("   请从 GitHub 下载数据:")
        print("   !git clone https://github.com/kotoriee/ecommerce-review-analysis.git")
        return

    if not Path(lora_path).exists():
        print(f"❌ LoRA 模型不存在: {lora_path}")
        return

    # 加载数据
    print(f"\n📂 加载验证数据: {data_path}")
    samples = load_validation_data(data_path, args.samples)
    print(f"   有效样本: {len(samples)}")

    if len(samples) == 0:
        print("❌ 没有有效样本")
        return

    # 评估
    eval_result = evaluate_with_unsloth(
        base_model=args.base_model,
        lora_path=lora_path,
        samples=samples,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    # 打印结果
    print(f"\n{'='*60}")
    print("📊 评估结果")
    print(f"{'='*60}")
    print(f"总样本: {eval_result['total']}")
    print(f"正确: {eval_result['correct']}")
    print(f"准确率: {eval_result['accuracy']:.2f}%")
    print(f"推理速度: {eval_result['speed']:.1f} 条/秒")

    # 类别分布
    print(f"\n类别准确率:")
    label_names = {0: "负面", 1: "中性", 2: "正面"}
    for label, stats in eval_result["class_stats"].items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {label_names[label]}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

    # 保存结果
    output_file = args.output or f"eval_{args.stage}_colab.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {output_file}")

    # 错误示例
    errors = [r for r in eval_result["results"] if not r["correct"]][:3]
    if errors:
        print(f"\n❌ 错误示例:")
        for i, e in enumerate(errors):
            print(f"\n{i+1}. 真实={e['true']}, 预测={e['pred']}")
            print(f"   评论: {e['review'][:80]}...")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

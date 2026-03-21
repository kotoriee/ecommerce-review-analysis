"""
使用 Transformers 批量评估模型准确率（优化版）
"""

import json
import argparse
import re
import time
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/qwen3-4b-sentiment-lora_merged_16bit")
    parser.add_argument("--data", type=str, default="data/processed/test.json")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8, help="批量大小")
    parser.add_argument("--max-new-tokens", type=int, default=60, help="减少生成长度加速")
    return parser.parse_args()


def extract_label_from_cot(text: str) -> tuple[int, str]:
    """从CoT格式数据中提取标签和review文本"""
    s = re.search(r'"sentiment":\s*([0-2])', text)
    true_label = int(s.group(1)) if s else -1

    review_match = re.search(r'Review:\s*([^<\n]+)', text)
    if review_match:
        review_text = review_match.group(1).strip()[:400]
    else:
        review_text = text[:300]

    return true_label, review_text


def create_prompt(review_text: str) -> str:
    """构建推理prompt"""
    return f"""<|im_start|>system
You are a sentiment analysis expert for e-commerce reviews.
Analyze the review and output ONLY a JSON object:
{{"sentiment": 0 or 1 or 2, "confidence": 0.0-1.0, "rationale": "brief"}}<|im_end|>
<|im_start|>user
Review: {review_text}<|im_end|>
<|im_start|>assistant
<thinking>"""


def extract_sentiment(text: str) -> int:
    """从输出提取情感标签"""
    match = re.search(r'"sentiment"\s*[:=]\s*([0-2])', text.lower())
    return int(match.group(1)) if match else -1


def batch_generate(model, tokenizer, prompts, max_new_tokens=60):
    """批量生成"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(["<|im_end|>", "</thinking>"]),
        )

    # 解码生成的部分
    input_lengths = inputs['input_ids'].shape[1]
    results = []
    for output in outputs:
        generated = output[input_lengths:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(text)
    return results


def main():
    args = parse_args()

    # 加载数据
    with open(args.data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    test_samples = raw_data[:args.max_samples]
    print(f"加载测试数据: {len(test_samples)} 条")

    # 准备数据
    prompts = []
    true_labels = []
    for item in test_samples:
        label, review = extract_label_from_cot(item["text"])
        if label != -1:
            prompts.append(create_prompt(review))
            true_labels.append(label)

    print(f"有效样本: {len(prompts)}")

    # 加载模型
    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"模型已加载到: {model.device}")

    # 批量推理
    print(f"\n开始批量推理 (batch_size={args.batch_size}, max_tokens={args.max_new_tokens})...")
    start_time = time.time()

    all_preds = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Batches"):
        batch_prompts = prompts[i:i + args.batch_size]
        outputs = batch_generate(model, tokenizer, batch_prompts, args.max_new_tokens)
        all_preds.extend(outputs)

    infer_time = time.time() - start_time
    speed = len(all_preds) / infer_time

    print(f"\n推理完成: {infer_time:.1f}s, {speed:.2f} 条/秒")

    # 评估准确率
    correct = 0
    valid = 0
    for true, pred_text in zip(true_labels, all_preds):
        pred = extract_sentiment(pred_text)
        if pred != -1:
            valid += 1
            if pred == true:
                correct += 1

    accuracy = correct / valid * 100 if valid > 0 else 0

    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"总样本: {len(all_preds)}")
    print(f"解析成功: {valid}")
    print(f"正确: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"推理速度: {speed:.2f} 条/秒")


if __name__ == "__main__":
    main()

"""
评估微调后的模型准确率
使用合并后的16bit模型直接推理（无需Ollama）
"""

import json
import argparse
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/qwen3-4b-sentiment-lora_merged_16bit")
    parser.add_argument("--data", type=str, default="data/processed/test.json")
    parser.add_argument("--max-samples", type=int, default=500, help="评估样本数")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def extract_sentiment_from_output(text: str) -> int:
    """从模型输出中提取情感标签"""
    # 优先匹配格式: "sentiment": 0/1/2
    patterns = [
        r'"sentiment"\s*[:=]\s*([0-2])',
        r'sentiment["\']?\s*[:=]\s*["\']?([0-2])',
        r'([0-2])\s*[:=]\s*["\']?sentiment',
        r'情感[:：]\s*([0-2])',
        r'[\[\{\(]([0-2])[\]\}\)]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    # 关键词匹配（fallback）
    text_lower = text.lower()
    if any(w in text_lower for w in ['negative', '消极', '负面', '差', '坏', '不满意']):
        return 0
    if any(w in text_lower for w in ['positive', '积极', '正面', '好', '优秀', '满意']):
        return 2
    if any(w in text_lower for w in ['neutral', '中性', '一般', '普通', '还行']):
        return 1
    return -1  # 未识别


def extract_label_from_cot(text: str) -> tuple[int, str]:
    """从CoT格式数据中提取标签和review文本"""
    # 提取助手回复部分的sentiment
    match = re.search(r'assistant\s*\n.*?<think>.*?</think>\s*(\{[^}]+\})', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            true_label = int(result.get("sentiment", -1))
        except:
            # 简单正则提取
            s = re.search(r'"sentiment":\s*([0-2])', text)
            true_label = int(s.group(1)) if s else -1
    else:
        s = re.search(r'"sentiment":\s*([0-2])', text)
        true_label = int(s.group(1)) if s else -1

    # 提取review文本（user prompt中 Review: 后面）
    review_match = re.search(r'Review:\s*([^<]+)', text)
    if review_match:
        review_text = review_match.group(1).strip()[:500]
    else:
        review_text = text[:300]  # fallback

    return true_label, review_text


def create_prompt(review_text: str) -> str:
    """构建与训练格式一致的推理prompt"""
    return f"""<|im_start|>system
You are a sentiment analysis expert for e-commerce reviews.
Analyze the review and output ONLY a JSON object with these exact fields:
- sentiment: 0 (negative) or 1 (positive)
- confidence: float between 0 and 1
- rationale: brief explanation in English

Output format: {{"sentiment": X, "confidence": Y.YZ, "rationale": "explanation"}}<|im_end|>
<|im_start|>user
Review: {review_text}<|im_end|>
<|im_start|>assistant
<think>
"""


def evaluate():
    args = parse_args()

    # 加载测试数据
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 测试数据不存在: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 取前max_samples个样本
    test_samples = raw_data[:args.max_samples]
    print(f"加载测试数据: {len(test_samples)} 条")

    # 加载模型
    print(f"\n加载模型: {args.model}")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型不存在: {model_path}")
        print("可用模型路径:")
        for p in Path("models").glob("*"):
            print(f"  - {p}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"模型加载完成，参数: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # 评估
    correct = 0
    total = 0
    predictions = []
    skipped = 0

    print(f"\n开始评估...")
    for item in tqdm(test_samples, desc="Evaluating"):
        text = item["text"]
        true_label, review_text = extract_label_from_cot(text)

        if true_label == -1 or not review_text:
            skipped += 1
            continue

        # 推理
        prompt = create_prompt(review_text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred_label = extract_sentiment_from_output(response)

        is_correct = (pred_label == true_label) if pred_label != -1 else False
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "review": review_text[:100],
            "true": true_label,
            "pred": pred_label,
            "raw": response[:200],
            "correct": is_correct
        })

    # 统计结果
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"评估结果")
    print(f"{'='*50}")
    print(f"总样本: {total}")
    if skipped > 0:
        print(f"跳过(无法解析): {skipped}")
    print(f"正确: {correct}")
    print(f"准确率: {accuracy:.2f}%")

    # 显示部分预测示例
    print(f"\n预测示例:")
    for i, p in enumerate(predictions[:5]):
        print(f"\n{i+1}. {'[对]' if p['correct'] else '[错]'}")
        print(f"   文本: {p['review'][:80]}...")
        print(f"   真实: {p['true']}, 预测: {p['pred']}")
        print(f"   输出: {p['raw'][:100]}...")

    # 按类别统计
    labels = {0: "负面", 1: "中性", 2: "正面"}
    for label, name in labels.items():
        label_preds = [p for p in predictions if p["true"] == label]
        if label_preds:
            label_correct = sum(p["correct"] for p in label_preds)
            label_acc = label_correct / len(label_preds) * 100
            print(f"\n{name} 类: {label_correct}/{len(label_preds)} = {label_acc:.1f}%")


if __name__ == "__main__":
    evaluate()

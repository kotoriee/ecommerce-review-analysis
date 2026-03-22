"""
使用 Unsloth FastLanguageModel 评估模型准确率
复用训练时的加载方式，避免依赖问题
"""

import json
import re
import time
from pathlib import Path
import torch
from tqdm import tqdm


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
    """构建推理prompt - 与训练格式统一"""
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
<think>


def extract_sentiment(text: str) -> int:
    """从输出提取情感标签（支持 CoT thinking 格式）"""
    text_lower = text.lower()

    # 首先尝试提取 JSON 格式
    match = re.search(r'"sentiment"\s*[:=]\s*([0-2])', text_lower)
    if match:
        return int(match.group(1))

    # 从 thinking 内容中检测关键词
    # 负面关键词
    negative_words = ['negative', '负面', '消极', '不满意', 'disappointed', 'terrible', 'bad', 'worst', 'hate', 'poor', 'awful']
    # 正面关键词
    positive_words = ['positive', '正面', '积极', '满意', 'excellent', 'great', 'good', 'amazing', 'love', 'perfect', 'wonderful']

    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)

    # 根据关键词判断
    if neg_count > pos_count:
        return 0
    elif pos_count > neg_count:
        return 2
    else:
        # 中性或其他情况
        return 1


def main():
    import argparse
    from unsloth import FastLanguageModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/curriculum/lora_s1_600",
                        help="Path to LoRA adapters")
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen3-4B-unsloth-bnb-4bit",
                        help="Base model name")
    parser.add_argument("--data", type=str, default="data/curriculum/val_fixed.json",
                        help="Validation data path")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    model_path = args.model
    data_path = args.data
    max_samples = args.max_samples
    max_new_tokens = args.max_tokens

    # 加载验证数据 (curriculum format)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 支持从指定偏移量开始
    offset = 0  # 从第1条开始评估
    test_samples = raw_data[offset:offset+max_samples]
    print(f"加载验证数据: {len(test_samples)} 条 (从第 {offset+1} 条开始)")

    # 准备数据 - 支持多种格式: curriculum/conversations, Alpaca, 旧CoT
    prompts = []
    true_labels = []
    for item in test_samples:
        review_text = None
        label = -1

        # 1. 尝试 curriculum format (conversations)
        if "conversations" in item:
            for msg in item["conversations"]:
                if msg.get("role") == "user":
                    review_match = re.search(r'Review:\s*([^<\n]+)', msg.get("content", ""))
                    if review_match:
                        review_text = review_match.group(1).strip()[:400]
                    break
            label = item.get("label", -1)

        # 2. 尝试 Alpaca format (instruction/input/output)
        elif "input" in item and "output" in item:
            review_text = item.get("input", "").strip()[:400]
            try:
                label = int(item.get("output", "-1").strip())
            except ValueError:
                label = -1

        # 3. 尝试旧CoT format
        elif "text" in item:
            label, review = extract_label_from_cot(item.get("text", ""))
            review_text = review

        # 验证并添加
        if review_text and label in [0, 1, 2]:
            prompts.append(create_prompt(review_text))
            true_labels.append(label)

    print(f"有效样本: {len(prompts)} (负={sum(1 for l in true_labels if l==0)}, 中={sum(1 for l in true_labels if l==1)}, 正={sum(1 for l in true_labels if l==2)})")

    print(f"有效样本: {len(prompts)}")

    # 使用Unsloth加载模型 - base model + LoRA
    print(f"\n加载基础模型: {args.base_model}")
    print(f"加载LoRA适配器: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=512,
        load_in_4bit=True,
        load_in_8bit=False,
    )
    # 加载LoRA权重
    model = FastLanguageModel.get_peft_model(model)
    model.load_adapter(model_path, adapter_name="default")
    model.set_adapter("default")
    FastLanguageModel.for_inference(model)  # 启用推理优化

    print(f"模型已加载到: {next(model.parameters()).device}")

    # 单条推理（Unsloth优化的generate）
    print(f"\n开始推理 (max_tokens={max_new_tokens})...")
    start_time = time.time()

    all_preds = []
    for prompt in tqdm(prompts, desc="Evaluating"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        all_preds.append(response)

    infer_time = time.time() - start_time
    speed = len(all_preds) / infer_time

    print(f"\n推理完成: {infer_time:.1f}s, {speed:.2f} 条/秒 ({speed*60:.0f} 条/分钟)")

    # 评估准确率
    correct = 0
    valid = 0
    # 分类统计
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # true x pred

    for true, pred_text in zip(true_labels, all_preds):
        pred = extract_sentiment(pred_text)
        if pred != -1:
            valid += 1
            class_total[true] += 1
            confusion[true][pred] += 1
            if pred == true:
                correct += 1
                class_correct[true] += 1

    accuracy = correct / valid * 100 if valid > 0 else 0

    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"总样本: {len(all_preds)}")
    print(f"解析成功: {valid}")
    print(f"正确: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"推理速度: {speed:.2f} 条/秒")

    # 类别分布
    print(f"\n{'='*60}")
    print(f"类别分布与准确率")
    print(f"{'='*60}")
    label_names = {0: "负面(0)", 1: "中性(1)", 2: "正面(2)"}
    for label in [0, 1, 2]:
        acc = class_correct[label] / class_total[label] * 100 if class_total[label] > 0 else 0
        print(f"{label_names[label]}: 总数={class_total[label]}, 正确={class_correct[label]}, 准确率={acc:.1f}%")

    # 混淆矩阵
    print(f"\n{'='*60}")
    print(f"混淆矩阵 (真实 x 预测)")
    print(f"{'='*60}")
    print("          预:0    预:1    预:2")
    for true_label in [0, 1, 2]:
        row = confusion[true_label]
        print(f"真:{true_label}     {row[0]:4d}    {row[1]:4d}    {row[2]:4d}")

    # 显示示例
    print(f"\n预测示例:")
    for i in range(min(3, len(all_preds))):
        print(f"\n{i+1}. 真实={true_labels[i]}")
        print(f"   输出: {all_preds[i][:100]}...")


if __name__ == "__main__":
    main()

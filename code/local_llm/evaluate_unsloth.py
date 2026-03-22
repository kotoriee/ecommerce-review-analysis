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
<thinking>
"""


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
    from unsloth import FastLanguageModel

    model_path = "models/qwen3-4b-sentiment-lora_merged_16bit"
    data_path = "data/processed/test.json"
    max_samples = 1000  # 评估1000条
    max_new_tokens = 60

    # 加载测试数据
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 支持从指定偏移量开始
    offset = 0  # 从第1条开始评估
    test_samples = raw_data[offset:offset+max_samples]
    print(f"加载测试数据: {len(test_samples)} 条 (从第 {offset+1} 条开始)"),

    # 准备数据
    prompts = []
    true_labels = []
    for item in test_samples:
        label, review = extract_label_from_cot(item["text"])
        if label != -1:
            prompts.append(create_prompt(review))
            true_labels.append(label)

    print(f"有效样本: {len(prompts)}")

    # 使用Unsloth加载模型（与训练时相同）
    print(f"\n加载模型: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        load_in_4bit=True,  # 4bit节省显存，避免offload
        load_in_8bit=False,
    )
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

    # 显示示例
    print(f"\n预测示例:")
    for i in range(min(3, len(all_preds))):
        print(f"\n{i+1}. 真实={true_labels[i]}")
        print(f"   输出: {all_preds[i][:100]}...")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
数据格式化脚本：将 CoT JSONL 转换为 Unsloth 训练格式

Input:  data/processed/en_cot_2000.jsonl  (generate_cot_data.py 输出)
Output: data/processed/train.json / val.json / test.json (Unsloth SFT 格式)

Usage:
    python data_formatter.py
    python data_formatter.py --input data/processed/en_cot_2000.jsonl --split 70 15 15
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple

# ============== 系统提示词 ==============
# 三分类情感分析 (0=negative, 1=neutral, 2=positive)

SYSTEM_PROMPT = """You are a sentiment analysis expert for e-commerce reviews.
Analyze the review and output ONLY a JSON object with these exact fields:
- sentiment: 0 (negative), 1 (neutral), or 2 (positive)
- confidence: float between 0 and 1
- rationale: brief explanation in English

Output format: {"sentiment": X, "confidence": Y.YZ, "rationale": "explanation"}"""


# ============== 格式转换 ==============

def record_to_conversation(record: Dict) -> Dict:
    """
    将 CoT 记录转换为 Qwen3 ChatML 对话格式

    assistant 内容：<think>...</think> + JSON答案
    """
    text = record.get("text", "")
    label = record.get("predicted_label", record.get("ground_truth_label", 1))
    confidence = record.get("confidence", 0.8)
    rationale = record.get("rationale", "Sentiment detected from review context.")
    cot = record.get("cot", "").strip()

    # 构建 <think> 块（若无CoT则生成简短版本）
    if cot:
        think_content = cot[:1000]  # 限制长度
    else:
        # 三分类映射
        sentiment_words = {0: "negative", 1: "neutral", 2: "positive"}
        think_content = (
            f"Let me analyze this review.\n"
            f"Key observation: {rationale}\n"
            f"Conclusion: {sentiment_words.get(label, 'neutral')} sentiment."
        )

    assistant_content = (
        f"<think>\n{think_content}\n</think>\n"
        f'{{"sentiment": {label}, "confidence": {confidence:.2f}, "rationale": "{rationale}"}}'
    )

    return {
        "conversations": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Review: {text}"},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def apply_chat_template_manual(conv: Dict) -> str:
    """
    手动应用 Qwen3 ChatML 模板（不依赖 tokenizer，便于离线预处理）

    格式：
    <|im_start|>system\n...<|im_end|>\n
    <|im_start|>user\n...<|im_end|>\n
    <|im_start|>assistant\n...<|im_end|>\n
    """
    parts = []
    for msg in conv["conversations"]:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


# ============== 数据划分 ==============

def split_dataset(
    records: List[Dict],
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[List, List, List]:
    """分层划分：保持标签分布"""
    from collections import defaultdict

    buckets: Dict[int, List] = defaultdict(list)
    for r in records:
        label = r.get("predicted_label", r.get("ground_truth_label", 1))
        buckets[label].append(r)

    train, val, test = [], [], []
    for label_records in buckets.values():
        random.shuffle(label_records)
        n = len(label_records)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        train.extend(label_records[:n_train])
        val.extend(label_records[n_train: n_train + n_val])
        test.extend(label_records[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description="CoT JSONL → Unsloth 训练格式")
    parser.add_argument(
        "--input", type=str,
        default="data/processed/en_cot_2000.jsonl",
        help="输入 CoT JSONL 文件",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/processed",
        help="输出目录",
    )
    parser.add_argument(
        "--split", type=float, nargs=3,
        default=[70, 15, 15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="划分比例 (default: 70 15 15)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        print("请先运行: python generate_cot_data.py --from-hf --count 2000")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"加载 {len(records)} 条记录")

    # 过滤低置信度
    before = len(records)
    records = [r for r in records if r.get("confidence", 0) >= 0.65]
    print(f"过滤后: {len(records)} 条（去掉 {before - len(records)} 条低置信度）")

    # 划分
    total = sum(args.split)
    train_pct = args.split[0] / total
    val_pct = args.split[1] / total
    train, val, test = split_dataset(records, train_pct, val_pct)
    print(f"划分: train={len(train)}, val={len(val)}, test={len(test)}")

    # 转换并保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        # 格式1：conversations JSON（供 Unsloth apply_chat_template 使用）
        conv_path = output_dir / f"{split_name}_conversations.json"
        conversations = [record_to_conversation(r) for r in split_data]
        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)

        # 格式2：已应用模板的纯文本 JSON（dataset_text_field="text"）
        text_path = output_dir / f"{split_name}.json"
        texts = [{"text": apply_chat_template_manual(c)} for c in conversations]
        with open(text_path, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)

        print(f"  {split_name}: {len(split_data)} 条 → {text_path}")

    # 标签分布统计
    from collections import Counter
    all_labels = [r.get("predicted_label", r.get("ground_truth_label", 1)) for r in records]
    dist = Counter(all_labels)
    print(f"\n全集标签分布:")
    print(f"  Negative (0): {dist.get(0, 0)}")
    print(f"  Neutral  (1): {dist.get(1, 0)}")
    print(f"  Positive (2): {dist.get(2, 0)}")

    print("\n格式化完成！可直接用于 train_sentiment.py 训练。")


if __name__ == "__main__":
    main()

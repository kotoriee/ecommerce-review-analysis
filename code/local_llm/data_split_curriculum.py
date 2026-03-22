#!/usr/bin/env python3
"""
阶梯式微调数据分割脚本
将 ~8500 条软标注数据分层采样为 600/1200/2400/4800/8500
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from local_llm.prompt_templates import get_system_prompt, get_sentiment_prompt

# 固定随机种子确保可复现
random.seed(42)


def load_soft_labels(path: str):
    """加载软标注数据"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def get_soft_label(item: dict) -> int:
    """获取软标签（三分类）

    支持两种格式:
    - soft_label_probs: {"negative": x, "positive": y, "confidence": z}
    - probabilities: [neg, neu, pos] 数组
    - hard_label: 直接返回

    软标签概率分布 -> 硬标签:
    - 0 (负面): P(negative) > 0.6
    - 2 (正面): P(positive) > 0.6
    - 1 (中性): 其他情况
    """
    # 优先使用已有的 hard_label
    if "hard_label" in item:
        return item["hard_label"]

    # 支持 probabilities 数组格式 [neg, neu, pos]
    if "probabilities" in item:
        probs = item["probabilities"]
        neg_prob, neu_prob, pos_prob = probs[0], probs[1], probs[2]

        if neg_prob > 0.6:
            return 0
        elif pos_prob > 0.6:
            return 2
        else:
            return 1

    # 支持 soft_label_probs 字典格式
    probs = item.get("soft_label_probs", {})
    neg_prob = probs.get("negative", 0.33)
    pos_prob = probs.get("positive", 0.33)

    if neg_prob > 0.6:
        return 0
    elif pos_prob > 0.6:
        return 2
    else:
        return 1

def format_for_training_qwen3(item: dict, label_fn=None) -> dict:
    """转换为 Qwen3 ChatML 训练格式 - 与推理统一"""
    if label_fn is None:
        label_fn = get_soft_label

    text = item.get("text", item.get("review", ""))
    label = label_fn(item)

    # 构建对话格式
    system_msg = get_system_prompt("en")  # 使用英文prompt
    user_msg = get_sentiment_prompt(text, "en")

    # assistant输出 - 包含JSON结果
    confidence = item.get("soft_label_probs", {}).get("confidence", 0.8)
    rationale = item.get("rationale", "Sentiment detected from review context.")

    assistant_content = f'''<think>
Let me analyze this review.
Key observation: {rationale}
Conclusion: {"negative" if label==0 else "neutral" if label==1 else "positive"} sentiment.
</think>
{{"sentiment": {label}, "confidence": {confidence:.2f}, "rationale": "{rationale}"}}'''

    return {
        "conversations": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_content}
        ],
        "text": item.get("text", ""),
        "label": label
    }


def sample_balanced(data: list, n_total: int, label_fn=get_soft_label) -> list:
    """分层均衡采样"""
    # 按标签分组
    by_label = {0: [], 1: [], 2: []}
    for item in data:
        label = label_fn(item)
        by_label[label].append(item)

    print(f"  原始分布: 负={len(by_label[0])}, 中={len(by_label[1])}, 正={len(by_label[2])}")

    # 每类目标数量
    n_per_class = n_total // 3

    samples = []
    for label in [0, 1, 2]:
        pool = by_label[label]
        if len(pool) >= n_per_class:
            samples.extend(random.sample(pool, n_per_class))
        else:
            # 如果某类不足，全取并用其他类补足
            samples.extend(pool)
            print(f"  警告: 类别 {label} 只有 {len(pool)} 条，不足 {n_per_class}")

    # 补足剩余 (从全量随机采)
    while len(samples) < n_total:
        samples.append(random.choice(data))

    random.shuffle(samples)
    return samples[:n_total]

def main():
    # 配置
    input_file = "data/processed/soft_labels_raw.jsonl"
    output_dir = Path("data/curriculum")
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        (600, "train_600.json"),
        (1200, "train_1200.json"),
        (2400, "train_2400.json"),
        (4800, "train_4800.json"),
        (8500, "train_full.json"),  # 实际可能是 8400-8600
    ]

    # 检查输入文件
    if not Path(input_file).exists():
        print(f"错误: 找不到软标注数据 {input_file}")
        print("请先运行: python code/cloud_agent/run_3cls_annotation.py")
        return

    # 加载全部数据
    print(f"加载软标注数据: {input_file}")
    all_data = load_soft_labels(input_file)
    print(f"总数据量: {len(all_data)} 条")

    # 统计标签分布
    labels = [get_soft_label(item) for item in all_data]
    print(f"标签分布: {Counter(labels)}")

    # 生成各阶段数据
    print("\n" + "="*50)
    print("生成阶梯式训练数据")
    print("="*50)

    for n_total, filename in stages:
        print(f"\n阶段: {n_total} 条")

        # 采样
        samples = sample_balanced(all_data, n_total)

        # 转换为训练格式
        formatted = [format_for_training_qwen3(s) for s in samples]

        # 检查分布
        labels_in_sample = [s["label"] for s in formatted]
        print(f"  采样分布: {Counter(labels_in_sample)}")

        # 保存
        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 已保存: {output_path}")

    # 同时复制验证集
    val_source = "data/processed/val_3cls.json"
    val_target = output_dir / "val_fixed.json"
    if Path(val_source).exists():
        import shutil
        shutil.copy(val_source, val_target)
        print(f"\n✓ 验证集已复制: {val_target}")

    print("\n" + "="*50)
    print("数据准备完成！")
    print(f"输出目录: {output_dir}")
    print("\n下一步: bash code/local_llm/run_curriculum.sh")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM 基线推理脚本 — Phase 5 Route A

从 *_3cls.json（run_3cls_annotation.py 输出格式）加载数据，
训练 SVM + TF-IDF 分类器，在测试集上生成预测，
输出与 api_eval_sentiment.py 格式一致的 JSONL 文件。

Usage:
    python evaluation/run_svm_baseline.py \
        --train data/processed/train_3cls.json \
        --test  data/processed/test_3cls.json \
        --output data/predictions/svm_predictions.jsonl
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# 复用项目现有 SVM 分类器
sys.path.insert(0, str(Path(__file__).parent.parent))
from baseline.svm_classifier import SVMSentimentClassifier
from evaluation.metrics import compute_metrics, print_report


LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_3cls_json(path: str) -> Tuple[List[str], List[int]]:
    """
    从 *_3cls.json（JSON array 格式）提取 (texts, labels)。

    输入格式（run_3cls_annotation.py 输出）：
    [{"instruction": "...", "input": "review text", "output": "1", ...}]

    Returns:
        texts: List[str] — 评论文本（来自 input 字段）
        labels: List[int] — 硬标签（来自 output 字段，转为 int）
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for record in data:
        texts.append(record["input"])
        labels.append(int(record["output"]))
    return texts, labels


# ─── 预测格式化 ───────────────────────────────────────────────────────────────

def format_predictions(
    texts: List[str],
    y_pred: np.ndarray,
    y_true: Optional[List[int]] = None,
) -> List[dict]:
    """
    将预测结果格式化为统一的 JSONL 格式（对齐 api_eval_sentiment.py）。

    Output per record:
        {"text": "...", "sentiment": 1, "label": "neutral", "true_label": 1}
    """
    records = []
    for i, (text, pred) in enumerate(zip(texts, y_pred)):
        record = {
            "text": text,
            "sentiment": int(pred),
            "label": LABEL_MAP[int(pred)],
            "true_label": int(y_true[i]) if y_true is not None else None,
        }
        records.append(record)
    return records


# ─── 主推理流程 ───────────────────────────────────────────────────────────────

def run_svm(
    train_path: str,
    test_path: str,
    output_path: str,
    model_save_path: Optional[str] = None,
) -> dict:
    """
    完整的 SVM 训练 → 预测 → 保存流程。

    Args:
        train_path: train_3cls.json 路径
        test_path:  test_3cls.json 路径
        output_path: 输出 JSONL 路径
        model_save_path: （可选）保存训练好的 SVM 模型 .pkl 路径

    Returns:
        metrics dict from compute_metrics()
    """
    # 加载数据
    print(f"Loading training data: {train_path}")
    train_texts, train_labels = load_3cls_json(train_path)
    print(f"  {len(train_texts)} training samples")

    print(f"Loading test data: {test_path}")
    test_texts, test_labels = load_3cls_json(test_path)
    print(f"  {len(test_texts)} test samples")

    # 训练
    print("Training SVM classifier...")
    clf = SVMSentimentClassifier()
    clf.fit(train_texts, train_labels)
    print("  Training complete")

    # 预测
    y_pred = clf.predict(test_texts)

    # 保存模型（可选）
    if model_save_path:
        clf.save(model_save_path)
        print(f"Model saved: {model_save_path}")

    # 保存预测 JSONL
    records = format_predictions(test_texts, y_pred, test_labels)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions saved: {output_path} ({len(records)} records)")

    # 计算并打印指标
    metrics = compute_metrics(test_labels, [int(p) for p in y_pred])
    print_report(metrics, "SVM + TF-IDF Baseline")

    return metrics


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SVM baseline training and evaluation for 3-class sentiment"
    )
    parser.add_argument(
        "--train", required=True,
        help="Path to train_3cls.json (JSON array with input/output fields)"
    )
    parser.add_argument(
        "--test", required=True,
        help="Path to test_3cls.json (JSON array with input/output fields)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL path for predictions"
    )
    parser.add_argument(
        "--save-model", default=None, dest="save_model",
        help="Optional path to save trained SVM model (.pkl)"
    )
    args = parser.parse_args()

    run_svm(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        model_save_path=args.save_model,
    )


if __name__ == "__main__":
    main()

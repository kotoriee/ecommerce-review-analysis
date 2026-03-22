#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三路模型对比框架 — Phase 5

汇总 SVM / Qwen3-4B / DeepSeek API 三路预测结果，
调用 metrics.py 计算指标，保存 comparison_results.json。

Usage:
    python evaluation/run_comparison.py \
        --svm  data/predictions/svm_predictions.jsonl \
        --api  data/predictions/api_predictions.jsonl \
        --qwen data/predictions/qwen_predictions.jsonl \
        --output reports/
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import compute_metrics, load_predictions, print_report


# ─── 预测加载 ─────────────────────────────────────────────────────────────────

def load_all_predictions(pred_paths: Dict[str, Optional[str]]) -> Dict[str, Dict]:
    """
    按路线加载预测 JSONL。

    Args:
        pred_paths: {"svm": path, "api": path, "qwen": path}
                    None 值表示该路线不可用，跳过。

    Returns:
        {route_name: {"y_pred": [...], "y_true": [...]}}
    """
    result = {}
    for route, path in pred_paths.items():
        if not path:
            continue
        records = load_predictions(path)
        y_pred = [r.get("predicted_label", r.get("sentiment", 1)) for r in records]
        y_true = [r.get("true_label", r.get("ground_truth_label", 1)) for r in records]
        result[route] = {"y_pred": y_pred, "y_true": y_true}
    return result


# ─── 三路对比 ─────────────────────────────────────────────────────────────────

ROUTE_DISPLAY = {
    "svm":  "SVM + TF-IDF",
    "qwen": "Qwen3-4B (Fine-tuned)",
    "api":  "DeepSeek API",
}


def run_comparison(
    pred_paths: Dict[str, Optional[str]],
    output_dir: str,
) -> Dict[str, Dict]:
    """
    对每个可用路线计算 F1/准确率/混淆矩阵，保存汇总 JSON。

    Returns:
        {route: metrics_dict}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_data = load_all_predictions(pred_paths)
    all_metrics = {}

    for route, data in all_data.items():
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        metrics = compute_metrics(y_true, y_pred)
        all_metrics[route] = metrics
        display_name = ROUTE_DISPLAY.get(route, route)
        print_report(metrics, display_name)

    # 保存机器可读结果
    out_path = Path(output_dir) / "comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nComparison results saved: {out_path}")

    # 打印 F1-macro 汇总
    if len(all_metrics) > 1:
        print("\n  F1-macro Summary:")
        for route, m in all_metrics.items():
            print(f"    {ROUTE_DISPLAY.get(route, route):25s}: {m['f1_macro']:.4f}")
        best = max(all_metrics, key=lambda r: all_metrics[r]["f1_macro"])
        print(f"\n  Best model: {ROUTE_DISPLAY.get(best, best)} (F1={all_metrics[best]['f1_macro']:.4f})")

    return all_metrics


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Three-way model comparison for 3-class sentiment analysis"
    )
    parser.add_argument("--svm",  default=None,
                        help="SVM predictions JSONL")
    parser.add_argument("--api",  default=None,
                        help="API (DeepSeek/Qwen) predictions JSONL")
    parser.add_argument("--qwen", default=None,
                        help="Qwen3-4B local predictions JSONL (optional)")
    parser.add_argument("--output", default="reports/",
                        help="Output directory for comparison_results.json")
    args = parser.parse_args()

    pred_paths = {
        "svm":  args.svm,
        "api":  args.api,
        "qwen": args.qwen,
    }
    # Remove routes that were not provided
    pred_paths = {k: v for k, v in pred_paths.items() if v is not None}

    if not pred_paths:
        print("Error: at least one prediction file must be provided (--svm, --api, or --qwen)")
        sys.exit(1)

    run_comparison(pred_paths, output_dir=args.output)


if __name__ == "__main__":
    main()

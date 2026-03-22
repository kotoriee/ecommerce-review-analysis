#!/usr/bin/env python3
"""
汇总阶梯式训练结果，生成对比表格和图表
"""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

STAGES = [600, 1200, 2400, 4800, 8500]
RESULTS = []

def parse_eval_log(log_path: Path):
    """从评估日志解析准确率"""
    if not log_path.exists():
        return None, None

    content = log_path.read_text(encoding='utf-8')

    # 匹配准确率
    acc_match = re.search(r'准确率[:\s]+(\d+\.?\d*)%', content)
    accuracy = float(acc_match.group(1)) / 100 if acc_match else 0

    # 匹配速度
    speed_match = re.search(r'速度[:\s]+(\d+\.?\d*)\s*条/秒', content)
    speed = float(speed_match.group(1)) if speed_match else 0

    return accuracy, speed

def main():
    results_dir = Path("results/curriculum")

    print("=" * 70)
    print("阶梯式微调结果汇总")
    print("=" * 70)
    print()

    # 收集各阶段结果
    for n in STAGES:
        log_file = results_dir / f"eval_{n}.log"
        accuracy, speed = parse_eval_log(log_file)

        RESULTS.append({
            "stage": n,
            "accuracy": accuracy,
            "speed": speed,
            "log": str(log_file) if log_file.exists() else "N/A"
        })

    # 打印表格
    print(f"{'阶段':>6} | {'数据量':>8} | {'准确率':>10} | {'推理速度':>10} | {'边际提升':>10}")
    print("-" * 70)

    prev_acc = None
    for i, r in enumerate(RESULTS):
        delta = ""
        if prev_acc is not None and r["accuracy"] > 0:
            d = r["accuracy"] - prev_acc
            delta = f"+{d:.1%}"
        prev_acc = r["accuracy"] if r["accuracy"] > 0 else prev_acc

        status = "✓" if r["accuracy"] > 0 else "✗"
        print(f"S{i+1}   | {r['stage']:>8} | {r['accuracy']:>9.1%} | {r['speed']:>9.1f} | {delta:>10}")

    # 保存 JSON
    summary_path = results_dir / "curriculum_summary.json"
    with open(summary_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n✓ 结果已保存: {summary_path}")

    # 生成图表
    if any(r["accuracy"] > 0 for r in RESULTS):
        plot_results(RESULTS, results_dir)

def plot_results(results, output_dir):
    """绘制学习曲线"""
    valid = [r for r in results if r["accuracy"] > 0]
    if len(valid) < 2:
        return

    x = [r["stage"] for r in valid]
    y = [r["accuracy"] * 100 for r in valid]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Training Samples', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Qwen3-4B Curriculum Training: Accuracy vs Data Size', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 标注每个点
    for xi, yi in zip(x, y):
        plt.annotate(f'{yi:.1f}%', (xi, yi), textcoords="offset points",
                     xytext=(0,10), ha='center', fontsize=10)

    plt.tight_layout()
    chart_path = output_dir / "curriculum_curve.png"
    plt.savefig(chart_path, dpi=150)
    print(f"✓ 图表已保存: {chart_path}")

if __name__ == "__main__":
    main()

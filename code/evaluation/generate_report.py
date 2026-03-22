#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成器 — Phase 5

从 comparison_results.json 生成：
  - reports/summary.md       (Markdown 对比表)
  - reports/summary_latex.tex (LaTeX tabular，论文直接插入)

Usage:
    python evaluation/generate_report.py \
        --results reports/comparison_results.json \
        --output  reports/
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict


ROUTE_DISPLAY = {
    "svm":  "SVM+TF-IDF",
    "qwen": "Qwen3-4B",
    "api":  "DeepSeek API",
}

SPEED_NOTE = {
    "svm":  "\\textasciitilde{}1200/s",
    "qwen": "\\textasciitilde{}5/s",
    "api":  "\\textasciitilde{}15/s",
}

COST_NOTE = {
    "svm":  "Free",
    "qwen": "Local GPU",
    "api":  "\\textasciitilde{}\\yen 0.7/1K",
}


# ─── Markdown 报告 ────────────────────────────────────────────────────────────

def generate_markdown_table(results: Dict) -> str:
    """生成 Markdown 对比表格。"""
    routes = list(results.keys())
    headers = ["Metric"] + [ROUTE_DISPLAY.get(r, r) for r in routes]
    sep = [":---"] + ["---:"] * len(routes)

    def fmt(val: float) -> str:
        return f"{val:.4f}"

    def row(label, extractor):
        cells = [label] + [extractor(results[r]) for r in routes]
        return "| " + " | ".join(cells) + " |"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
        row("Accuracy",    lambda m: f"{m['accuracy'] * 100:.1f}%"),
        row("F1-macro",    lambda m: fmt(m["f1_macro"])),
        row("F1-negative", lambda m: fmt(m["per_class"]["Negative"]["f1"])),
        row("F1-neutral",  lambda m: fmt(m["per_class"]["Neutral"]["f1"])),
        row("F1-positive", lambda m: fmt(m["per_class"]["Positive"]["f1"])),
        row("Prec-macro",  lambda m: fmt(
            sum(m["per_class"][c]["precision"] for c in ["Negative","Neutral","Positive"]) / 3
        )),
        row("Recall-macro", lambda m: fmt(
            sum(m["per_class"][c]["recall"] for c in ["Negative","Neutral","Positive"]) / 3
        )),
        row("Samples",     lambda m: str(m["total_samples"])),
    ]

    best_route = max(routes, key=lambda r: results[r]["f1_macro"])
    lines.append("")
    lines.append(f"**Best F1-macro**: {ROUTE_DISPLAY.get(best_route, best_route)} "
                 f"({results[best_route]['f1_macro']:.4f})")

    return "\n".join(lines)


def generate_markdown_report(results: Dict) -> str:
    """生成完整的 Markdown 报告。"""
    lines = [
        "# Sentiment Analysis Evaluation — Three-Way Comparison",
        "",
        "## Model Performance Summary",
        "",
        generate_markdown_table(results),
        "",
        "## Per-Class F1 Details",
        "",
    ]

    for route, m in results.items():
        name = ROUTE_DISPLAY.get(route, route)
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Class    | Precision | Recall | F1    | Support |")
        lines.append("|----------|-----------|--------|-------|---------|")
        for cls in ["Negative", "Neutral", "Positive"]:
            pc = m["per_class"][cls]
            lines.append(
                f"| {cls:8s} | {pc['precision']:.4f}    | "
                f"{pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']:7d} |"
            )
        lines.append("")

    return "\n".join(lines)


# ─── LaTeX 报告 ───────────────────────────────────────────────────────────────

def generate_latex_table(results: Dict) -> str:
    """
    生成可直接插入论文的 LaTeX tabular 表格。
    无外部包依赖，仅使用标准 tabular 环境。
    """
    routes = list(results.keys())
    col_headers = [ROUTE_DISPLAY.get(r, r) for r in routes]
    n_cols = len(routes)
    col_spec = "l" + "r" * n_cols

    def fmt(val: float) -> str:
        return f"{val:.4f}"

    def row(label, values):
        return f"    {label} & " + " & ".join(values) + r" \\"

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Comparison of Sentiment Analysis Methods on Amazon Reviews}",
        r"  \label{tab:sentiment_comparison}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \hline",
        "    \\textbf{Metric} & " + " & ".join(
            f"\\textbf{{{h}}}" for h in col_headers
        ) + r" \\",
        r"    \hline",
        row("Accuracy (\\%)",
            [f"{results[r]['accuracy']*100:.1f}" for r in routes]),
        row("F1-macro",
            [fmt(results[r]["f1_macro"]) for r in routes]),
        r"    \hline",
        row("F1-Negative",
            [fmt(results[r]["per_class"]["Negative"]["f1"]) for r in routes]),
        row("F1-Neutral",
            [fmt(results[r]["per_class"]["Neutral"]["f1"]) for r in routes]),
        row("F1-Positive",
            [fmt(results[r]["per_class"]["Positive"]["f1"]) for r in routes]),
        r"    \hline",
        row("Precision-macro",
            [fmt(sum(results[r]["per_class"][c]["precision"]
                     for c in ["Negative","Neutral","Positive"]) / 3)
             for r in routes]),
        row("Recall-macro",
            [fmt(sum(results[r]["per_class"][c]["recall"]
                     for c in ["Negative","Neutral","Positive"]) / 3)
             for r in routes]),
        r"    \hline",
        row("Speed",
            [SPEED_NOTE.get(r, "---") for r in routes]),
        row("Cost",
            [COST_NOTE.get(r, "---") for r in routes]),
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─── 保存报告 ─────────────────────────────────────────────────────────────────

def save_reports(results: Dict, output_dir: str):
    """保存 summary.md 和 summary_latex.tex 到指定目录。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    md_path = Path(output_dir) / "summary.md"
    md_content = generate_markdown_report(results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved: {md_path}")

    tex_path = Path(output_dir) / "summary_latex.tex"
    tex_content = generate_latex_table(results)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"Saved: {tex_path}")


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown and LaTeX reports from comparison results"
    )
    parser.add_argument(
        "--results", default="reports/comparison_results.json",
        help="Path to comparison_results.json (from run_comparison.py)"
    )
    parser.add_argument(
        "--output", default="reports/",
        help="Output directory for summary.md and summary_latex.tex"
    )
    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"Error: results file not found: {args.results}")
        print("Run run_comparison.py first to generate comparison_results.json")
        sys.exit(1)

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    save_reports(results, args.output)


if __name__ == "__main__":
    main()

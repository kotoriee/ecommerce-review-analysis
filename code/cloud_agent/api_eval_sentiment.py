#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 情感评估脚本 - review-reporter 专用
用于三路模型对比中的 API 路线（DeepSeek / Qwen）

特点：
- 零样本直接三分类（0/1/2），无 CoT，成本最低
- 可选 few-shot 模式提升 neutral 类召回
- 复用 generate_soft_labels.py 的 API 配置
- 速率限制保护（默认 10 req/s）
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI

# ─── 模型配置 ────────────────────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "deepseek": "Pro/deepseek-ai/DeepSeek-V3.2",
    "deepseek-r1": "Pro/deepseek-ai/DeepSeek-R1",
    "qwen": "Qwen/Qwen3-32B",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}

# ─── 提示词 ──────────────────────────────────────────────────────────────────

# 零样本系统提示（经济模式，最低 token 消耗）
SYSTEM_PROMPT_ZEROSHOT = (
    "You are a sentiment classifier for e-commerce product reviews. "
    "Read the review and output ONLY a single digit:\n"
    "  0 = negative (complaints, dissatisfied, bad quality, returns)\n"
    "  1 = neutral  (mixed feelings, minor issues, average, ok)\n"
    "  2 = positive (satisfied, happy, recommend, great quality)\n\n"
    "Output the digit only. No explanation. No punctuation. No spaces."
)

# Few-shot 示例（提升 neutral 类识别）
FEW_SHOT_EXAMPLES = [
    {
        "review": "Absolutely love this product! Works perfectly and arrived fast.",
        "label": "2"
    },
    {
        "review": "Complete waste of money. Broke after two days. Would not recommend.",
        "label": "0"
    },
    {
        "review": "The color looks nice but the quality feels a bit cheap for the price.",
        "label": "1"
    },
]

def build_messages_zeroshot(text: str) -> list:
    """零样本模式：系统提示 + 用户评论"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_ZEROSHOT},
        {"role": "user", "content": f"Review: {text}"},
    ]


def build_messages_fewshot(text: str) -> list:
    """Few-shot 模式：系统提示 + 3 个示例 + 用户评论"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT_ZEROSHOT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Review: {ex['review']}"})
        messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": f"Review: {text}"})
    return messages


# ─── API 客户端 ───────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """获取 SiliconFlow API 客户端（复用 generate_soft_labels.py 的配置）"""
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")

    if not api_key:
        config_path = Path(__file__).parent.parent.parent / "config" / "api_keys.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
                api_key = config.get("siliconflow", {}).get("api_key", "")

    if not api_key:
        raise ValueError(
            "API key not found. Set SILICONFLOW_API_KEY env var or add to config/api_keys.json:\n"
            '  {"siliconflow": {"api_key": "your-key"}}'
        )

    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")


# ─── 单条推理 ─────────────────────────────────────────────────────────────────

def predict_one(
    client: OpenAI,
    text: str,
    model: str,
    use_fewshot: bool = False,
    max_retries: int = 3,
) -> dict:
    """
    单条评论情感分类

    Returns:
        {"text": str, "sentiment": int, "label": str, "raw_response": str}
    """
    messages = build_messages_fewshot(text) if use_fewshot else build_messages_zeroshot(text)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,   # deterministic
                max_tokens=3,      # only need "0", "1", or "2"
            )
            raw = response.choices[0].message.content.strip()
            sentiment = int(raw[0])  # take first character
            if sentiment not in (0, 1, 2):
                raise ValueError(f"Unexpected label: {raw!r}")
            return {
                "text": text,
                "sentiment": sentiment,
                "label": label_map[sentiment],
                "raw_response": raw,
            }
        except (ValueError, IndexError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            # Fallback: return neutral on parse failure
            return {
                "text": text,
                "sentiment": 1,
                "label": "neutral",
                "raw_response": f"PARSE_ERROR: {e}",
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            raise


# ─── 批量推理 ─────────────────────────────────────────────────────────────────

def predict_batch(
    client: OpenAI,
    records: list,
    model: str,
    use_fewshot: bool = False,
    rate_limit: float = 10.0,
) -> list:
    """
    批量推理，带速率限制

    Args:
        rate_limit: max requests per second (default 10)
    """
    results = []
    interval = 1.0 / rate_limit
    parse_errors = 0

    for record in tqdm(records, desc=f"API inference ({model.split('/')[-1]})"):
        text = record.get("input") or record.get("llm_text") or record.get("text", "")
        if not text:
            continue

        result = predict_one(client, text, model, use_fewshot)
        result["true_label"] = record.get("output") or record.get("label")  # ground truth if available
        results.append(result)

        if "PARSE_ERROR" in result.get("raw_response", ""):
            parse_errors += 1

        time.sleep(interval)

    if parse_errors > 0:
        print(f"  ⚠ Parse errors: {parse_errors}/{len(results)} ({parse_errors/len(results)*100:.1f}%)")

    return results


# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_records(input_path: str, max_n: int = None) -> list:
    """加载 JSON 或 JSONL 格式的评论数据"""
    path = Path(input_path)
    records = []

    with open(path, encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            # JSON: either list or newline-delimited
            content = f.read().strip()
            if content.startswith("["):
                records = json.loads(content)
            else:
                for line in content.splitlines():
                    if line.strip():
                        records.append(json.loads(line))

    if max_n:
        records = records[:max_n]

    return records


def save_results(results: list, output_path: str):
    """保存预测结果为 JSONL"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─── 结果统计 ─────────────────────────────────────────────────────────────────

def print_summary(results: list, model_name: str):
    """打印推理结果摘要"""
    total = len(results)
    counts = {0: 0, 1: 0, 2: 0}
    correct = 0
    has_gt = False

    for r in results:
        counts[r["sentiment"]] += 1
        if r.get("true_label") is not None:
            has_gt = True
            try:
                if int(r["true_label"]) == r["sentiment"]:
                    correct += 1
            except (ValueError, TypeError):
                pass

    print(f"\n{'─'*50}")
    print(f"API Inference Summary — {model_name}")
    print(f"{'─'*50}")
    print(f"Total samples: {total}")
    print(f"  Negative (0): {counts[0]:5d}  ({counts[0]/total*100:.1f}%)")
    print(f"  Neutral  (1): {counts[1]:5d}  ({counts[1]/total*100:.1f}%)")
    print(f"  Positive (2): {counts[2]:5d}  ({counts[2]/total*100:.1f}%)")
    if has_gt:
        print(f"Accuracy (vs ground truth): {correct/total*100:.1f}%")
    print(f"{'─'*50}\n")


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="API-based 3-class sentiment evaluation for e-commerce reviews"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input file path (JSON or JSONL, with 'input'/'text' field)"
    )
    parser.add_argument(
        "--model", default="deepseek",
        choices=list(AVAILABLE_MODELS.keys()),
        help=f"Model to use. Options: {', '.join(AVAILABLE_MODELS.keys())} (default: deepseek)"
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Max number of samples to process (default: all)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL path (default: data/predictions/api_{model}_{timestamp}.jsonl)"
    )
    parser.add_argument(
        "--fewshot", action="store_true",
        help="Use few-shot examples (better neutral recall, slightly higher cost)"
    )
    parser.add_argument(
        "--rate-limit", type=float, default=10.0,
        help="Max API requests per second (default: 10)"
    )
    args = parser.parse_args()

    # Resolve model name
    model_id = AVAILABLE_MODELS[args.model]

    # Default output path
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = Path(__file__).parent.parent.parent
        args.output = str(project_root / "data" / "predictions" / f"api_{args.model}_{ts}.jsonl")

    print(f"Model:      {model_id}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Max samples:{args.n or 'all'}")
    print(f"Few-shot:   {args.fewshot}")
    print(f"Rate limit: {args.rate_limit} req/s")
    print()

    # Load data
    records = load_records(args.input, max_n=args.n)
    print(f"Loaded {len(records)} records from {args.input}")

    # Estimate cost (rough: ~100 tokens per request at $0.0014/1K tokens for DeepSeek V3)
    est_cost_usd = len(records) * 100 / 1000 * 0.0014
    est_cost_cny = est_cost_usd * 7.2
    print(f"Estimated cost: ~¥{est_cost_cny:.2f} CNY (${est_cost_usd:.3f} USD)")
    print()

    # Get client and run
    client = get_client()
    results = predict_batch(
        client, records, model_id,
        use_fewshot=args.fewshot,
        rate_limit=args.rate_limit,
    )

    # Save and summarize
    save_results(results, args.output)
    print_summary(results, args.model)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

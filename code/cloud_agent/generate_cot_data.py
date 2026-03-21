#!/usr/bin/env python3
"""
CoT 数据生成脚本 - 使用硅基流动 DeepSeek R1（异步并行版）

生成思维链(CoT)数据用于知识蒸馏训练。

教师模型: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (硅基流动)
学生模型: Qwen3-4B

Usage:
    # 测试模式 (5条样本)
    python generate_cot_data.py --test

    # 从 HuggingFace 生成英文数据 (2000条, 10并发)
    python generate_cot_data.py --from-hf --count 2000 --concurrency 10

    # 从本地文件生成
    python generate_cot_data.py --input data/raw/amazon_train.jsonl --count 2000

    # 从断点恢复
    python generate_cot_data.py --from-hf --count 2000 --resume
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter

# ============== 硅基流动 API 配置 ==============

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

DEFAULT_OUTPUT = "data/processed/en_cot_2000.jsonl"


def get_api_key(api_key_override: str = None) -> str:
    """获取 API Key（优先级：参数 > 环境变量 > 配置文件）"""
    if api_key_override:
        return api_key_override

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if key:
        return key

    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "config" / "api_keys.json",
        Path(__file__).parent.parent.parent / "config" / "api_keys.json",
        Path("e:/20260122/config/api_keys.json"),
    ]
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                key = config.get("siliconflow", {}).get("api_key", "")
                if key:
                    return key

    print("错误: 未找到 SILICONFLOW_API_KEY")
    print("请设置环境变量: set SILICONFLOW_API_KEY=your-key")
    sys.exit(1)


# ============== CoT 提示模板 ==============

COT_PROMPT = """Analyze the sentiment of the following e-commerce review.

Review: {text}

Steps:
1. Extract sentiment-bearing keywords
2. Analyze overall tone
3. Consider context and nuance
4. Give final verdict

Output format:
<thinking>
[Your reasoning process]
</thinking>

<result>
label: [0=negative / 1=positive]
confidence: [0.0-1.0]
rationale: [one sentence explanation in English]
</result>
"""


# ============== 异步 API 调用 ==============

async def call_deepseek_r1_async(
    session,
    text: str,
    api_key: str,
    sem: asyncio.Semaphore,
    max_retries: int = 3,
) -> Tuple[str, str, Optional[Dict]]:
    """
    异步调用 DeepSeek R1（非流式，更适合并发）

    Returns:
        (cot_content, final_answer, parsed_result)
    """
    import aiohttp

    prompt = COT_PROMPT.format(text=text[:800])
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with sem:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{SILICONFLOW_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                choice = data["choices"][0]["message"]
                cot_content = choice.get("reasoning_content", "") or ""
                final_answer = choice.get("content", "") or ""
                parsed = _parse_result(final_answer)
                return cot_content, final_answer, parsed

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 2
                    await asyncio.sleep(wait)
                else:
                    raise e

    return "", "", None


def _parse_result(answer: str) -> Optional[Dict]:
    """解析模型输出"""
    import re

    result = {}

    label_match = re.search(r"label:\s*([01])", answer, re.IGNORECASE)
    if label_match:
        result["label"] = int(label_match.group(1))
    else:
        lower = answer.lower()
        if "negative" in lower:
            result["label"] = 0
        elif "positive" in lower:
            result["label"] = 1

    conf_match = re.search(r"confidence:\s*([0-9.]+)", answer, re.IGNORECASE)
    if conf_match:
        result["confidence"] = float(conf_match.group(1))

    rationale_match = re.search(r"rationale:\s*(.+)", answer, re.IGNORECASE)
    if rationale_match:
        result["rationale"] = rationale_match.group(1).strip()

    return result if "label" in result else None


# ============== 数据源 ==============

def load_from_file(data_path: str) -> List[Dict]:
    """从 JSONL 文件加载"""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


MS_BASE_URL = "https://www.modelscope.cn/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/master/raw/review_categories"

# ModelScope 文件名映射（空格用下划线，与 HF 配置名对应）
MS_CATEGORY_FILES = {
    "All_Beauty":    "All_Beauty.jsonl",
    "Electronics":   "Electronics.jsonl",
    "Pet_Supplies":  "Pet_Supplies.jsonl",
}


def load_from_hf(count: int = 3333, category: str = "All_Beauty") -> List[Dict]:
    """
    从 ModelScope 流式下载 Amazon 英文评论，按标签均衡采样（二分类）。
    自动回退：先尝试 HuggingFace 本地缓存，失败则用 ModelScope 在线流式下载。

    每类约 count//2 条（负面/正面）
    """
    per_class = count // 2  # 二分类每类各取一半
    label_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}  # 1-2星→负面, 3-5星→正面

    # 先尝试 HuggingFace 本地缓存（已缓存的 Beauty 优先用缓存）
    try:
        from datasets import load_dataset
        print(f"尝试 HuggingFace 缓存加载 [{category}]...")
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            streaming=True,
        )
        return _sample_from_iterable(dataset, per_class, label_map, category,
                                     key_text="text", key_rating="rating",
                                     key_id=lambda x: x.get("asin","") + "_" + str(x.get("unixReviewTime","")))
    except Exception as e:
        print(f"HuggingFace 缓存不可用 ({type(e).__name__})，切换到 ModelScope 流式下载...")

    # 回退：ModelScope 流式下载
    filename = MS_CATEGORY_FILES.get(category)
    if not filename:
        print(f"错误: 不支持的品类 {category}")
        sys.exit(1)

    url = f"{MS_BASE_URL}/{filename}"
    print(f"从 ModelScope 流式下载: {filename}")
    import requests as req

    with req.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()

        def ms_iter():
            for raw_line in resp.iter_lines():
                if raw_line:
                    yield json.loads(raw_line)

        return _sample_from_iterable(ms_iter(), per_class, label_map, category,
                                     key_text="text", key_rating="rating",
                                     key_id=lambda x: x.get("asin","") + "_" + str(x.get("timestamp","")))


def _sample_from_iterable(iterable, per_class: int, label_map: dict,
                          category: str, key_text: str, key_rating: str,
                          key_id) -> List[Dict]:
    """从可迭代数据源中均衡采样"""
    import random

    buckets: Dict[int, List[Dict]] = {0: [], 1: []}

    for item in iterable:
        try:
            rating = int(float(item.get(key_rating, 0)))
        except (ValueError, TypeError):
            continue

        label = label_map.get(rating, -1)
        if label == -1 or len(buckets[label]) >= per_class:
            continue

        text = (item.get(key_text) or "").strip()
        if not text or len(text) < 10:
            continue

        try:
            item_id = key_id(item) if callable(key_id) else item.get(key_id, "")
        except Exception:
            item_id = str(len(buckets[0]) + len(buckets[1]))

        buckets[label].append({
            "id": item_id,
            "text": text[:800],
            "sentiment_label": label,
            "rating": rating,
            "category": category,
            "language": "en",
        })

        if all(len(v) >= per_class for v in buckets.values()):
            break

    records = buckets[0] + buckets[1]
    random.seed(42)
    random.shuffle(records)
    print(f"加载完成: 负面={len(buckets[0])}, 正面={len(buckets[1])}")
    return records


# ============== 进度管理 ==============

def load_progress(progress_path: str) -> set:
    """返回已完成的 ID 集合"""
    if Path(progress_path).exists():
        with open(progress_path, "r") as f:
            data = json.load(f)
            return set(data.get("completed_ids", []))
    return set()


def save_progress(progress_path: str, completed_ids: set):
    with open(progress_path, "w") as f:
        json.dump(
            {"completed_ids": list(completed_ids), "count": len(completed_ids),
             "updated_at": datetime.now().isoformat()},
            f,
        )


def append_result(output_path: str, record: Dict):
    """追加写入（线程安全的文件锁在 asyncio 中不需要，gather 后统一写）"""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============== 软标签 ==============

def generate_soft_label(label: int, confidence: float) -> List[float]:
    probs = [0.05, 0.05, 0.05]
    probs[label] = max(confidence, 0.7)
    total = sum(probs)
    return [round(p / total, 4) for p in probs]


# ============== 主异步逻辑 ==============

async def process_batch(
    records: List[Dict],
    api_key: str,
    output_path: str,
    progress_path: str,
    completed_ids: set,
    concurrency: int,
) -> Tuple[int, int]:
    """并行处理所有记录，返回 (成功数, 失败数)"""
    try:
        import aiohttp
    except ImportError:
        print("请安装 aiohttp: pip install aiohttp")
        sys.exit(1)

    sem = asyncio.Semaphore(concurrency)
    success = 0
    failure = 0
    lock = asyncio.Lock()

    async def process_one(item: Dict):
        nonlocal success, failure
        item_id = item.get("id", "")
        if item_id in completed_ids:
            return

        text = item.get("text", item.get("original_text", ""))
        ground_truth = item.get("sentiment_label", item.get("sentiment", -1))

        try:
            async with session:
                cot, answer, parsed = await call_deepseek_r1_async(
                    session, text, api_key, sem
                )

            if parsed and "label" in parsed:
                label = parsed["label"]
                confidence = parsed.get("confidence", 0.8)
                rationale = parsed.get("rationale", "")
            else:
                label = ground_truth
                confidence = 0.7
                rationale = ""

            record = {
                "id": item_id,
                "text": text,
                "ground_truth_label": ground_truth,
                "predicted_label": label,
                "confidence": confidence,
                "rationale": rationale,
                "cot": cot,
                "soft_label": generate_soft_label(label, confidence),
                "model": MODEL_NAME,
                "created_at": datetime.now().isoformat(),
            }

            async with lock:
                append_result(output_path, record)
                completed_ids.add(item_id)
                success += 1
                done = success + failure
                if done % 100 == 0:
                    save_progress(progress_path, completed_ids)
                    print(f"  进度: {done}/{len(records)} | 成功:{success} 失败:{failure}")

        except Exception as e:
            async with lock:
                failure += 1
            print(f"  错误 [{item_id[:20]}]: {e}")

    # 为每个任务创建独立 session（避免并发冲突）
    async with aiohttp.ClientSession() as session:
        sem2 = asyncio.Semaphore(concurrency)

        async def process_one_v2(item: Dict):
            nonlocal success, failure
            item_id = item.get("id", "")
            if item_id in completed_ids:
                return

            text = item.get("text", item.get("original_text", ""))
            ground_truth = item.get("sentiment_label", item.get("sentiment", -1))

            try:
                cot, answer, parsed = await call_deepseek_r1_async(
                    session, text, api_key, sem2
                )

                if parsed and "label" in parsed:
                    label = parsed["label"]
                    confidence = parsed.get("confidence", 0.8)
                    rationale = parsed.get("rationale", "")
                else:
                    label = ground_truth
                    confidence = 0.7
                    rationale = ""

                record = {
                    "id": item_id,
                    "text": text,
                    "ground_truth_label": ground_truth,
                    "predicted_label": label,
                    "confidence": confidence,
                    "rationale": rationale,
                    "cot": cot,
                    "soft_label": generate_soft_label(label, confidence),
                    "model": MODEL_NAME,
                    "created_at": datetime.now().isoformat(),
                }

                async with lock:
                    append_result(output_path, record)
                    completed_ids.add(item_id)
                    success += 1
                    done = success + failure
                    if done % 100 == 0:
                        save_progress(progress_path, completed_ids)
                        print(f"  进度: {done}/{len(records)} | 成功:{success} 失败:{failure}")

            except Exception as e:
                async with lock:
                    failure += 1
                print(f"  错误 [{item_id[:20]}]: {e}")

        tasks = [process_one_v2(item) for item in records]
        await asyncio.gather(*tasks)

    save_progress(progress_path, completed_ids)
    return success, failure


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description="生成 CoT 蒸馏数据（异步并行版）")
    parser.add_argument("--test", action="store_true", help="测试模式 (4条样本)")
    parser.add_argument("--from-hf", action="store_true", help="从 HuggingFace 加载英文数据")
    parser.add_argument("--category", type=str, default="All_Beauty",
                        choices=["All_Beauty", "Electronics", "Pet_Supplies"],
                        help="Amazon 品类 (default: All_Beauty)")
    parser.add_argument("--input", type=str, help="输入 JSONL 文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 JSONL 文件路径（默认按品类自动命名）")
    parser.add_argument("--count", type=int, default=3333, help="生成条数 (default: 3333)")
    parser.add_argument("--concurrency", type=int, default=7, help="并发请求数 (default: 7)")
    parser.add_argument("--resume", action="store_true", help="从断点恢复")
    parser.add_argument("--api-key", type=str, help="硅基流动 API Key")
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)

    # 自动输出文件名
    if args.output is None:
        cat_lower = args.category.lower().replace("_", "")
        args.output = f"data/processed/cot_{cat_lower}_{args.count}.jsonl"

    # 加载数据
    if args.from_hf:
        count = 4 if args.test else args.count
        records = load_from_hf(count, args.category)
    elif args.input:
        records = load_from_file(args.input)
        if args.test:
            records = records[:5]
        elif args.count:
            records = records[: args.count]
    else:
        parser.error("请指定 --from-hf 或 --input")

    print(f"共 {len(records)} 条待处理")

    # 准备输出
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.replace(".jsonl", "_progress.json")

    completed_ids = load_progress(progress_path) if args.resume else set()
    if completed_ids:
        print(f"断点恢复: 已完成 {len(completed_ids)} 条，跳过已处理记录")

    # 若非 resume 且输出文件已存在，清空它
    if not args.resume and Path(output_path).exists():
        Path(output_path).unlink()

    print(f"模型: {MODEL_NAME}")
    print(f"并发数: {args.concurrency}")
    print(f"输出: {output_path}")
    print("=" * 60)

    # 运行异步生成
    success, failure = asyncio.run(
        process_batch(records, api_key, output_path, progress_path, completed_ids, args.concurrency)
    )

    # 最终统计
    print("\n" + "=" * 60)
    print("生成完成!")
    print(f"成功: {success}  失败: {failure}")

    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        dist = Counter(r["predicted_label"] for r in results)
        print(f"\n标签分布 (二分类):")
        print(f"  Negative (0): {dist.get(0, 0)}")
        print(f"  Positive (1): {dist.get(1, 0)}")
        if results:
            avg_cot = sum(len(r.get("cot", "")) for r in results) / len(results)
            print(f"\n平均 CoT 长度: {avg_cot:.0f} 字符")


if __name__ == "__main__":
    main()

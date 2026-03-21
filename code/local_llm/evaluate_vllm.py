"""
使用 vLLM 批量评估模型准确率（高速推理）
"""

import json
import argparse
import re
import time
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/qwen3-4b-sentiment-lora_merged_16bit")
    parser.add_argument("--data", type=str, default="data/processed/test.json")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32, help="vLLM batch size")
    parser.add_argument("--max-tokens", type=int, default=100, help="生成最大token数")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def extract_label_from_cot(text: str) -> tuple[int, str]:
    """从CoT格式数据中提取标签和review文本"""
    # 提取sentiment标签
    s = re.search(r'"sentiment":\s*([0-2])', text)
    true_label = int(s.group(1)) if s else -1

    # 提取review文本
    review_match = re.search(r'Review:\s*([^<\n]+)', text)
    if review_match:
        review_text = review_match.group(1).strip()[:500]
    else:
        review_text = text[:300]

    return true_label, review_text


def create_prompt(review_text: str) -> str:
    """构建推理prompt"""
    return f"""<|im_start|>system
You are a sentiment analysis expert for e-commerce reviews.
Analyze the review and output ONLY a JSON object with these exact fields:
- sentiment: 0 (negative) or 1 (positive)
- confidence: float between 0 and 1
- rationale: brief explanation in English

Output format: {{"sentiment": X, "confidence": Y.YZ, "rationale": "explanation"}}<|im_end|>
<|im_start|>user
Review: {review_text}<|im_end|>
<|im_start|>assistant
<thinking>
"""


def extract_sentiment_from_output(text: str) -> int:
    """从模型输出中提取情感标签"""
    patterns = [
        r'"sentiment"\s*[:=]\s*([0-2])',
        r'sentiment["\']?\s*[:=]\s*["\']?([0-2])',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))

    # 关键词匹配
    text_lower = text.lower()
    if any(w in text_lower for w in ['negative', '消极', '负面']):
        return 0
    if any(w in text_lower for w in ['positive', '积极', '正面']):
        return 2
    if any(w in text_lower for w in ['neutral', '中性']):
        return 1
    return -1


def main():
    args = parse_args()

    # 延迟导入vLLM（加快启动）
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("错误: 未安装vLLM")
        print("安装: pip install vllm")
        return

    # 加载测试数据
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 测试数据不存在: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    test_samples = raw_data[:args.max_samples]
    print(f"加载测试数据: {len(test_samples)} 条")

    # 准备prompts
    prompts = []
    true_labels = []
    skipped = 0

    for item in test_samples:
        text = item["text"]
        label, review = extract_label_from_cot(text)
        if label == -1:
            skipped += 1
            continue
        prompts.append(create_prompt(review))
        true_labels.append(label)

    print(f"有效样本: {len(prompts)}, 跳过: {skipped}")

    # 加载vLLM模型
    print(f"\n加载vLLM模型: {args.model}")
    print(f"GPU内存利用率: {args.gpu_memory_utilization}")
    start_time = time.time()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=512,
        trust_remote_code=True,
    )

    load_time = time.time() - start_time
    print(f"模型加载完成: {load_time:.1f}s")

    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "</thinking>"],
    )

    # 批量推理
    print(f"\n开始批量推理 (batch_size={args.batch_size})...")
    start_time = time.time()

    all_outputs = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Batches"):
        batch_prompts = prompts[i:i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(outputs)

    infer_time = time.time() - start_time
    total_samples = len(all_outputs)
    speed = total_samples / infer_time * 60  # 条/分钟

    print(f"\n推理完成: {infer_time:.1f}s ({speed:.0f} 条/分钟)")

    # 解析结果
    correct = 0
    predictions = []
    failed_parse = 0

    for i, output in enumerate(all_outputs):
        pred_text = output.outputs[0].text
        pred_label = extract_sentiment_from_output(pred_text)
        true_label = true_labels[i]

        if pred_label == -1:
            failed_parse += 1
            # 尝试用关键词再匹配一次
            if re.search(r'"sentiment":\s*0', pred_text):
                pred_label = 0
            elif re.search(r'"sentiment":\s*2', pred_text):
                pred_label = 2
            elif re.search(r'"sentiment":\s*1', pred_text):
                pred_label = 1

        is_correct = (pred_label == true_label) if pred_label != -1 else False
        if is_correct:
            correct += 1

        predictions.append({
            "true": true_label,
            "pred": pred_label,
            "correct": is_correct,
            "raw": pred_text[:150],
        })

    # 统计结果
    total_valid = len([p for p in predictions if p["pred"] != -1])
    accuracy = correct / total_valid * 100 if total_valid > 0 else 0

    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"总样本: {total_samples}")
    print(f"解析成功: {total_valid} (失败: {failed_parse})")
    print(f"正确: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"推理速度: {total_samples/infer_time:.2f} 条/秒 ({speed:.0f} 条/分钟)")

    # 按类别统计
    labels = {0: "负面", 1: "中性", 2: "正面"}
    for label, name in labels.items():
        label_preds = [p for p in predictions if p["true"] == label and p["pred"] != -1]
        if label_preds:
            label_correct = sum(p["correct"] for p in label_preds)
            label_acc = label_correct / len(label_preds) * 100
            print(f"\n{name} 类: {label_correct}/{len(label_preds)} = {label_acc:.1f}%")

    # 显示错误示例
    errors = [p for p in predictions if not p["correct"] and p["pred"] != -1][:3]
    if errors:
        print(f"\n错误示例:")
        for i, p in enumerate(errors):
            print(f"\n{i+1}. 真实={p['true']}, 预测={p['pred']}")
            print(f"   输出: {p['raw'][:100]}...")

    # 保存详细结果
    result_path = Path("evaluation_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "total": total_samples,
            "correct": correct,
            "speed_samples_per_sec": total_samples / infer_time,
            "predictions": predictions,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {result_path}")


if __name__ == "__main__":
    main()

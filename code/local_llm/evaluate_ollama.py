"""
Ollama 批量评估 - 使用本地 GGUF 模型
速度: ~20-30条/秒 (比 Unsloth 单条快 200x)
"""

import json
import argparse
import re
import time
from pathlib import Path
from typing import List
import concurrent.futures
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Ollama Batch Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Ollama model name (e.g., qwen3-s5)")
    parser.add_argument("--data", type=str,
                        default="data/curriculum/val_fixed.json",
                        help="Validation data path")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers (Ollama handles batching internally)")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def create_prompt(review_text: str) -> str:
    """Build prompt"""
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


def load_validation_data(data_path: str, max_samples: int) -> List[dict]:
    """Load validation data (Alpaca format)"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data[:max_samples]:
        if "input" in item and "output" in item:
            review = item.get("input", "").strip()
            try:
                label = int(item.get("output", "-1").strip())
            except ValueError:
                continue
            if review and label in [0, 1, 2]:
                samples.append({
                    "review": review[:500],
                    "label": label
                })

    return samples


def extract_sentiment(text: str) -> int:
    """Extract sentiment from output"""
    text_lower = text.lower()

    match = re.search(r'"sentiment"\s*[:=]\s*([0-2])', text_lower)
    if match:
        return int(match.group(1))

    negative_words = ['negative', 'disappointed', 'terrible', 'bad', 'worst']
    positive_words = ['positive', 'excellent', 'great', 'good', 'amazing']

    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)

    if neg_count > pos_count:
        return 0
    elif pos_count > neg_count:
        return 2
    else:
        return 1


def query_ollama(prompt: str, model: str) -> str:
    """Query Ollama API"""
    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 128,
            }
        }
    )
    response.raise_for_status()
    return response.json()["response"]


def evaluate_with_ollama(model: str, samples: List[dict], workers: int):
    """Evaluate using Ollama"""
    print(f"🚀 Evaluating with Ollama model: {model}")
    print(f"   Samples: {len(samples)}, Workers: {workers}")

    prompts = [create_prompt(s["review"]) for s in samples]
    true_labels = [s["label"] for s in samples]

    # Check Ollama is running
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags")
        if r.status_code != 200:
            raise ConnectionError()
        models = [m["name"] for m in r.json()["models"]]
        if model not in models:
            print(f"❌ Model '{model}' not found in Ollama")
            print(f"   Available models: {models}")
            print(f"   Please run: ollama create {model} -f <modelfile>")
            return None
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Please ensure Ollama is running:")
        print("   - Windows: Start Ollama app")
        print("   - Linux/Mac: ollama serve")
        return None

    # Parallel inference
    print(f"\n⏱️  Starting inference...")
    start_time = time.time()

    all_outputs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(query_ollama, p, model): i for i, p in enumerate(prompts)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="Inferencing"):
            i = futures[future]
            try:
                response = future.result()
                all_outputs.append((i, response))
            except Exception as e:
                print(f"\n❌ Error on sample {i}: {e}")
                all_outputs.append((i, ""))

    # Sort by original order
    all_outputs.sort(key=lambda x: x[0])
    all_outputs = [r for _, r in all_outputs]

    infer_time = time.time() - start_time
    speed = len(all_outputs) / infer_time

    print(f"\n✅ Inference complete: {infer_time:.1f}s")
    print(f"   Speed: {speed:.1f} samples/sec ({speed*60:.0f} samples/min)")

    # Parse results
    results = []
    correct = 0
    class_stats = {0: {"total": 0, "correct": 0},
                   1: {"total": 0, "correct": 0},
                   2: {"total": 0, "correct": 0}}

    for i, pred_text in enumerate(all_outputs):
        pred_label = extract_sentiment(pred_text)
        true_label = true_labels[i]

        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1

        class_stats[true_label]["total"] += 1
        if is_correct:
            class_stats[true_label]["correct"] += 1

        results.append({
            "true": true_label,
            "pred": pred_label,
            "correct": is_correct,
            "review": samples[i]["review"][:100],
            "raw": pred_text[:150],
        })

    accuracy = correct / len(results) * 100

    return {
        "accuracy": accuracy,
        "total": len(results),
        "correct": correct,
        "time": infer_time,
        "speed": speed,
        "class_stats": class_stats,
        "results": results,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("🎓 Ollama Batch Evaluation")
    print("=" * 60)

    # Check data
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        return

    # Load data
    print(f"\n📂 Loading validation data: {args.data}")
    samples = load_validation_data(args.data, args.samples)
    print(f"   Valid samples: {len(samples)}")

    if len(samples) == 0:
        print("❌ No valid samples")
        return

    # Evaluate
    eval_result = evaluate_with_ollama(args.model, samples, args.workers)

    if eval_result is None:
        return

    # Print results
    print(f"\n{'='*60}")
    print("📊 Evaluation Results")
    print(f"{'='*60}")
    print(f"Total: {eval_result['total']}")
    print(f"Correct: {eval_result['correct']}")
    print(f"Accuracy: {eval_result['accuracy']:.2f}%")
    print(f"Speed: {eval_result['speed']:.1f} samples/sec")

    # Class distribution
    print(f"\nClass Accuracy:")
    label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for label, stats in eval_result["class_stats"].items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {label_names[label]}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

    # Save results
    output_file = args.output or f"results/curriculum/eval_{args.model}_ollama.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved: {output_file}")

    # Error examples
    errors = [r for r in eval_result["results"] if not r["correct"]][:3]
    if errors:
        print(f"\n❌ Error examples:")
        for i, e in enumerate(errors):
            print(f"\n{i+1}. True={e['true']}, Pred={e['pred']}")
            print(f"   Review: {e['review'][:80]}...")


if __name__ == "__main__":
    main()

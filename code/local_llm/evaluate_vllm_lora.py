"""
本地 vLLM + LoRA 批量评估 (RTX 3070Ti 8GB)
速度: ~50-100条/秒 (比 Unsloth 单条快 500-1000x)
"""

import json
import argparse
import re
import time
from pathlib import Path
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM + LoRA Batch Evaluation (Local)")
    parser.add_argument("--stage", type=str, default="s5",
                        choices=["s1", "s2", "s3", "s4", "s5"],
                        help="Stage to evaluate")
    parser.add_argument("--merged-model", type=str, default=None,
                        help="Path to merged model (overrides --stage)")
    parser.add_argument("--base-model", type=str,
                        default="unsloth/Qwen3-4B-unsloth-bnb-4bit",
                        help="Base model name (for LoRA mode)")
    parser.add_argument("--model-dir", type=str, default="./models/curriculum",
                        help="Directory containing LoRA adapters")
    parser.add_argument("--data", type=str,
                        default="./data/curriculum/val_fixed.json",
                        help="Validation data path")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    return parser.parse_args()


def get_model_path(stage: str, model_dir: str) -> str:
    """Get LoRA adapter path for stage"""
    stage_map = {
        "s1": "lora_s1_600",
        "s2": "lora_s2_1200",
        "s3": "lora_s3_2400",
        "s4": "lora_s4_4800",
        "s5": "lora_s5_full",
    }
    return f"{model_dir}/{stage_map[stage]}"


def create_prompt(review_text: str) -> str:
    """Build prompt matching training format"""
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
<think>
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
                samples.append({"review": review[:500], "label": label})

    return samples


def extract_sentiment(text: str) -> int:
    """Extract sentiment from output"""
    text_lower = text.lower()

    match = re.search(r'"sentiment"\s*[:=]\s*([0-2])', text_lower)
    if match:
        return int(match.group(1))

    # Keyword fallback
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


def evaluate_with_vllm_merged(
    model_path: str,
    samples: list,
    batch_size: int,
    max_tokens: int
):
    """Evaluate using vLLM with merged model (stable)"""
    from vllm import LLM, SamplingParams

    print(f"🚀 Loading vLLM model: {model_path}")
    print(f"   Samples: {len(samples)}, Batch: {batch_size}")

    # Load merged model (more stable than LoRA)
    llm = LLM(
        model=model_path,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=512,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "</think>"],
    )

    # Prepare prompts
    prompts = [create_prompt(s["review"]) for s in samples]
    true_labels = [s["label"] for s in samples]

    # Batch inference
    print(f"\n⏱️  Starting batch inference...")
    start_time = time.time()

    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        all_outputs.extend(outputs)

        if (i // batch_size + 1) % 5 == 0 or i == 0:
            progress = min((i + batch_size), len(prompts))
            print(f"   Progress: {progress}/{len(prompts)}")

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

    for i, output in enumerate(all_outputs):
        pred_text = output.outputs[0].text
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


def evaluate_with_vllm_lora(
    base_model: str,
    lora_path: str,
    samples: List[dict],
    batch_size: int,
    max_tokens: int
):
    """Evaluate using vLLM with LoRA adapters"""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"🚀 Loading vLLM with LoRA")
    print(f"   Base: {base_model}")
    print(f"   LoRA: {lora_path}")
    print(f"   Samples: {len(samples)}, Batch: {batch_size}")

    # Load model with LoRA support
    llm = LLM(
        model=base_model,
        dtype="float16",  # RTX 3070Ti supports float16
        gpu_memory_utilization=0.75,  # Lower for 8GB VRAM
        max_model_len=512,
        trust_remote_code=True,
        enable_lora=True,  # Enable LoRA support
        max_lora_rank=16,
    )

    # Create LoRA request
    lora_request = LoRARequest(
        lora_name="sentiment_adapter",
        lora_int_id=1,
        lora_local_path=lora_path,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "</think>"],
    )

    # Prepare prompts
    prompts = [create_prompt(s["review"]) for s in samples]
    true_labels = [s["label"] for s in samples]

    # Batch inference
    print(f"\n⏱️  Starting batch inference...")
    start_time = time.time()

    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(
            batch,
            sampling_params,
            lora_request=lora_request,  # Apply LoRA
        )
        all_outputs.extend(outputs)

        if (i // batch_size + 1) % 5 == 0 or i == 0:
            progress = min((i + batch_size), len(prompts))
            print(f"   Progress: {progress}/{len(prompts)}")

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

    for i, output in enumerate(all_outputs):
        pred_text = output.outputs[0].text
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

    # Determine model path
    if args.merged_model:
        model_path = args.merged_model
        mode = "merged"
    else:
        model_path = get_model_path(args.stage, args.model_dir) + "_merged_16bit"
        mode = "merged"

    print("=" * 60)
    print(f"🎓 vLLM Evaluation - {args.stage.upper() if not args.merged_model else 'Custom'}")
    print("=" * 60)

    # Check files
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Please run: python code/local_llm/export_merged_models.py --stages {args.stage}")
        return

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

    # Evaluate with merged model (more stable)
    eval_result = evaluate_with_vllm_merged(
        model_path=model_path,
        samples=samples,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

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
    output_file = args.output or f"results/curriculum/eval_{args.stage}_vllm.json"
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

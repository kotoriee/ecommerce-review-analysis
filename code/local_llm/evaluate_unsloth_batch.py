"""
Unsloth Batch Evaluation - 批量推理 (比单条快 5-10x)
RTX 3070Ti 8GB: ~1-2条/秒，500条约4-8分钟
"""

import json
import argparse
import re
import time
from pathlib import Path
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth Batch Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to LoRA adapters (e.g., models/curriculum/lora_s5_full)")
    parser.add_argument("--base-model", type=str,
                        default="unsloth/Qwen3-4B-unsloth-bnb-4bit",
                        help="Base model name")
    parser.add_argument("--data", type=str,
                        default="data/curriculum/val_fixed.json",
                        help="Validation data path")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference (16 for 8GB VRAM)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    return parser.parse_args()


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


def load_validation_data(data_path: str, max_samples: int):
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


def batch_infer(model, tokenizer, samples, batch_size, max_tokens):
    """Batch inference with Unsloth"""
    prompts = [create_prompt(s["review"]) for s in samples]
    true_labels = [s["label"] for s in samples]

    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    print(f"\n⏱️  Starting batch inference...")
    print(f"   Total samples: {len(prompts)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total batches: {total_batches}")

    start_time = time.time()

    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True  # Key for batching
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode outputs (skip input tokens)
        input_lengths = inputs['input_ids'].shape[1]
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[input_lengths:], skip_special_tokens=True)
            all_outputs.append(response)

    infer_time = time.time() - start_time
    speed = len(all_outputs) / infer_time

    print(f"\n✅ Inference complete: {infer_time:.1f}s")
    print(f"   Speed: {speed:.2f} samples/sec ({speed*60:.1f} samples/min)")

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
    from unsloth import FastLanguageModel

    args = parse_args()

    print("=" * 60)
    print("🎓 Unsloth Batch Evaluation")
    print("=" * 60)

    # Check files
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
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

    # Load model
    print(f"\n🚀 Loading model...")
    print(f"   Base: {args.base_model}")
    print(f"   LoRA: {args.model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=512,
        load_in_4bit=True,
    )

    # Load LoRA
    model = FastLanguageModel.get_peft_model(model)
    model.load_adapter(args.model, adapter_name="default")
    model.set_adapter("default")
    FastLanguageModel.for_inference(model)

    print(f"   Model loaded on: {next(model.parameters()).device}")

    # Batch inference
    eval_result = batch_infer(
        model=model,
        tokenizer=tokenizer,
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
    print(f"Speed: {eval_result['speed']:.2f} samples/sec")

    # Class distribution
    print(f"\nClass Accuracy:")
    label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for label, stats in eval_result["class_stats"].items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {label_names[label]}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

    # Save results
    output_file = args.output or f"results/curriculum/eval_{Path(args.model).name}_batch.json"
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

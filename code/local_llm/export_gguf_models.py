"""
批量导出所有 curriculum 阶段的 GGUF 4-bit 模型
用于 Ollama / llama.cpp 高速推理
"""

import argparse
from pathlib import Path
from unsloth import FastLanguageModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=str, default="s1,s2,s3,s4,s5",
                        help="Stages to export, comma-separated")
    parser.add_argument("--model-dir", type=str, default="./models/curriculum")
    parser.add_argument("--base-model", type=str,
                        default="unsloth/Qwen3-4B-unsloth-bnb-4bit")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="GGUF quantization method")
    return parser.parse_args()


def export_gguf_model(stage: str, model_dir: str, base_model: str, quantization: str):
    """Export LoRA to GGUF 4-bit model"""
    stage_map = {
        "s1": "lora_s1_600",
        "s2": "lora_s2_1200",
        "s3": "lora_s3_2400",
        "s4": "lora_s4_4800",
        "s5": "lora_s5_full",
    }

    lora_path = f"{model_dir}/{stage_map[stage]}"
    output_path = f"{model_dir}/{stage_map[stage]}_{quantization}.gguf"

    if Path(output_path).exists():
        print(f"⚠️  {stage} GGUF already exists, skipping")
        return output_path

    print(f"\n{'='*60}")
    print(f"Exporting {stage.upper()} to GGUF ({quantization})")
    print(f"{'='*60}")
    print(f"LoRA: {lora_path}")
    print(f"Output: {output_path}")

    # Load base model (4-bit, no need to load 16-bit)
    print("\nLoading base model (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=512,
        load_in_4bit=True,
    )

    # Load LoRA
    print("Loading LoRA adapters...")
    model = FastLanguageModel.get_peft_model(model)
    model.load_adapter(lora_path, adapter_name="default")
    model.set_adapter("default")

    # Export to GGUF
    print(f"Exporting to GGUF ({quantization})...")
    print("This may take 2-3 minutes...")

    model.save_pretrained_gguf(
        output_path.replace(f".gguf", ""),  # remove extension, unsloth adds it
        tokenizer,
        quantization_method=quantization,
    )

    print(f"✅ Exported: {output_path}")
    return output_path


def create_ollama_modelfile(model_path: str, stage: str):
    """Create Ollama Modelfile"""
    modelfile_content = f'''FROM {model_path}

SYSTEM """You are a sentiment analysis expert for e-commerce reviews.
Analyze the review and output ONLY a JSON object with sentiment (0=negative, 1=neutral, 2=positive),
confidence (0-1), and rationale."""

PARAMETER temperature 0.1
PARAMETER num_predict 128
PARAMETER stop "<|im_end|>"
PARAMETER stop "</think>"
'''

    modelfile_path = model_path.replace(".gguf", ".modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"📝 Created Modelfile: {modelfile_path}")
    print(f"\nTo use with Ollama:")
    print(f"  ollama create qwen3-{stage} -f {modelfile_path}")
    print(f"  ollama run qwen3-{stage}")


def main():
    args = parse_args()
    stages = [s.strip() for s in args.stages.split(",")]

    print("=" * 60)
    print("Batch Export Curriculum Models to GGUF")
    print("=" * 60)
    print(f"Quantization: {args.quantization}")

    exported = []
    for stage in stages:
        if stage not in ["s1", "s2", "s3", "s4", "s5"]:
            print(f"⚠️  Invalid stage: {stage}, skipping")
            continue

        try:
            output_path = export_gguf_model(
                stage, args.model_dir, args.base_model, args.quantization
            )
            exported.append((stage, output_path))

            # Create Ollama modelfile
            create_ollama_modelfile(output_path, stage)

        except Exception as e:
            print(f"❌ Failed to export {stage}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)

    if exported:
        print("\nExported models:")
        for stage, path in exported:
            print(f"  {stage}: {path}")

        print("\nTo evaluate with Ollama:")
        print("  1. ollama create qwen3-s5 -f models/curriculum/lora_s5_full_q4_k_m.modelfile")
        print("  2. python code/local_llm/evaluate_ollama.py --model qwen3-s5 --samples 500")


if __name__ == "__main__":
    main()

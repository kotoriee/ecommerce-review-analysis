"""
批量导出所有 curriculum 阶段的 merged 模型 (16-bit)
用于 vLLM 高速推理
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
    return parser.parse_args()


def export_merged_model(stage: str, model_dir: str, base_model: str):
    """Export LoRA to merged 16-bit model"""
    stage_map = {
        "s1": "lora_s1_600",
        "s2": "lora_s2_1200",
        "s3": "lora_s3_2400",
        "s4": "lora_s4_4800",
        "s5": "lora_s5_full",
    }

    lora_path = f"{model_dir}/{stage_map[stage]}"
    output_path = f"{model_dir}/{stage_map[stage]}_merged_16bit"

    if Path(output_path).exists():
        print(f"⚠️  {stage} already exported, skipping")
        return

    print(f"\n{'='*60}")
    print(f"Exporting {stage.upper()}")
    print(f"{'='*60}")
    print(f"LoRA: {lora_path}")
    print(f"Output: {output_path}")

    # Load base model
    print("\nLoading base model...")
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

    # Merge and export
    print("Merging and exporting (16-bit)...")
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")

    print(f"✅ Exported: {output_path}")


def main():
    args = parse_args()
    stages = args.stages.split(",")

    print("=" * 60)
    print("Batch Export Curriculum Models")
    print("=" * 60)

    for stage in stages:
        stage = stage.strip()
        if stage not in ["s1", "s2", "s3", "s4", "s5"]:
            print(f"⚠️  Invalid stage: {stage}, skipping")
            continue

        try:
            export_merged_model(stage, args.model_dir, args.base_model)
        except Exception as e:
            print(f"❌ Failed to export {stage}: {e}")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

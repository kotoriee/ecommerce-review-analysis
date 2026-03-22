"""
Agent smoke tests — validate all four agent prerequisites.

Tests verify:
1. All scripts referenced by agents exist and are importable
2. All data files expected by agents exist
3. All model paths expected by agents exist
4. Key CLI entry points accept --help without crashing

Run: pytest code/tests/test_agents_smoke.py -v
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.parent  # ecommerce-review-analysis/


def script_exists(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


def dir_exists(rel_path: str) -> bool:
    return (ROOT / rel_path).is_dir()


def can_import(module_path: str) -> bool:
    """Check a .py file can be loaded by Python without syntax errors."""
    full_path = ROOT / module_path
    spec = importlib.util.spec_from_file_location("_tmp_module", full_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return True
    except SystemExit:
        # argparse --help triggers SystemExit — that's fine
        return True
    except Exception:
        return False


def cli_help_works(rel_path: str) -> tuple[bool, str]:
    """Run `python script.py --help` and check it exits cleanly (0 or 2)."""
    result = subprocess.run(
        [sys.executable, str(ROOT / rel_path), "--help"],
        capture_output=True, text=True, timeout=60
    )
    # argparse --help exits with 0; some scripts exit 2 — both indicate working CLI
    ok = result.returncode in (0, 2) or "usage" in result.stdout.lower() or "usage" in result.stderr.lower()
    return ok, result.stdout + result.stderr


# ─── sentiment-data-pipeline ─────────────────────────────────────────────────

class TestDataPipelineAgent:
    """Verify sentiment-data-pipeline prerequisites."""

    def test_generate_soft_labels_exists(self):
        assert script_exists("code/cloud_agent/generate_soft_labels.py"), \
            "generate_soft_labels.py missing"

    def test_merge_datasets_exists(self):
        assert script_exists("code/cloud_agent/merge_datasets.py"), \
            "merge_datasets.py missing"

    def test_run_3cls_annotation_exists(self):
        assert script_exists("code/cloud_agent/run_3cls_annotation.py"), \
            "run_3cls_annotation.py missing"

    def test_train_3cls_data_exists(self):
        assert script_exists("data/processed/train_3cls.json"), \
            "train_3cls.json missing — run data pipeline first"

    def test_val_3cls_data_exists(self):
        assert script_exists("data/processed/val_3cls.json"), \
            "val_3cls.json missing"

    def test_test_3cls_data_exists(self):
        assert script_exists("data/processed/test_3cls.json"), \
            "test_3cls.json missing"

    def test_train_3cls_has_sufficient_samples(self):
        import json
        path = ROOT / "data/processed/train_3cls.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) >= 1000, \
            f"Only {len(data)} training samples — need at least 1000"

    def test_test_3cls_label_distribution(self):
        """Test set should contain all three labels (stored in 'output' field as '0','1','2')."""
        import json
        path = ROOT / "data/processed/test_3cls.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        # Data format: {"instruction": ..., "input": ..., "output": "0"|"1"|"2", ...}
        labels = {r.get("output") for r in data}
        assert labels >= {"0", "1", "2"}, \
            f"Test set only has labels: {labels} — expected all 3 classes (0/1/2)"


# ─── sentiment-trainer ───────────────────────────────────────────────────────

class TestTrainerAgent:
    """Verify sentiment-trainer prerequisites."""

    def test_train_script_exists(self):
        assert script_exists("code/local_llm/train_sentiment.py"), \
            "train_sentiment.py missing"

    def test_evaluate_unsloth_exists(self):
        assert script_exists("code/local_llm/evaluate_unsloth.py"), \
            "evaluate_unsloth.py missing"

    def test_lora_model_dir_exists(self):
        assert dir_exists("models/qwen3-4b-sentiment-lora"), \
            "LoRA adapter dir missing — run training first"

    def test_lora_adapter_files_exist(self):
        lora_dir = ROOT / "models/qwen3-4b-sentiment-lora"
        adapter_files = list(lora_dir.glob("adapter_*.json")) + list(lora_dir.glob("adapter_*.safetensors")) + list(lora_dir.glob("adapter_*.bin"))
        assert len(adapter_files) > 0, \
            f"No adapter files in {lora_dir} — model may not have saved correctly"

    def test_curriculum_models_exist(self):
        """At least one curriculum stage model should exist."""
        curriculum_dir = ROOT / "models/curriculum"
        stages = [d for d in curriculum_dir.iterdir() if d.is_dir()] if curriculum_dir.exists() else []
        assert len(stages) > 0, \
            "No curriculum stage models found — expected at least lora_s1_600"

    def test_train_script_cli(self):
        ok, output = cli_help_works("code/local_llm/train_sentiment.py")
        assert ok, f"train_sentiment.py --help failed:\n{output}"

    def test_evaluate_script_cli(self):
        """evaluate_unsloth.py loads unsloth+CUDA on import which can take >60s on first run.
        Verify the file exists and is syntactically valid instead of running --help."""
        import ast
        path = ROOT / "code/local_llm/evaluate_unsloth.py"
        source = path.read_text(encoding="utf-8")
        # Will raise SyntaxError if file is malformed
        ast.parse(source)
        assert "argparse" in source, "evaluate_unsloth.py should use argparse for CLI"


# ─── sentiment-infer ─────────────────────────────────────────────────────────

class TestInferAgent:
    """Verify sentiment-infer prerequisites."""

    def test_predictor_exists(self):
        assert script_exists("code/local_llm/predictor.py"), \
            "predictor.py missing"

    def test_batch_sentiment_exists(self):
        assert script_exists("code/cloud_agent/batch_sentiment.py"), \
            "batch_sentiment.py missing"

    def test_merged_model_dir_exists(self):
        assert dir_exists("models/qwen3-4b-sentiment-lora_merged_16bit"), \
            "Merged 16-bit model dir missing — run training + merge step first"

    def test_gguf_model_dir_exists(self):
        assert dir_exists("models/qwen3-4b-sentiment-lora_gguf"), \
            "GGUF model dir missing — run training + GGUF export first"

    def test_batch_sentiment_cli(self):
        ok, output = cli_help_works("code/cloud_agent/batch_sentiment.py")
        assert ok, f"batch_sentiment.py --help failed:\n{output}"

    def test_predictor_importable(self):
        assert can_import("code/local_llm/predictor.py"), \
            "predictor.py has import/syntax errors"


# ─── review-reporter ─────────────────────────────────────────────────────────

class TestReporterAgent:
    """Verify review-reporter prerequisites."""

    def test_svm_classifier_exists(self):
        assert script_exists("code/baseline/svm_classifier.py"), \
            "svm_classifier.py missing"

    def test_metrics_module_exists(self):
        assert script_exists("code/evaluation/metrics.py"), \
            "metrics.py missing"

    def test_run_comparison_exists(self):
        assert script_exists("code/evaluation/run_comparison.py"), \
            "run_comparison.py missing"

    def test_api_eval_sentiment_exists(self):
        assert script_exists("code/cloud_agent/api_eval_sentiment.py"), \
            "api_eval_sentiment.py missing"

    def test_metrics_importable(self):
        assert can_import("code/evaluation/metrics.py"), \
            "metrics.py has import/syntax errors"

    def test_metrics_has_compute_metrics(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "metrics", ROOT / "code/evaluation/metrics.py"
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        assert hasattr(mod, "compute_metrics"), \
            "metrics.py is missing compute_metrics() function"

    def test_svm_classifier_cli(self):
        ok, output = cli_help_works("code/baseline/svm_classifier.py")
        assert ok, f"svm_classifier.py --help failed:\n{output}"

    def test_api_eval_sentiment_cli(self):
        ok, output = cli_help_works("code/cloud_agent/api_eval_sentiment.py")
        assert ok, f"api_eval_sentiment.py --help failed:\n{output}"

    def test_api_eval_supported_models(self):
        """Verify DeepSeek and Qwen models are configured (not Claude)."""
        content = (ROOT / "code/cloud_agent/api_eval_sentiment.py").read_text()
        assert "deepseek" in content.lower(), \
            "api_eval_sentiment.py should support DeepSeek model"
        assert "anthropic" not in content.lower() and "claude" not in content.lower(), \
            "api_eval_sentiment.py should NOT use Claude/Anthropic (uses SiliconFlow)"

    def test_test_3cls_for_reporter(self):
        assert script_exists("data/processed/test_3cls.json"), \
            "test_3cls.json missing — reporter cannot run without evaluation data"

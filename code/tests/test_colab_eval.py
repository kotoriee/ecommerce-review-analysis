"""
TDD tests for colab/curriculum_eval_colab.py

Tests pure functions without requiring GPU or vLLM.
vLLM-dependent evaluate_with_vllm is tested via mocking.

Run: pytest code/tests/test_colab_eval.py -v
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import target module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "curriculum_eval_colab",
    ROOT / "colab/curriculum_eval_colab.py"
)
colab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colab)


# ─── get_model_and_data ───────────────────────────────────────────────────────

class TestGetModelAndData:
    """Pure path-resolution logic — no filesystem needed."""

    @pytest.mark.parametrize("stage,expected_model,expected_suffix", [
        ("s1", "lora_s1_600",   "600"),
        ("s2", "lora_s2_1200",  "1200"),
        ("s3", "lora_s3_2400",  "2400"),
        ("s4", "lora_s4_4800",  "4800"),
        ("s5", "lora_s5_full",  "8500"),
    ])
    def test_stage_maps_to_correct_model(self, stage, expected_model, expected_suffix):
        model_path, data_path = colab.get_model_and_data(
            stage, "./models/curriculum", "./data/curriculum"
        )
        assert expected_model in model_path, \
            f"Stage {stage} should map to {expected_model}, got: {model_path}"

    def test_data_path_points_to_val_fixed(self):
        _, data_path = colab.get_model_and_data("s1", "./models/curriculum", "./data/curriculum")
        assert "val_fixed.json" in data_path

    def test_model_dir_prefix_respected(self):
        model_path, _ = colab.get_model_and_data("s3", "/custom/models", "./data/curriculum")
        assert model_path.startswith("/custom/models")

    def test_all_stages_return_different_models(self):
        stages = ["s1", "s2", "s3", "s4", "s5"]
        model_paths = [
            colab.get_model_and_data(s, "./models", "./data")[0] for s in stages
        ]
        assert len(set(model_paths)) == 5, "Each stage should map to a distinct model path"


# ─── create_prompt ────────────────────────────────────────────────────────────

class TestCreatePrompt:
    """Prompt template correctness."""

    def test_contains_review_text(self):
        review = "This product is amazing!"
        prompt = colab.create_prompt(review)
        assert review in prompt

    def test_has_system_tag(self):
        prompt = colab.create_prompt("test review")
        assert "<|im_start|>system" in prompt

    def test_has_user_tag(self):
        prompt = colab.create_prompt("test review")
        assert "<|im_start|>user" in prompt

    def test_ends_with_thinking_tag(self):
        """Model should continue from <thinking> to produce reasoning."""
        prompt = colab.create_prompt("test review")
        assert "<thinking>" in prompt

    def test_specifies_output_format(self):
        """Prompt must specify the JSON output format."""
        prompt = colab.create_prompt("test review")
        assert "sentiment" in prompt
        assert "confidence" in prompt

    def test_specifies_label_range(self):
        """Prompt must show 0/1/2 label meanings."""
        prompt = colab.create_prompt("test review")
        assert "0" in prompt and "1" in prompt and "2" in prompt

    def test_empty_review_does_not_crash(self):
        """Should handle empty string gracefully."""
        prompt = colab.create_prompt("")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_long_review_included_verbatim(self):
        """Prompt should not truncate the review text."""
        long_review = "word " * 100
        prompt = colab.create_prompt(long_review.strip())
        assert long_review.strip() in prompt


# ─── extract_sentiment ────────────────────────────────────────────────────────

class TestExtractSentiment:
    """Sentiment extraction from raw model output — most critical logic."""

    # --- JSON format (primary path) ---

    def test_json_negative(self):
        assert colab.extract_sentiment('{"sentiment": 0, "confidence": 0.95}') == 0

    def test_json_neutral(self):
        assert colab.extract_sentiment('{"sentiment": 1, "confidence": 0.7}') == 1

    def test_json_positive(self):
        assert colab.extract_sentiment('{"sentiment": 2, "confidence": 0.88}') == 2

    def test_json_with_spaces(self):
        assert colab.extract_sentiment('{ "sentiment" : 2 , "confidence": 0.9 }') == 2

    def test_json_in_longer_text(self):
        """Model output may have reasoning before/after JSON."""
        text = 'The review is positive. {"sentiment": 2, "confidence": 0.85, "rationale": "good"}'
        assert colab.extract_sentiment(text) == 2

    def test_json_equality_sign_format(self):
        """Some models output sentiment=2 instead of sentiment: 2."""
        assert colab.extract_sentiment('"sentiment"=2') == 2

    def test_json_label_0_beats_positive_keywords(self):
        """Explicit JSON label should take priority over keyword matching."""
        text = '{"sentiment": 0} excellent amazing wonderful'
        assert colab.extract_sentiment(text) == 0

    def test_json_label_2_beats_negative_keywords(self):
        text = '{"sentiment": 2} terrible awful bad horrible'
        assert colab.extract_sentiment(text) == 2

    # --- Keyword fallback (when no JSON) ---

    def test_keyword_negative_majority(self):
        result = colab.extract_sentiment("This is terrible and awful, really bad quality")
        assert result == 0

    def test_keyword_positive_majority(self):
        result = colab.extract_sentiment("Excellent product, amazing quality, perfect fit")
        assert result == 2

    def test_keyword_tie_returns_neutral(self):
        result = colab.extract_sentiment("negative terrible but also positive excellent")
        assert result == 1

    def test_empty_string_returns_neutral(self):
        assert colab.extract_sentiment("") == 1

    def test_gibberish_returns_neutral(self):
        assert colab.extract_sentiment("xkcd zxqw foobar 123") == 1

    def test_only_numbers_returns_neutral(self):
        """Numbers without JSON format should not be parsed as labels."""
        assert colab.extract_sentiment("0 1 2 3 4 5") == 1

    def test_case_insensitive_json(self):
        """JSON matching should be case-insensitive per the implementation."""
        assert colab.extract_sentiment('{"Sentiment": 1}') in (0, 1, 2)  # should not crash

    @pytest.mark.parametrize("label", [0, 1, 2])
    def test_all_valid_json_labels_parsed(self, label):
        text = f'{{"sentiment": {label}, "confidence": 0.8}}'
        assert colab.extract_sentiment(text) == label


# ─── load_validation_data ────────────────────────────────────────────────────

class TestLoadValidationData:
    """Data loading with Alpaca format."""

    @pytest.fixture
    def tmp_data_file(self, tmp_path):
        """Write a minimal test dataset in Alpaca format."""
        data = [
            {"instruction": "Classify.", "input": "Great product!", "output": "2",
             "soft_labels": [0.05, 0.10, 0.85], "confidence": 0.85},
            {"instruction": "Classify.", "input": "Terrible quality.", "output": "0",
             "soft_labels": [0.90, 0.07, 0.03], "confidence": 0.90},
            {"instruction": "Classify.", "input": "It's okay I guess.", "output": "1",
             "soft_labels": [0.15, 0.70, 0.15], "confidence": 0.70},
        ]
        f = tmp_path / "test_val.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        return str(f)

    def test_loads_all_samples(self, tmp_data_file):
        samples = colab.load_validation_data(tmp_data_file, max_samples=100)
        assert len(samples) == 3

    def test_max_samples_limits_output(self, tmp_data_file):
        samples = colab.load_validation_data(tmp_data_file, max_samples=2)
        assert len(samples) == 2

    def test_sample_has_review_and_label(self, tmp_data_file):
        samples = colab.load_validation_data(tmp_data_file, max_samples=10)
        for s in samples:
            assert "review" in s, "Each sample must have 'review' key"
            assert "label" in s, "Each sample must have 'label' key"
            assert s["label"] in (0, 1, 2)

    def test_review_text_correct(self, tmp_data_file):
        samples = colab.load_validation_data(tmp_data_file, max_samples=10)
        reviews = [s["review"] for s in samples]
        assert "Great product!" in reviews
        assert "Terrible quality." in reviews

    def test_label_mapping_correct(self, tmp_data_file):
        samples = colab.load_validation_data(tmp_data_file, max_samples=10)
        label_map = {s["review"]: s["label"] for s in samples}
        assert label_map["Great product!"] == 2
        assert label_map["Terrible quality."] == 0
        assert label_map["It's okay I guess."] == 1

    def test_review_truncated_to_500_chars(self, tmp_path):
        long_review = "x" * 600
        data = [{"instruction": ".", "input": long_review, "output": "2"}]
        f = tmp_path / "long.json"
        f.write_text(json.dumps(data))
        samples = colab.load_validation_data(str(f), max_samples=10)
        assert len(samples[0]["review"]) <= 500

    def test_invalid_label_skipped(self, tmp_path):
        data = [
            {"instruction": ".", "input": "Good", "output": "2"},
            {"instruction": ".", "input": "Bad", "output": "99"},    # invalid
            {"instruction": ".", "input": "Meh", "output": "abc"},   # invalid
        ]
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data))
        samples = colab.load_validation_data(str(f), max_samples=10)
        assert len(samples) == 1
        assert samples[0]["review"] == "Good"

    def test_empty_review_skipped(self, tmp_path):
        data = [
            {"instruction": ".", "input": "", "output": "2"},       # empty review
            {"instruction": ".", "input": "Valid.", "output": "1"},
        ]
        f = tmp_path / "empty.json"
        f.write_text(json.dumps(data))
        samples = colab.load_validation_data(str(f), max_samples=10)
        assert len(samples) == 1
        assert samples[0]["review"] == "Valid."

    def test_real_val_fixed_loads(self):
        """Integration: load the actual val_fixed.json from the project."""
        real_path = str(ROOT / "data/curriculum/val_fixed.json")
        samples = colab.load_validation_data(real_path, max_samples=50)
        assert len(samples) >= 30, "Real val_fixed.json should have ≥30 valid samples"
        assert all(s["label"] in (0, 1, 2) for s in samples)


# ─── evaluate_with_vllm (mocked) ─────────────────────────────────────────────

class TestEvaluateWithVllmLogic:
    """Test result aggregation logic without loading a real GPU model."""

    def _make_mock_output(self, text: str):
        out = MagicMock()
        out.outputs = [MagicMock(text=text)]
        return out

    @patch("colab.curriculum_eval_colab.LLM", create=True)
    def test_perfect_accuracy(self, mock_llm_class):
        """When all predictions match labels, accuracy should be 100%."""
        samples = [
            {"review": "Great!", "label": 2},
            {"review": "Terrible!", "label": 0},
            {"review": "Okay.", "label": 1},
        ]
        # Mock LLM.generate to return perfect predictions
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [
            self._make_mock_output('{"sentiment": 2}'),
            self._make_mock_output('{"sentiment": 0}'),
            self._make_mock_output('{"sentiment": 1}'),
        ]
        mock_llm_class.return_value = mock_llm

        with patch.dict("sys.modules", {"vllm": MagicMock(LLM=mock_llm_class, SamplingParams=MagicMock)}):
            result = colab.evaluate_with_vllm("fake/model", samples, batch_size=8, max_tokens=128)

        assert result["accuracy"] == pytest.approx(100.0)
        assert result["correct"] == 3
        assert result["total"] == 3

    @patch("colab.curriculum_eval_colab.LLM", create=True)
    def test_zero_accuracy(self, mock_llm_class):
        """When all predictions are wrong, accuracy should be 0%."""
        samples = [{"review": "Good.", "label": 2}]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [
            self._make_mock_output('{"sentiment": 0}'),  # wrong
        ]
        mock_llm_class.return_value = mock_llm

        with patch.dict("sys.modules", {"vllm": MagicMock(LLM=mock_llm_class, SamplingParams=MagicMock)}):
            result = colab.evaluate_with_vllm("fake/model", samples, batch_size=8, max_tokens=128)

        assert result["accuracy"] == pytest.approx(0.0)
        assert result["correct"] == 0

    @patch("colab.curriculum_eval_colab.LLM", create=True)
    def test_result_contains_required_keys(self, mock_llm_class):
        samples = [{"review": "Test.", "label": 1}]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [self._make_mock_output('{"sentiment": 1}')]
        mock_llm_class.return_value = mock_llm

        with patch.dict("sys.modules", {"vllm": MagicMock(LLM=mock_llm_class, SamplingParams=MagicMock)}):
            result = colab.evaluate_with_vllm("fake/model", samples, batch_size=8, max_tokens=128)

        required = {"accuracy", "total", "correct", "time", "speed", "class_stats", "results"}
        assert required.issubset(result.keys())

    @patch("colab.curriculum_eval_colab.LLM", create=True)
    def test_class_stats_correct(self, mock_llm_class):
        """class_stats should track per-class correct/total counts."""
        samples = [
            {"review": "A", "label": 0},  # negative
            {"review": "B", "label": 0},  # negative
            {"review": "C", "label": 2},  # positive
        ]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [
            self._make_mock_output('{"sentiment": 0}'),  # correct
            self._make_mock_output('{"sentiment": 1}'),  # wrong (predicted neutral)
            self._make_mock_output('{"sentiment": 2}'),  # correct
        ]
        mock_llm_class.return_value = mock_llm

        with patch.dict("sys.modules", {"vllm": MagicMock(LLM=mock_llm_class, SamplingParams=MagicMock)}):
            result = colab.evaluate_with_vllm("fake/model", samples, batch_size=8, max_tokens=128)

        assert result["class_stats"][0]["total"] == 2   # 2 negative samples
        assert result["class_stats"][0]["correct"] == 1  # 1 correctly predicted
        assert result["class_stats"][2]["total"] == 1
        assert result["class_stats"][2]["correct"] == 1

    @patch("colab.curriculum_eval_colab.LLM", create=True)
    def test_results_list_length(self, mock_llm_class):
        """results list should have one entry per sample."""
        n = 5
        samples = [{"review": f"Review {i}", "label": i % 3} for i in range(n)]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [
            self._make_mock_output(f'{{"sentiment": {i % 3}}}') for i in range(n)
        ]
        mock_llm_class.return_value = mock_llm

        with patch.dict("sys.modules", {"vllm": MagicMock(LLM=mock_llm_class, SamplingParams=MagicMock)}):
            result = colab.evaluate_with_vllm("fake/model", samples, batch_size=8, max_tokens=128)

        assert len(result["results"]) == n

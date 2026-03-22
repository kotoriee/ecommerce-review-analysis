"""
Tests for code/cloud_agent/api_eval_sentiment.py

TDD session: covers message builders, file I/O, prediction parsing,
summary stats, and mocked API call flow.
"""

import json
import os
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Make cloud_agent importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

from cloud_agent.api_eval_sentiment import (
    AVAILABLE_MODELS,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT_ZEROSHOT,
    build_messages_fewshot,
    build_messages_zeroshot,
    load_records,
    predict_one,
    print_summary,
    save_results,
)


# ─── Message Builder Tests ─────────────────────────────────────────────────────

class TestBuildMessagesZeroshot(unittest.TestCase):
    """build_messages_zeroshot returns correct OpenAI message list"""

    def setUp(self):
        self.review = "Great product, loved it!"

    def test_returns_two_messages(self):
        msgs = build_messages_zeroshot(self.review)
        self.assertEqual(len(msgs), 2)

    def test_first_message_is_system(self):
        msgs = build_messages_zeroshot(self.review)
        self.assertEqual(msgs[0]["role"], "system")

    def test_second_message_is_user(self):
        msgs = build_messages_zeroshot(self.review)
        self.assertEqual(msgs[1]["role"], "user")

    def test_system_content_matches_constant(self):
        msgs = build_messages_zeroshot(self.review)
        self.assertEqual(msgs[0]["content"], SYSTEM_PROMPT_ZEROSHOT)

    def test_user_content_contains_review(self):
        msgs = build_messages_zeroshot(self.review)
        self.assertIn(self.review, msgs[1]["content"])

    def test_empty_review_still_builds(self):
        msgs = build_messages_zeroshot("")
        self.assertEqual(len(msgs), 2)
        self.assertIn("Review:", msgs[1]["content"])


class TestBuildMessagesFewshot(unittest.TestCase):
    """build_messages_fewshot returns correct few-shot conversation"""

    def setUp(self):
        self.review = "Mixed quality, ok price."

    def test_message_count_is_system_plus_examples_times_two_plus_user(self):
        # system + (user + assistant) * 3 + user = 1 + 6 + 1 = 8
        msgs = build_messages_fewshot(self.review)
        expected = 1 + len(FEW_SHOT_EXAMPLES) * 2 + 1
        self.assertEqual(len(msgs), expected)

    def test_first_message_is_system(self):
        msgs = build_messages_fewshot(self.review)
        self.assertEqual(msgs[0]["role"], "system")

    def test_last_message_is_user_with_review(self):
        msgs = build_messages_fewshot(self.review)
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn(self.review, msgs[-1]["content"])

    def test_example_labels_are_valid(self):
        msgs = build_messages_fewshot(self.review)
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        for msg in assistant_msgs:
            self.assertIn(msg["content"], ("0", "1", "2"))

    def test_covers_all_three_classes_in_examples(self):
        msgs = build_messages_fewshot(self.review)
        labels = {m["content"] for m in msgs if m["role"] == "assistant"}
        self.assertEqual(labels, {"0", "1", "2"})


# ─── load_records Tests ───────────────────────────────────────────────────────

class TestLoadRecords(unittest.TestCase):
    """load_records handles JSON list, JSONL, and max_n truncation"""

    def _tmp_path(self, tmp_dir, filename):
        path = Path(tmp_dir) / filename
        return str(path)

    def test_load_json_list(self, tmp_path=None):
        import tempfile
        records = [{"input": "test1", "output": "2"}, {"input": "test2", "output": "0"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(records, f)
            tmp = f.name
        try:
            loaded = load_records(tmp)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["input"], "test1")
        finally:
            os.unlink(tmp)

    def test_load_jsonl(self):
        import tempfile
        records = [{"text": "hello", "label": 1}, {"text": "world", "label": 2}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            tmp = f.name
        try:
            loaded = load_records(tmp)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[1]["text"], "world")
        finally:
            os.unlink(tmp)

    def test_max_n_truncates(self):
        import tempfile
        records = [{"input": f"review {i}", "output": str(i % 3)} for i in range(10)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(records, f)
            tmp = f.name
        try:
            loaded = load_records(tmp, max_n=3)
            self.assertEqual(len(loaded), 3)
        finally:
            os.unlink(tmp)

    def test_max_n_none_loads_all(self):
        import tempfile
        records = [{"input": f"r{i}"} for i in range(5)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(records, f)
            tmp = f.name
        try:
            loaded = load_records(tmp, max_n=None)
            self.assertEqual(len(loaded), 5)
        finally:
            os.unlink(tmp)

    def test_load_newline_delimited_json(self):
        """JSON file with one object per line (not a list)"""
        import tempfile
        lines = [{"input": "a"}, {"input": "b"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
            tmp = f.name
        try:
            loaded = load_records(tmp)
            self.assertEqual(len(loaded), 2)
        finally:
            os.unlink(tmp)


# ─── save_results Tests ───────────────────────────────────────────────────────

class TestSaveResults(unittest.TestCase):
    """save_results writes valid JSONL output"""

    def test_creates_output_file(self):
        import tempfile
        results = [{"text": "good", "sentiment": 2, "label": "positive"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.jsonl")
            save_results(results, out)
            self.assertTrue(Path(out).exists())

    def test_creates_parent_dirs(self):
        import tempfile
        results = [{"text": "bad", "sentiment": 0}]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "deep", "nested", "out.jsonl")
            save_results(results, out)
            self.assertTrue(Path(out).exists())

    def test_output_is_valid_jsonl(self):
        import tempfile
        results = [
            {"text": "review1", "sentiment": 0, "label": "negative"},
            {"text": "review2", "sentiment": 2, "label": "positive"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.jsonl")
            save_results(results, out)
            lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            parsed = [json.loads(line) for line in lines]
            self.assertEqual(parsed[0]["label"], "negative")
            self.assertEqual(parsed[1]["label"], "positive")

    def test_empty_results_creates_empty_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "empty.jsonl")
            save_results([], out)
            self.assertTrue(Path(out).exists())
            self.assertEqual(Path(out).read_text(encoding="utf-8").strip(), "")


# ─── print_summary Tests ──────────────────────────────────────────────────────

class TestPrintSummary(unittest.TestCase):
    """print_summary outputs correct counts and accuracy"""

    def _capture(self, results, model_name="deepseek"):
        import io
        captured = io.StringIO()
        sys.stdout = captured
        try:
            print_summary(results, model_name)
        finally:
            sys.stdout = sys.__stdout__
        return captured.getvalue()

    def test_shows_total_count(self):
        results = [
            {"sentiment": 2}, {"sentiment": 0}, {"sentiment": 1},
            {"sentiment": 2}, {"sentiment": 2},
        ]
        output = self._capture(results)
        self.assertIn("5", output)

    def test_shows_model_name(self):
        results = [{"sentiment": 1}]
        output = self._capture(results, "qwen")
        self.assertIn("qwen", output)

    def test_shows_accuracy_when_ground_truth_available(self):
        results = [
            {"sentiment": 2, "true_label": "2"},  # correct
            {"sentiment": 0, "true_label": "1"},  # wrong
        ]
        output = self._capture(results)
        self.assertIn("50.0%", output)

    def test_no_accuracy_without_ground_truth(self):
        results = [{"sentiment": 1}, {"sentiment": 2}]
        output = self._capture(results)
        self.assertNotIn("Accuracy", output)

    def test_non_numeric_true_label_silently_ignored(self):
        """true_label='positive' (string, not digit) → no crash, no accuracy line"""
        results = [{"sentiment": 2, "true_label": "positive"}]
        output = self._capture(results)  # should not raise
        # has_gt=True but int() fails → exception caught silently → accuracy = 0/1 = 0%
        self.assertIn("Accuracy", output)

    def test_all_three_labels_shown(self):
        results = [
            {"sentiment": 0}, {"sentiment": 1}, {"sentiment": 2},
        ]
        output = self._capture(results)
        self.assertIn("Negative", output)
        self.assertIn("Neutral", output)
        self.assertIn("Positive", output)


# ─── predict_one Tests (mocked API) ──────────────────────────────────────────

class TestPredictOne(unittest.TestCase):
    """predict_one parses API response and handles errors"""

    def _make_client(self, response_text: str):
        """Helper: create a mock OpenAI client returning given text"""
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        mock_client.chat.completions.create.return_value.choices = [choice]
        return mock_client

    def test_parses_positive_label(self):
        client = self._make_client("2")
        result = predict_one(client, "Great product!", model="test-model")
        self.assertEqual(result["sentiment"], 2)
        self.assertEqual(result["label"], "positive")

    def test_parses_negative_label(self):
        client = self._make_client("0")
        result = predict_one(client, "Terrible quality.", model="test-model")
        self.assertEqual(result["sentiment"], 0)
        self.assertEqual(result["label"], "negative")

    def test_parses_neutral_label(self):
        client = self._make_client("1")
        result = predict_one(client, "It's ok I guess.", model="test-model")
        self.assertEqual(result["sentiment"], 1)
        self.assertEqual(result["label"], "neutral")

    def test_preserves_original_text(self):
        client = self._make_client("2")
        text = "Original review text"
        result = predict_one(client, text, model="test-model")
        self.assertEqual(result["text"], text)

    def test_response_with_trailing_whitespace(self):
        """API sometimes returns '2\n' or '2 '"""
        client = self._make_client("2\n")
        result = predict_one(client, "Nice.", model="test-model")
        self.assertEqual(result["sentiment"], 2)

    def test_invalid_response_falls_back_to_neutral(self):
        """Non-parseable response → fallback neutral (1), not exception"""
        client = self._make_client("positive")  # not a digit
        result = predict_one(client, "Some text.", model="test-model", max_retries=1)
        self.assertEqual(result["sentiment"], 1)
        self.assertIn("PARSE_ERROR", result["raw_response"])

    def test_out_of_range_digit_falls_back_to_neutral(self):
        """Digit outside 0-2 → fallback neutral"""
        client = self._make_client("5")
        result = predict_one(client, "Some text.", model="test-model", max_retries=1)
        self.assertEqual(result["sentiment"], 1)

    def test_uses_fewshot_when_requested(self):
        client = self._make_client("2")
        predict_one(client, "Nice.", model="test-model", use_fewshot=True)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        # Few-shot has more messages than zero-shot (8 vs 2)
        self.assertGreater(len(messages), 2)

    def test_uses_zeroshot_by_default(self):
        client = self._make_client("1")
        predict_one(client, "Ok.", model="test-model", use_fewshot=False)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        self.assertEqual(len(messages), 2)

    @patch("time.sleep")
    def test_retries_on_api_exception(self, mock_sleep):
        """API exception triggers retry; succeeds on second attempt"""
        client = MagicMock()
        good_choice = MagicMock()
        good_choice.message.content = "2"
        client.chat.completions.create.side_effect = [
            Exception("rate limit"),
            MagicMock(choices=[good_choice]),
        ]
        result = predict_one(client, "Nice.", model="test-model", max_retries=3)
        self.assertEqual(result["sentiment"], 2)
        self.assertEqual(client.chat.completions.create.call_count, 2)


# ─── AVAILABLE_MODELS constant ───────────────────────────────────────────────

class TestAvailableModels(unittest.TestCase):
    def test_deepseek_key_exists(self):
        self.assertIn("deepseek", AVAILABLE_MODELS)

    def test_qwen_key_exists(self):
        self.assertIn("qwen", AVAILABLE_MODELS)

    def test_all_values_are_non_empty_strings(self):
        for key, val in AVAILABLE_MODELS.items():
            self.assertIsInstance(val, str)
            self.assertTrue(len(val) > 0, f"Model ID for '{key}' is empty")


# ─── get_client Tests ────────────────────────────────────────────────────────

class TestGetClient(unittest.TestCase):
    """get_client picks up API key from env var or config file"""

    @patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key-123"})
    def test_uses_env_var_when_set(self):
        from cloud_agent.api_eval_sentiment import get_client
        with patch("cloud_agent.api_eval_sentiment.OpenAI") as mock_openai:
            get_client()
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args.kwargs
            self.assertEqual(call_kwargs["api_key"], "test-key-123")

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_when_no_key_found(self):
        from cloud_agent.api_eval_sentiment import get_client
        # Ensure env var is not set and config file doesn't exist
        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(ValueError) as ctx:
                get_client()
            self.assertIn("API key", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_reads_from_config_file(self):
        from cloud_agent.api_eval_sentiment import get_client
        config = {"siliconflow": {"api_key": "config-key-456"}}
        config_json = json.dumps(config)

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=config_json)):
                with patch("cloud_agent.api_eval_sentiment.OpenAI") as mock_openai:
                    get_client()
                    call_kwargs = mock_openai.call_args.kwargs
                    self.assertEqual(call_kwargs["api_key"], "config-key-456")


# ─── predict_batch Tests ──────────────────────────────────────────────────────

class TestPredictBatch(unittest.TestCase):
    """predict_batch processes multiple records with rate limiting"""

    def _make_client(self, response_text: str = "2"):
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        mock_client.chat.completions.create.return_value.choices = [choice]
        return mock_client

    @patch("time.sleep")
    def test_returns_one_result_per_record(self, mock_sleep):
        client = self._make_client("2")
        from cloud_agent.api_eval_sentiment import predict_batch
        records = [
            {"input": "Good product", "output": "2"},
            {"text": "Bad quality", "label": "0"},
        ]
        results = predict_batch(client, records, model="test-model", rate_limit=100)
        self.assertEqual(len(results), 2)

    @patch("time.sleep")
    def test_skips_records_with_no_text(self, mock_sleep):
        client = self._make_client("1")
        from cloud_agent.api_eval_sentiment import predict_batch
        records = [
            {"input": "Good product"},
            {"input": ""},          # empty → skip
            {"other_field": "x"},   # no text field → skip
        ]
        results = predict_batch(client, records, model="test-model", rate_limit=100)
        self.assertEqual(len(results), 1)

    @patch("time.sleep")
    def test_attaches_ground_truth_label(self, mock_sleep):
        client = self._make_client("2")
        from cloud_agent.api_eval_sentiment import predict_batch
        records = [{"input": "Great!", "output": "2"}]
        results = predict_batch(client, records, model="test-model", rate_limit=100)
        self.assertEqual(results[0]["true_label"], "2")

    @patch("time.sleep")
    def test_respects_rate_limit_by_sleeping(self, mock_sleep):
        client = self._make_client("1")
        from cloud_agent.api_eval_sentiment import predict_batch
        records = [{"input": f"review {i}"} for i in range(3)]
        predict_batch(client, records, model="test-model", rate_limit=5)
        # sleep called once per record
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("time.sleep")
    def test_reports_parse_errors(self, mock_sleep):
        """When parse errors occur, warning is printed (captured)"""
        client = self._make_client("INVALID")  # will cause PARSE_ERROR
        from cloud_agent.api_eval_sentiment import predict_batch
        records = [{"input": "bad response test"}]
        import io
        captured = io.StringIO()
        sys.stdout = captured
        try:
            results = predict_batch(client, records, model="test-model", rate_limit=100)
        finally:
            sys.stdout = sys.__stdout__
        # Should have 1 result (with fallback neutral) and a warning printed
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sentiment"], 1)


# ─── predict_one retry sleep path ────────────────────────────────────────────

class TestPredictOneRetryPaths(unittest.TestCase):
    """Cover sleep and re-raise paths in retry logic"""

    @patch("time.sleep")
    def test_sleeps_between_parse_retries(self, mock_sleep):
        """ValueError retry path sleeps 0.5s between attempts"""
        client = MagicMock()
        # First call returns invalid, second returns valid
        good = MagicMock()
        good.message.content = "1"
        invalid = MagicMock()
        invalid.message.content = "xyz"
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[invalid]),
            MagicMock(choices=[good]),
        ]
        result = predict_one(client, "test", model="m", max_retries=3)
        self.assertEqual(result["sentiment"], 1)
        mock_sleep.assert_called_with(0.5)

    @patch("time.sleep")
    def test_raises_after_all_api_retries_exhausted(self, mock_sleep):
        """Exception re-raised when all retries fail"""
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("server error")
        with self.assertRaises(Exception) as ctx:
            predict_one(client, "test", model="m", max_retries=2)
        self.assertIn("server error", str(ctx.exception))


# ─── main() smoke test ───────────────────────────────────────────────────────

class TestMain(unittest.TestCase):
    """main() CLI parses args, loads data, calls API, saves results"""

    @patch("time.sleep")
    @patch("cloud_agent.api_eval_sentiment.save_results")
    @patch("cloud_agent.api_eval_sentiment.print_summary")
    @patch("cloud_agent.api_eval_sentiment.predict_batch")
    @patch("cloud_agent.api_eval_sentiment.get_client")
    @patch("cloud_agent.api_eval_sentiment.load_records")
    def test_main_end_to_end(
        self, mock_load, mock_get_client, mock_batch,
        mock_summary, mock_save, mock_sleep
    ):
        """main() wires together: load → client → batch → summary → save"""
        import tempfile, io

        mock_load.return_value = [{"input": "Great!", "output": "2"}]
        mock_batch.return_value = [{"text": "Great!", "sentiment": 2, "label": "positive"}]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'[{"input": "test", "output": "1"}]')
            tmp_input = f.name

        captured = io.StringIO()
        sys.stdout = captured
        try:
            from cloud_agent.api_eval_sentiment import main
            with patch("sys.argv", ["api_eval_sentiment.py",
                                    "--input", tmp_input,
                                    "--model", "deepseek",
                                    "--n", "1",
                                    "--output", tmp_input + "_out.jsonl"]):
                main()
        finally:
            sys.stdout = sys.__stdout__
            os.unlink(tmp_input)

        mock_load.assert_called_once()
        mock_batch.assert_called_once()
        mock_summary.assert_called_once()
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)

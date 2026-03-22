#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDD tests for code/evaluation/metrics.py
"""

import json
import tempfile
import os
import pytest
from pathlib import Path

# Allow running from any directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    compute_metrics,
    print_report,
    load_predictions,
    load_ground_truth,
    LABEL_NAMES,
    main,
)


# ─── compute_metrics ─────────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_predictions_all_classes(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [1, 2, 0, 2, 0, 1]
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_accuracy_calculation(self):
        # 4 correct out of 6
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 0, 0]  # last 2 wrong
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        m = compute_metrics(y_true, y_pred)
        assert len(m["confusion_matrix"]) == 3
        assert all(len(row) == 3 for row in m["confusion_matrix"])

    def test_confusion_matrix_values(self):
        # True: [0,0,1,1,2,2], Pred: [0,1,1,1,2,2]
        # 0→0:1, 0→1:1, 1→1:2, 2→2:2
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 2]
        m = compute_metrics(y_true, y_pred)
        cm = m["confusion_matrix"]
        assert cm[0][0] == 1   # TN for class 0
        assert cm[0][1] == 1   # class 0 predicted as 1
        assert cm[1][1] == 2   # correct class 1
        assert cm[2][2] == 2   # correct class 2

    def test_per_class_keys(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        m = compute_metrics(y_true, y_pred)
        assert "Negative" in m["per_class"]
        assert "Neutral" in m["per_class"]
        assert "Positive" in m["per_class"]

    def test_per_class_fields(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        m = compute_metrics(y_true, y_pred)
        for cls in m["per_class"].values():
            assert "precision" in cls
            assert "recall" in cls
            assert "f1" in cls
            assert "support" in cls

    def test_per_class_perfect_scores(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]
        m = compute_metrics(y_true, y_pred)
        for cls in m["per_class"].values():
            assert cls["precision"] == 1.0
            assert cls["recall"] == 1.0
            assert cls["f1"] == 1.0

    def test_total_samples(self):
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 0, 1]
        m = compute_metrics(y_true, y_pred)
        assert m["total_samples"] == 5

    def test_class_zero_support_no_division_error(self):
        """Class 2 (Positive) absent from y_true — should return 0.0, not crash."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        m = compute_metrics(y_true, y_pred)
        # Should not raise; Positive class gets 0 scores
        assert m["per_class"]["Positive"]["f1"] == 0.0
        assert m["per_class"]["Positive"]["support"] == 0

    def test_all_same_class_predicted(self):
        """Model predicts everything as positive (2)."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [2, 2, 2, 2, 2, 2]
        m = compute_metrics(y_true, y_pred)
        assert m["per_class"]["Positive"]["recall"] == 1.0
        assert m["per_class"]["Negative"]["recall"] == 0.0

    def test_f1_macro_is_mean_of_per_class(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 2]
        m = compute_metrics(y_true, y_pred)
        expected_macro = (
            m["per_class"]["Negative"]["f1"] +
            m["per_class"]["Neutral"]["f1"] +
            m["per_class"]["Positive"]["f1"]
        ) / 3
        assert m["f1_macro"] == pytest.approx(expected_macro, abs=1e-4)

    def test_empty_labels_returns_zero_accuracy(self):
        m = compute_metrics([], [])
        assert m["accuracy"] == 0.0
        assert m["total_samples"] == 0

    def test_output_values_are_rounded_to_4_decimals(self):
        y_true = [0, 1, 2, 0, 1, 2, 0]
        y_pred = [0, 1, 2, 1, 1, 2, 2]
        m = compute_metrics(y_true, y_pred)
        # Values should be floats rounded to 4 decimal places
        assert m["accuracy"] == round(m["accuracy"], 4)
        assert m["f1_macro"] == round(m["f1_macro"], 4)

    def test_imbalanced_dataset(self):
        """Heavy class imbalance: mostly positive."""
        y_true = [2] * 8 + [0] * 1 + [1] * 1
        y_pred = [2] * 8 + [0] * 1 + [1] * 1
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["per_class"]["Positive"]["support"] == 8

    def test_support_matches_class_count_in_y_true(self):
        y_true = [0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 0, 1, 1, 2]
        m = compute_metrics(y_true, y_pred)
        assert m["per_class"]["Negative"]["support"] == 3
        assert m["per_class"]["Neutral"]["support"] == 2
        assert m["per_class"]["Positive"]["support"] == 1


# ─── load_ground_truth ───────────────────────────────────────────────────────

class TestLoadGroundTruth:

    def _write_json(self, data: list) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False, encoding="utf-8")
        json.dump(data, f, ensure_ascii=False)
        f.close()
        return f.name

    def teardown_method(self):
        # Cleanup handled per test
        pass

    def test_load_label_field(self):
        path = self._write_json([{"label": 0}, {"label": 1}, {"label": 2}])
        try:
            labels = load_ground_truth(path)
            assert labels == [0, 1, 2]
        finally:
            os.unlink(path)

    def test_load_ground_truth_label_field(self):
        path = self._write_json([
            {"ground_truth_label": 2},
            {"ground_truth_label": 0},
        ])
        try:
            labels = load_ground_truth(path)
            assert labels == [2, 0]
        finally:
            os.unlink(path)

    def test_load_sentiment_label_field(self):
        path = self._write_json([
            {"sentiment_label": 1},
            {"sentiment_label": 2},
        ])
        try:
            labels = load_ground_truth(path)
            assert labels == [1, 2]
        finally:
            os.unlink(path)

    def test_load_output_field_string(self):
        """*_3cls.json uses output='0'/'1'/'2' (string) — must be supported."""
        path = self._write_json([
            {"input": "review A", "output": "0"},
            {"input": "review B", "output": "1"},
            {"input": "review C", "output": "2"},
        ])
        try:
            labels = load_ground_truth(path)
            assert labels == [0, 1, 2]
        finally:
            os.unlink(path)

    def test_output_field_returns_integers(self):
        """output='2' (string) must be converted to int 2."""
        path = self._write_json([{"output": "2"}])
        try:
            labels = load_ground_truth(path)
            assert labels[0] == 2
            assert isinstance(labels[0], int)
        finally:
            os.unlink(path)

    def test_field_priority_ground_truth_over_label(self):
        """ground_truth_label takes priority over label."""
        path = self._write_json([{"ground_truth_label": 0, "label": 2}])
        try:
            labels = load_ground_truth(path)
            assert labels == [0]
        finally:
            os.unlink(path)

    def test_items_without_any_label_field_are_skipped(self):
        path = self._write_json([
            {"text": "no label here"},
            {"label": 1},
        ])
        try:
            labels = load_ground_truth(path)
            assert labels == [1]
        finally:
            os.unlink(path)

    def test_mixed_field_names(self):
        path = self._write_json([
            {"label": 0},
            {"sentiment_label": 1},
            {"output": "2"},
        ])
        try:
            labels = load_ground_truth(path)
            assert labels == [0, 1, 2]
        finally:
            os.unlink(path)


# ─── load_predictions ────────────────────────────────────────────────────────

class TestLoadPredictions:

    def _write_jsonl(self, records: list) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                        delete=False, encoding="utf-8")
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def test_load_returns_list_of_dicts(self):
        path = self._write_jsonl([{"sentiment": 0}, {"sentiment": 1}])
        try:
            preds = load_predictions(path)
            assert isinstance(preds, list)
            assert all(isinstance(p, dict) for p in preds)
        finally:
            os.unlink(path)

    def test_load_preserves_all_fields(self):
        record = {"text": "good product", "sentiment": 2, "true_label": 2}
        path = self._write_jsonl([record])
        try:
            preds = load_predictions(path)
            assert preds[0]["text"] == "good product"
            assert preds[0]["sentiment"] == 2
            assert preds[0]["true_label"] == 2
        finally:
            os.unlink(path)

    def test_empty_lines_skipped(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                        delete=False, encoding="utf-8")
        f.write('{"sentiment": 0}\n\n{"sentiment": 1}\n')
        f.close()
        try:
            preds = load_predictions(f.name)
            assert len(preds) == 2
        finally:
            os.unlink(f.name)

    def test_predicted_label_field_supported(self):
        path = self._write_jsonl([{"predicted_label": 0}, {"predicted_label": 2}])
        try:
            preds = load_predictions(path)
            assert preds[0]["predicted_label"] == 0
        finally:
            os.unlink(path)


# ─── print_report ────────────────────────────────────────────────────────────

class TestPrintReport:

    def test_print_report_no_error(self, capsys):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 0, 2]
        m = compute_metrics(y_true, y_pred)
        print_report(m, "Test Report")
        captured = capsys.readouterr()
        assert "Test Report" in captured.out
        assert "F1-macro" in captured.out
        assert "Accuracy" in captured.out
        assert "Negative" in captured.out
        assert "Neutral" in captured.out
        assert "Positive" in captured.out

    def test_print_report_default_title(self, capsys):
        m = compute_metrics([0], [0])
        print_report(m)
        captured = capsys.readouterr()
        assert "Results" in captured.out


# ─── LABEL_NAMES constant ─────────────────────────────────────────────────────

class TestLabelNames:

    def test_label_names_keys(self):
        assert LABEL_NAMES[0] == "Negative"
        assert LABEL_NAMES[1] == "Neutral"
        assert LABEL_NAMES[2] == "Positive"


# ─── main() CLI ───────────────────────────────────────────────────────────────

class TestMain:

    def _make_pred_jsonl(self, records: list) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                        delete=False, encoding="utf-8")
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def _make_gt_json(self, records: list) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False, encoding="utf-8")
        json.dump(records, f)
        f.close()
        return f.name

    def test_main_with_predictions_and_ground_truth(self, tmp_path, monkeypatch, capsys):
        preds = [
            {"sentiment": 0, "true_label": 0},
            {"sentiment": 1, "true_label": 1},
            {"sentiment": 2, "true_label": 2},
        ]
        gt = [{"label": 0}, {"label": 1}, {"label": 2}]
        pred_path = self._make_pred_jsonl(preds)
        gt_path = self._make_gt_json(gt)
        out_path = str(tmp_path / "results.json")
        try:
            monkeypatch.setattr(
                "sys.argv",
                ["metrics.py",
                 "--predictions", pred_path,
                 "--ground-truth", gt_path,
                 "--output", out_path]
            )
            main()
            captured = capsys.readouterr()
            assert "F1-macro" in captured.out
            assert Path(out_path).exists()
            with open(out_path) as f:
                result = json.load(f)
            assert "qwen3_4b_finetuned" in result
        finally:
            os.unlink(pred_path)
            os.unlink(gt_path)

    def test_main_missing_predictions_file(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["metrics.py",
             "--predictions", str(tmp_path / "nonexistent.jsonl"),
             "--ground-truth", str(tmp_path / "gt.json"),
             "--output", str(tmp_path / "out.json")]
        )
        main()
        captured = capsys.readouterr()
        assert "错误" in captured.out or "不存在" in captured.out

    def test_main_with_baseline_comparison(self, tmp_path, monkeypatch, capsys):
        preds = [{"sentiment": i % 3, "true_label": i % 3} for i in range(9)]
        gt = [{"label": i % 3} for i in range(9)]
        svm_baseline = {
            "f1_macro": 0.75, "accuracy": 0.78,
            "per_class": {
                "Negative": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 3},
                "Neutral":  {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 3},
                "Positive": {"precision": 0.75, "recall": 0.75, "f1": 0.75, "support": 3},
            },
            "confusion_matrix": [[2,1,0],[0,3,0],[0,0,3]],
            "total_samples": 9,
        }

        pred_path = self._make_pred_jsonl(preds)
        gt_path = self._make_gt_json(gt)
        baseline_path = str(tmp_path / "svm.json")
        out_path = str(tmp_path / "results.json")

        with open(baseline_path, "w") as f:
            json.dump(svm_baseline, f)

        try:
            monkeypatch.setattr(
                "sys.argv",
                ["metrics.py",
                 "--predictions", pred_path,
                 "--ground-truth", gt_path,
                 "--baseline", baseline_path,
                 "--output", out_path]
            )
            main()
            with open(out_path) as f:
                result = json.load(f)
            assert "svm_baseline" in result
            assert "f1_macro_delta" in result
        finally:
            os.unlink(pred_path)
            os.unlink(gt_path)

    def test_main_length_mismatch_truncates(self, tmp_path, monkeypatch, capsys):
        # 4 predictions but 3 ground truth labels
        preds = [{"sentiment": i % 3} for i in range(4)]
        gt = [{"label": i % 3} for i in range(3)]
        pred_path = self._make_pred_jsonl(preds)
        gt_path = self._make_gt_json(gt)
        out_path = str(tmp_path / "results.json")
        try:
            monkeypatch.setattr(
                "sys.argv",
                ["metrics.py",
                 "--predictions", pred_path,
                 "--ground-truth", gt_path,
                 "--output", out_path]
            )
            main()
            captured = capsys.readouterr()
            assert "警告" in captured.out
        finally:
            os.unlink(pred_path)
            os.unlink(gt_path)

    def test_main_ground_truth_from_predictions_when_no_gt_file(self, tmp_path, monkeypatch):
        preds = [
            {"sentiment": 0, "ground_truth_label": 0},
            {"sentiment": 2, "ground_truth_label": 2},
        ]
        pred_path = self._make_pred_jsonl(preds)
        out_path = str(tmp_path / "results.json")
        nonexistent_gt = str(tmp_path / "no_gt.json")
        try:
            monkeypatch.setattr(
                "sys.argv",
                ["metrics.py",
                 "--predictions", pred_path,
                 "--ground-truth", nonexistent_gt,
                 "--output", out_path]
            )
            main()
            assert Path(out_path).exists()
        finally:
            os.unlink(pred_path)

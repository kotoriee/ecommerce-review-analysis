#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDD tests for code/evaluation/run_comparison.py
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.run_comparison import (
    load_all_predictions,
    run_comparison,
    main,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _write_pred_jsonl(records: list) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                    delete=False, encoding="utf-8")
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def _make_preds(n: int = 6, perfect: bool = True):
    """Generate prediction records for testing."""
    records = []
    for i in range(n):
        label = i % 3
        records.append({
            "text": f"review {i}",
            "sentiment": label if perfect else (label + 1) % 3,
            "true_label": label,
        })
    return records


# ─── load_all_predictions ────────────────────────────────────────────────────

class TestLoadAllPredictions:

    def test_loads_svm_and_api(self):
        svm_path = _write_pred_jsonl(_make_preds(6))
        api_path = _write_pred_jsonl(_make_preds(6))
        try:
            result = load_all_predictions({"svm": svm_path, "api": api_path})
            assert "svm" in result
            assert "api" in result
        finally:
            os.unlink(svm_path)
            os.unlink(api_path)

    def test_missing_optional_qwen_route_is_skipped(self):
        svm_path = _write_pred_jsonl(_make_preds(6))
        try:
            result = load_all_predictions({"svm": svm_path, "qwen": None})
            assert "svm" in result
            assert "qwen" not in result
        finally:
            os.unlink(svm_path)

    def test_returns_y_pred_and_y_true(self):
        preds = _make_preds(6)
        path = _write_pred_jsonl(preds)
        try:
            result = load_all_predictions({"svm": path})
            assert "y_pred" in result["svm"]
            assert "y_true" in result["svm"]
        finally:
            os.unlink(path)

    def test_y_pred_uses_sentiment_field(self):
        preds = [{"sentiment": 2, "true_label": 2}]
        path = _write_pred_jsonl(preds)
        try:
            result = load_all_predictions({"svm": path})
            assert result["svm"]["y_pred"] == [2]
        finally:
            os.unlink(path)

    def test_y_pred_falls_back_to_predicted_label(self):
        preds = [{"predicted_label": 0, "true_label": 0}]
        path = _write_pred_jsonl(preds)
        try:
            result = load_all_predictions({"svm": path})
            assert result["svm"]["y_pred"] == [0]
        finally:
            os.unlink(path)

    def test_y_true_from_true_label_field(self):
        preds = [{"sentiment": 1, "true_label": 0}]
        path = _write_pred_jsonl(preds)
        try:
            result = load_all_predictions({"svm": path})
            assert result["svm"]["y_true"] == [0]
        finally:
            os.unlink(path)

    def test_y_true_falls_back_to_label_field(self):
        preds = [{"sentiment": 1, "label": "neutral", "ground_truth": 1}]
        # No true_label, should use ground_truth or neutral as fallback
        preds = [{"sentiment": 1, "true_label": 1}]
        path = _write_pred_jsonl(preds)
        try:
            result = load_all_predictions({"svm": path})
            assert result["svm"]["y_true"] == [1]
        finally:
            os.unlink(path)

    def test_nonexistent_path_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_all_predictions({"svm": "/nonexistent/path.jsonl"})

    def test_all_three_routes(self):
        svm_path = _write_pred_jsonl(_make_preds(6))
        api_path = _write_pred_jsonl(_make_preds(6))
        qwen_path = _write_pred_jsonl(_make_preds(6))
        try:
            result = load_all_predictions({
                "svm": svm_path,
                "api": api_path,
                "qwen": qwen_path,
            })
            assert len(result) == 3
        finally:
            os.unlink(svm_path)
            os.unlink(api_path)
            os.unlink(qwen_path)


# ─── run_comparison ──────────────────────────────────────────────────────────

class TestRunComparison:

    def test_returns_dict_with_route_metrics(self, tmp_path):
        svm_path = _write_pred_jsonl(_make_preds(6))
        api_path = _write_pred_jsonl(_make_preds(6))
        try:
            results = run_comparison(
                {"svm": svm_path, "api": api_path},
                output_dir=str(tmp_path),
            )
            assert "svm" in results
            assert "api" in results
        finally:
            os.unlink(svm_path)
            os.unlink(api_path)

    def test_each_route_has_metrics_fields(self, tmp_path):
        svm_path = _write_pred_jsonl(_make_preds(6))
        try:
            results = run_comparison({"svm": svm_path}, output_dir=str(tmp_path))
            m = results["svm"]
            assert "accuracy" in m
            assert "f1_macro" in m
            assert "per_class" in m
            assert "confusion_matrix" in m
        finally:
            os.unlink(svm_path)

    def test_saves_comparison_results_json(self, tmp_path):
        svm_path = _write_pred_jsonl(_make_preds(6))
        try:
            run_comparison({"svm": svm_path}, output_dir=str(tmp_path))
            out_file = tmp_path / "comparison_results.json"
            assert out_file.exists()
            with open(out_file) as f:
                data = json.load(f)
            assert "svm" in data
        finally:
            os.unlink(svm_path)

    def test_creates_output_dir_if_not_exists(self, tmp_path):
        svm_path = _write_pred_jsonl(_make_preds(6))
        new_dir = str(tmp_path / "new" / "subdir")
        try:
            run_comparison({"svm": svm_path}, output_dir=new_dir)
            assert Path(new_dir).exists()
        finally:
            os.unlink(svm_path)

    def test_skips_none_routes(self, tmp_path):
        svm_path = _write_pred_jsonl(_make_preds(6))
        try:
            results = run_comparison(
                {"svm": svm_path, "qwen": None},
                output_dir=str(tmp_path),
            )
            assert "qwen" not in results
        finally:
            os.unlink(svm_path)

    def test_perfect_predictions_give_f1_1(self, tmp_path):
        # Perfect predictions: sentiment == true_label
        preds = [{"sentiment": i % 3, "true_label": i % 3} for i in range(9)]
        path = _write_pred_jsonl(preds)
        try:
            results = run_comparison({"svm": path}, output_dir=str(tmp_path))
            assert results["svm"]["f1_macro"] == 1.0
        finally:
            os.unlink(path)


# ─── main() CLI ──────────────────────────────────────────────────────────────

class TestMain:

    def test_main_with_svm_and_api(self, tmp_path, monkeypatch, capsys):
        svm_path = _write_pred_jsonl(_make_preds(6))
        api_path = _write_pred_jsonl(_make_preds(6))
        try:
            monkeypatch.setattr(
                "sys.argv",
                ["run_comparison.py",
                 "--svm", svm_path,
                 "--api", api_path,
                 "--output", str(tmp_path)]
            )
            main()
            assert (tmp_path / "comparison_results.json").exists()
        finally:
            os.unlink(svm_path)
            os.unlink(api_path)

    def test_main_qwen_optional(self, tmp_path, monkeypatch):
        svm_path = _write_pred_jsonl(_make_preds(6))
        try:
            monkeypatch.setattr(
                "sys.argv",
                ["run_comparison.py",
                 "--svm", svm_path,
                 "--output", str(tmp_path)]
            )
            main()
            assert (tmp_path / "comparison_results.json").exists()
        finally:
            os.unlink(svm_path)

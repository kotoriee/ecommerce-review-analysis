#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDD tests for code/evaluation/run_svm_baseline.py
"""

import json
import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.run_svm_baseline import (
    load_3cls_json,
    format_predictions,
    run_svm,
    main,
    LABEL_MAP,
)


# ─── load_3cls_json ──────────────────────────────────────────────────────────

class TestLoad3ClsJson:

    def _write_json(self, records: list) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False, encoding="utf-8")
        json.dump(records, f, ensure_ascii=False)
        f.close()
        return f.name

    def test_returns_texts_and_labels(self):
        data = [
            {"instruction": "...", "input": "Great product!", "output": "2"},
            {"instruction": "...", "input": "Terrible quality.", "output": "0"},
        ]
        path = self._write_json(data)
        try:
            texts, labels = load_3cls_json(path)
            assert texts == ["Great product!", "Terrible quality."]
            assert labels == [2, 0]
        finally:
            os.unlink(path)

    def test_labels_are_integers(self):
        data = [{"input": "ok", "output": "1"}]
        path = self._write_json(data)
        try:
            texts, labels = load_3cls_json(path)
            assert isinstance(labels[0], int)
            assert labels[0] == 1
        finally:
            os.unlink(path)

    def test_all_three_classes(self):
        data = [
            {"input": "bad", "output": "0"},
            {"input": "ok", "output": "1"},
            {"input": "great", "output": "2"},
        ]
        path = self._write_json(data)
        try:
            texts, labels = load_3cls_json(path)
            assert set(labels) == {0, 1, 2}
        finally:
            os.unlink(path)

    def test_handles_soft_labels_field(self):
        """Records may have additional fields like soft_labels — should be ignored."""
        data = [
            {"input": "nice", "output": "2", "soft_labels": [0.1, 0.1, 0.8], "confidence": 0.8},
        ]
        path = self._write_json(data)
        try:
            texts, labels = load_3cls_json(path)
            assert texts == ["nice"]
            assert labels == [2]
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty(self):
        path = self._write_json([])
        try:
            texts, labels = load_3cls_json(path)
            assert texts == []
            assert labels == []
        finally:
            os.unlink(path)

    def test_returns_tuple(self):
        data = [{"input": "test", "output": "0"}]
        path = self._write_json(data)
        try:
            result = load_3cls_json(path)
            assert isinstance(result, tuple)
            assert len(result) == 2
        finally:
            os.unlink(path)


# ─── format_predictions ──────────────────────────────────────────────────────

class TestFormatPredictions:

    def test_basic_format(self):
        texts = ["good product", "bad quality"]
        y_pred = np.array([2, 0])
        y_true = [2, 0]
        records = format_predictions(texts, y_pred, y_true)
        assert len(records) == 2
        assert records[0]["text"] == "good product"
        assert records[0]["sentiment"] == 2
        assert records[0]["label"] == "positive"
        assert records[0]["true_label"] == 2

    def test_label_map_coverage(self):
        texts = ["a", "b", "c"]
        y_pred = np.array([0, 1, 2])
        y_true = [0, 1, 2]
        records = format_predictions(texts, y_pred, y_true)
        assert records[0]["label"] == "negative"
        assert records[1]["label"] == "neutral"
        assert records[2]["label"] == "positive"

    def test_true_label_none_when_not_provided(self):
        texts = ["review"]
        y_pred = np.array([1])
        records = format_predictions(texts, y_pred, y_true=None)
        assert records[0]["true_label"] is None

    def test_output_is_list_of_dicts(self):
        texts = ["review"]
        y_pred = np.array([0])
        records = format_predictions(texts, y_pred, y_true=[0])
        assert isinstance(records, list)
        assert isinstance(records[0], dict)


# ─── run_svm ─────────────────────────────────────────────────────────────────

class TestRunSvm:

    def _write_3cls_json(self, n_per_class: int = 5) -> str:
        """Write a minimal 3-class JSON file for testing."""
        data = []
        texts = {
            0: ["terrible product", "broke after one day", "waste of money",
                "very disappointed", "not worth it"],
            1: ["average quality", "ok for the price", "nothing special",
                "does the job", "mediocre performance"],
            2: ["great product", "highly recommend", "excellent quality",
                "very satisfied", "love this item"],
        }
        for label, label_texts in texts.items():
            for text in label_texts[:n_per_class]:
                data.append({"input": text, "output": str(label)})
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False, encoding="utf-8")
        json.dump(data, f)
        f.close()
        return f.name

    @patch("evaluation.run_svm_baseline.SVMSentimentClassifier")
    def test_run_svm_returns_metrics_dict(self, MockSVM, tmp_path):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([0, 1, 2, 0, 1])
        MockSVM.return_value = mock_clf

        train_path = self._write_3cls_json(5)
        test_path = self._write_3cls_json(2)
        out_path = str(tmp_path / "svm_preds.jsonl")

        try:
            metrics = run_svm(train_path, test_path, out_path)
            assert "accuracy" in metrics
            assert "f1_macro" in metrics
            assert "per_class" in metrics
        finally:
            os.unlink(train_path)
            os.unlink(test_path)

    @patch("evaluation.run_svm_baseline.SVMSentimentClassifier")
    def test_run_svm_creates_output_jsonl(self, MockSVM, tmp_path):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([0, 1, 2, 0, 1, 2])
        MockSVM.return_value = mock_clf

        train_path = self._write_3cls_json(5)
        test_path = self._write_3cls_json(2)
        out_path = str(tmp_path / "svm_preds.jsonl")

        try:
            run_svm(train_path, test_path, out_path)
            assert Path(out_path).exists()
            with open(out_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            assert len(lines) > 0
            assert "sentiment" in lines[0]
            assert "label" in lines[0]
            assert "true_label" in lines[0]
            assert "text" in lines[0]
        finally:
            os.unlink(train_path)
            os.unlink(test_path)

    @patch("evaluation.run_svm_baseline.SVMSentimentClassifier")
    def test_run_svm_calls_fit_and_predict(self, MockSVM, tmp_path):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([0, 0, 0, 0, 0, 0])
        MockSVM.return_value = mock_clf

        train_path = self._write_3cls_json(5)
        test_path = self._write_3cls_json(2)
        out_path = str(tmp_path / "svm_preds.jsonl")

        try:
            run_svm(train_path, test_path, out_path)
            mock_clf.fit.assert_called_once()
            mock_clf.predict.assert_called_once()
        finally:
            os.unlink(train_path)
            os.unlink(test_path)

    @patch("evaluation.run_svm_baseline.SVMSentimentClassifier")
    def test_run_svm_with_model_save_path(self, MockSVM, tmp_path):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([0, 0, 0, 0, 0, 0])
        MockSVM.return_value = mock_clf

        train_path = self._write_3cls_json(5)
        test_path = self._write_3cls_json(2)
        out_path = str(tmp_path / "svm_preds.jsonl")
        model_path = str(tmp_path / "svm_model.pkl")

        try:
            run_svm(train_path, test_path, out_path, model_save_path=model_path)
            mock_clf.save.assert_called_once_with(model_path)
        finally:
            os.unlink(train_path)
            os.unlink(test_path)

    @patch("evaluation.run_svm_baseline.SVMSentimentClassifier")
    def test_run_svm_no_save_when_no_model_path(self, MockSVM, tmp_path):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([0, 0])
        MockSVM.return_value = mock_clf

        train_path = self._write_3cls_json(5)
        test_path = self._write_3cls_json(1)
        out_path = str(tmp_path / "svm_preds.jsonl")

        try:
            run_svm(train_path, test_path, out_path)
            mock_clf.save.assert_not_called()
        finally:
            os.unlink(train_path)
            os.unlink(test_path)


# ─── LABEL_MAP ───────────────────────────────────────────────────────────────

class TestLabelMap:
    def test_label_map_contents(self):
        assert LABEL_MAP[0] == "negative"
        assert LABEL_MAP[1] == "neutral"
        assert LABEL_MAP[2] == "positive"


# ─── main() CLI ──────────────────────────────────────────────────────────────

class TestMain:

    def _write_3cls_json(self, n: int = 3) -> str:
        data = [{"input": f"review {i}", "output": str(i % 3)} for i in range(n)]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False, encoding="utf-8")
        json.dump(data, f)
        f.close()
        return f.name

    @patch("evaluation.run_svm_baseline.SVMSentimentClassifier")
    def test_main_cli_creates_output(self, MockSVM, tmp_path, monkeypatch):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([0, 1, 2])
        MockSVM.return_value = mock_clf

        train_path = self._write_3cls_json(9)
        test_path = self._write_3cls_json(3)
        out_path = str(tmp_path / "svm_preds.jsonl")

        try:
            monkeypatch.setattr(
                "sys.argv",
                ["run_svm_baseline.py",
                 "--train", train_path,
                 "--test", test_path,
                 "--output", out_path]
            )
            main()
            assert Path(out_path).exists()
        finally:
            os.unlink(train_path)
            os.unlink(test_path)

    @patch("evaluation.run_svm_baseline.SVMSentimentClassifier")
    def test_main_cli_with_model_save(self, MockSVM, tmp_path, monkeypatch):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([0, 1, 2])
        MockSVM.return_value = mock_clf

        train_path = self._write_3cls_json(9)
        test_path = self._write_3cls_json(3)
        out_path = str(tmp_path / "svm_preds.jsonl")
        model_path = str(tmp_path / "svm_model.pkl")

        try:
            monkeypatch.setattr(
                "sys.argv",
                ["run_svm_baseline.py",
                 "--train", train_path,
                 "--test", test_path,
                 "--output", out_path,
                 "--save-model", model_path]
            )
            main()
            mock_clf.save.assert_called_once_with(model_path)
        finally:
            os.unlink(train_path)
            os.unlink(test_path)

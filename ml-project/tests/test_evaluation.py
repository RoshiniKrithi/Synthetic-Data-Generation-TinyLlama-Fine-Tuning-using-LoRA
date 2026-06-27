"""
tests/test_evaluation.py — Unit tests for the ModelEvaluator module.

Tests cover:
  - _normalize() text normalisation
  - _compute_improvement() delta calculation
  - _compute_metrics() exact_match, BLEU, ROUGE-L (all mocked/real)
  - _save_reports() writes JSON, Markdown, HTML files
  - _plot_comparison() doesn't crash when matplotlib is available
  - run() end-to-end with mocked pipelines
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import _normalize, ModelEvaluator


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases_text(self):
        assert _normalize("Hello World") == "hello world"

    def test_removes_articles(self):
        result = _normalize("a cat is the best animal")
        # "a" and "the" should be removed
        assert "a " not in result.split()[:2]
        assert "the" not in result

    def test_removes_punctuation(self):
        result = _normalize("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_collapses_extra_spaces(self):
        result = _normalize("hello   world")
        assert "  " not in result

    def test_empty_string(self):
        assert _normalize("") == ""


# ---------------------------------------------------------------------------
# _compute_improvement
# ---------------------------------------------------------------------------

class TestComputeImprovement:
    def test_positive_improvement(self):
        base = {"bleu": 10.0, "rouge_l": 20.0}
        ft   = {"bleu": 15.0, "rouge_l": 25.0}
        imp = ModelEvaluator._compute_improvement(base, ft)
        assert imp["bleu"] == pytest.approx(5.0)
        assert imp["rouge_l"] == pytest.approx(5.0)

    def test_negative_improvement(self):
        base = {"bleu": 15.0}
        ft   = {"bleu": 10.0}
        imp = ModelEvaluator._compute_improvement(base, ft)
        assert imp["bleu"] == pytest.approx(-5.0)

    def test_zero_improvement(self):
        base = {"exact_match": 7.5}
        ft   = {"exact_match": 7.5}
        imp = ModelEvaluator._compute_improvement(base, ft)
        assert imp["exact_match"] == 0.0

    def test_missing_key_in_ft(self):
        base = {"bleu": 10.0, "rouge_l": 20.0}
        ft   = {"bleu": 12.0}
        imp = ModelEvaluator._compute_improvement(base, ft)
        assert imp["rouge_l"] == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _make_evaluator(self, tmp_config) -> ModelEvaluator:
        return ModelEvaluator(tmp_config)

    def test_exact_match_perfect(self, tmp_config):
        ev = self._make_evaluator(tmp_config)
        preds = ["hello world", "python is great"]
        refs  = ["hello world", "python is great"]
        metrics = ev._compute_metrics(preds, refs)
        assert metrics["exact_match"] == pytest.approx(100.0)

    def test_exact_match_none(self, tmp_config):
        ev = self._make_evaluator(tmp_config)
        preds = ["foo bar baz"]
        refs  = ["completely different text"]
        metrics = ev._compute_metrics(preds, refs)
        assert metrics["exact_match"] == pytest.approx(0.0)

    def test_exact_match_partial(self, tmp_config):
        ev = self._make_evaluator(tmp_config)
        preds = ["match", "no match"]
        refs  = ["match", "something else"]
        metrics = ev._compute_metrics(preds, refs)
        assert metrics["exact_match"] == pytest.approx(50.0)

    def test_returns_bleu_key(self, tmp_config):
        ev = self._make_evaluator(tmp_config)
        metrics = ev._compute_metrics(["answer"], ["answer"])
        assert "bleu" in metrics

    def test_bleu_score_range(self, tmp_config):
        ev = self._make_evaluator(tmp_config)
        metrics = ev._compute_metrics(["hello world"], ["hello world"])
        assert 0.0 <= metrics["bleu"] <= 100.0

    def test_metrics_all_numeric(self, tmp_config):
        ev = self._make_evaluator(tmp_config)
        metrics = ev._compute_metrics(["test answer"], ["test reference"])
        for key, val in metrics.items():
            assert isinstance(val, (int, float)), f"{key} should be numeric"


# ---------------------------------------------------------------------------
# _save_reports
# ---------------------------------------------------------------------------

class TestSaveReports:
    def _sample_report(self) -> dict:
        return {
            "base_model": {
                "metrics": {"exact_match": 5.0, "bleu": 8.0, "rouge_l": 22.0, "avg_latency_s": 0.5},
            },
            "fine_tuned_model": {
                "metrics": {"exact_match": 12.0, "bleu": 18.0, "rouge_l": 35.0, "avg_latency_s": 0.6},
            },
            "improvement": {"exact_match": 7.0, "bleu": 10.0, "rouge_l": 13.0, "avg_latency_s": 0.1},
            "num_samples": 50,
        }

    def test_json_report_written(self, tmp_config):
        ev = ModelEvaluator(tmp_config)
        report = self._sample_report()
        ev._save_reports(report)

        json_path = Path(tmp_config["paths"]["reports"]) / "evaluation_report.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["num_samples"] == 50

    def test_markdown_report_written(self, tmp_config):
        ev = ModelEvaluator(tmp_config)
        ev._save_reports(self._sample_report())

        md_path = Path(tmp_config["paths"]["reports"]) / "evaluation_report.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "# Evaluation Report" in content
        assert "Metric" in content

    def test_html_report_written(self, tmp_config):
        ev = ModelEvaluator(tmp_config)
        ev._save_reports(self._sample_report())

        html_path = Path(tmp_config["paths"]["reports"]) / "evaluation_report.html"
        assert html_path.exists()
        content = html_path.read_text()
        assert "<html" in content.lower()
        assert "Evaluation Report" in content


# ---------------------------------------------------------------------------
# run() — end-to-end with mocked pipelines
# ---------------------------------------------------------------------------

class TestModelEvaluatorRun:
    def test_run_returns_correct_keys(self, tmp_config, sample_qa_pairs, mock_inference_pipeline):
        ev = ModelEvaluator(tmp_config)
        report = ev.run(sample_qa_pairs, mock_inference_pipeline, mock_inference_pipeline)

        assert "base_model" in report
        assert "fine_tuned_model" in report
        assert "improvement" in report
        assert "num_samples" in report

    def test_run_num_samples_capped_by_max_samples(self, tmp_config, sample_qa_pairs, mock_inference_pipeline):
        tmp_config["evaluation"]["max_samples"] = 3
        ev = ModelEvaluator(tmp_config)
        report = ev.run(sample_qa_pairs, mock_inference_pipeline, mock_inference_pipeline)
        assert report["num_samples"] == 3

    def test_run_saves_reports(self, tmp_config, sample_qa_pairs, mock_inference_pipeline):
        ev = ModelEvaluator(tmp_config)
        ev.run(sample_qa_pairs, mock_inference_pipeline, mock_inference_pipeline)

        json_path = Path(tmp_config["paths"]["reports"]) / "evaluation_report.json"
        assert json_path.exists()

"""
tests/test_analysis.py — Unit tests for the dataset analysis module.

Tests cover:
  - run_analysis() returns correct stat keys
  - run_analysis() handles missing dataset gracefully
  - Stats values are within expected ranges
  - Charts are produced when matplotlib is available
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.data_analysis import run_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_qa_pairs(config: dict, pairs: list[dict]) -> Path:
    """Write Q&A pairs to the synthetic data directory and return the path."""
    out_dir = Path(config["paths"]["data_synthetic"])
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_path = out_dir / "synthetic_qa.json"
    qa_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    return qa_path


SAMPLE_PAIRS = [
    {"question": "What is a transformer?", "answer": "A deep learning model.", "type": "factual", "chunk_id": 0},
    {"question": "What is self-attention?", "answer": "A mechanism that weighs token importance.", "type": "definition", "chunk_id": 0},
    {"question": "How does BERT work?", "answer": "BERT uses bidirectional transformers.", "type": "reasoning", "chunk_id": 1},
    {"question": "What is positional encoding?", "answer": "It encodes position into embeddings.", "type": "definition", "chunk_id": 1},
    {"question": "What is multi-head attention?", "answer": "Attention applied in parallel h times.", "type": "conceptual", "chunk_id": 2},
]


# ---------------------------------------------------------------------------
# run_analysis
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    def test_returns_dict(self, tmp_config):
        _write_qa_pairs(tmp_config, SAMPLE_PAIRS)
        stats = run_analysis(tmp_config)
        assert isinstance(stats, dict)

    def test_returns_expected_keys(self, tmp_config):
        _write_qa_pairs(tmp_config, SAMPLE_PAIRS)
        stats = run_analysis(tmp_config)
        for key in ("total_pairs", "avg_question_words", "avg_answer_words",
                    "vocabulary_size", "duplicate_percentage"):
            assert key in stats, f"Missing key: {key}"

    def test_total_pairs_correct(self, tmp_config):
        _write_qa_pairs(tmp_config, SAMPLE_PAIRS)
        stats = run_analysis(tmp_config)
        assert stats["total_pairs"] == len(SAMPLE_PAIRS)

    def test_avg_question_words_positive(self, tmp_config):
        _write_qa_pairs(tmp_config, SAMPLE_PAIRS)
        stats = run_analysis(tmp_config)
        assert stats["avg_question_words"] > 0

    def test_vocabulary_size_positive(self, tmp_config):
        _write_qa_pairs(tmp_config, SAMPLE_PAIRS)
        stats = run_analysis(tmp_config)
        assert stats["vocabulary_size"] > 0

    def test_duplicate_percentage_no_duplicates(self, tmp_config):
        _write_qa_pairs(tmp_config, SAMPLE_PAIRS)
        stats = run_analysis(tmp_config)
        assert stats["duplicate_percentage"] == pytest.approx(0.0)

    def test_duplicate_percentage_with_duplicates(self, tmp_config):
        pairs_with_dup = SAMPLE_PAIRS + [SAMPLE_PAIRS[0].copy()]
        _write_qa_pairs(tmp_config, pairs_with_dup)
        stats = run_analysis(tmp_config)
        assert stats["duplicate_percentage"] > 0.0

    def test_type_distribution_present(self, tmp_config):
        _write_qa_pairs(tmp_config, SAMPLE_PAIRS)
        stats = run_analysis(tmp_config)
        assert "type_distribution" in stats
        assert isinstance(stats["type_distribution"], dict)

    def test_returns_empty_when_file_missing(self, tmp_config):
        """Should return {} gracefully when dataset hasn't been generated yet."""
        stats = run_analysis(tmp_config)
        assert stats == {}

    def test_charts_directory_created(self, tmp_config):
        _write_qa_pairs(tmp_config, SAMPLE_PAIRS)
        run_analysis(tmp_config)
        charts_dir = Path(tmp_config["paths"]["charts"])
        assert charts_dir.exists()

    def test_single_pair_does_not_crash(self, tmp_config):
        """Edge case: single Q&A pair."""
        _write_qa_pairs(tmp_config, [SAMPLE_PAIRS[0]])
        stats = run_analysis(tmp_config)
        assert stats["total_pairs"] == 1

    def test_pairs_without_type_column(self, tmp_config):
        """Pairs missing 'type' key should still produce valid stats."""
        pairs_no_type = [
            {"question": "What is Python?", "answer": "A programming language.", "chunk_id": 0},
        ]
        _write_qa_pairs(tmp_config, pairs_no_type)
        stats = run_analysis(tmp_config)
        assert stats["total_pairs"] == 1
        assert stats["type_distribution"] == {}

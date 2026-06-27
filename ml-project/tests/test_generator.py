"""
tests/test_generator.py — Unit tests for the QA generator module.

Tests cover:
  - JSON response parsing (valid, invalid, partial)
  - Deduplication logic
  - Prompt template formatting
  - Ollama error handling (mocked)
  - Data saving (JSON + CSV written correctly)
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator.qa_generator import QAGenerator, _fingerprint, PROMPT_TEMPLATES


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_same_question_same_hash(self):
        q = {"question": "What is Python?", "answer": "A language."}
        assert _fingerprint(q) == _fingerprint(q)

    def test_different_questions_different_hashes(self):
        q1 = {"question": "What is Python?", "answer": "A."}
        q2 = {"question": "What is Java?", "answer": "A."}
        assert _fingerprint(q1) != _fingerprint(q2)

    def test_case_insensitive(self):
        q1 = {"question": "What is Python?", "answer": "A."}
        q2 = {"question": "what is python?", "answer": "B."}
        assert _fingerprint(q1) == _fingerprint(q2)

    def test_strips_whitespace(self):
        q1 = {"question": "  What is Python?  ", "answer": "A."}
        q2 = {"question": "What is Python?", "answer": "B."}
        assert _fingerprint(q1) == _fingerprint(q2)


# ---------------------------------------------------------------------------
# QAGenerator._parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    """Tests for the static response parser — no Ollama connection needed."""

    def test_valid_json_array(self):
        raw = json.dumps([
            {"question": "What is a transformer?", "answer": "A deep learning model.", "type": "factual"}
        ])
        pairs = QAGenerator._parse_response(raw, "factual", 0)
        assert len(pairs) == 1
        assert pairs[0]["question"] == "What is a transformer?"
        assert pairs[0]["type"] == "factual"
        assert pairs[0]["chunk_id"] == 0

    def test_json_array_embedded_in_text(self):
        raw = (
            'Sure! Here are the Q&A pairs:\n'
            '[{"question": "What is attention?", "answer": "A weighting mechanism.", "type": "conceptual"}]\n'
            'Hope that helps!'
        )
        pairs = QAGenerator._parse_response(raw, "conceptual", 1)
        assert len(pairs) == 1
        assert pairs[0]["question"] == "What is attention?"

    def test_invalid_json_returns_empty(self):
        pairs = QAGenerator._parse_response("not JSON at all", "factual", 0)
        assert pairs == []

    def test_empty_string_returns_empty(self):
        pairs = QAGenerator._parse_response("", "factual", 0)
        assert pairs == []

    def test_filters_short_question(self):
        raw = json.dumps([{"question": "Hi?", "answer": "Hello there.", "type": "factual"}])
        pairs = QAGenerator._parse_response(raw, "factual", 0)
        assert len(pairs) == 0

    def test_filters_short_answer(self):
        raw = json.dumps([{"question": "What is the transformer model?", "answer": "Hi.", "type": "factual"}])
        pairs = QAGenerator._parse_response(raw, "factual", 0)
        assert len(pairs) == 0

    def test_filters_missing_keys(self):
        raw = json.dumps([{"question": "What is Python?"}])  # no "answer"
        pairs = QAGenerator._parse_response(raw, "factual", 0)
        assert len(pairs) == 0

    def test_multiple_pairs(self):
        pairs_raw = [
            {"question": "What is the encoder?", "answer": "It maps input to representation."},
            {"question": "What is the decoder?", "answer": "It generates output from representation."},
        ]
        raw = json.dumps(pairs_raw)
        pairs = QAGenerator._parse_response(raw, "conceptual", 2)
        assert len(pairs) == 2

    def test_type_and_chunk_id_added(self):
        raw = json.dumps([{"question": "What is BERT?", "answer": "A bidirectional transformer model."}])
        pairs = QAGenerator._parse_response(raw, "definition", 5)
        assert pairs[0]["type"] == "definition"
        assert pairs[0]["chunk_id"] == 5


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    def test_all_types_present(self):
        for q_type in ["factual", "reasoning", "conceptual", "definition"]:
            assert q_type in PROMPT_TEMPLATES

    def test_template_formatting(self):
        tmpl = PROMPT_TEMPLATES["factual"]
        rendered = tmpl.format(text="Some text here.", n=3)
        assert "Some text here." in rendered
        assert "3" in rendered

    def test_template_contains_json_instruction(self):
        for template in PROMPT_TEMPLATES.values():
            assert "JSON" in template or "json" in template.lower()


# ---------------------------------------------------------------------------
# QAGenerator.run — Ollama mocked
# ---------------------------------------------------------------------------

class TestQAGeneratorRun:

    def _make_config(self, tmp_path: Path) -> dict:
        return {
            "generator": {
                "ollama_model": "mistral",
                "ollama_base_url": "http://localhost:11434",
                "qa_pairs_per_chunk": 2,
                "max_qa_pairs": 10,
                "temperature": 0.7,
                "question_types": ["factual"],
                "retry_attempts": 1,
            },
            "paths": {"data_synthetic": str(tmp_path / "synthetic")},
        }

    def _valid_ollama_response(self) -> str:
        return json.dumps([
            {"question": "What is the transformer model?", "answer": "A sequence-to-sequence deep learning model."},
            {"question": "Who invented the transformer?", "answer": "Vaswani et al. in 2017."},
        ])

    def test_run_generates_and_saves_pairs(self, tmp_path, sample_chunks):
        config = self._make_config(tmp_path)
        gen = QAGenerator(config)

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": self._valid_ollama_response()}

        with patch("src.generator.qa_generator.QAGenerator._call_ollama",
                   return_value=self._valid_ollama_response()):
            pairs = gen.run(sample_chunks[:2])

        assert len(pairs) > 0

        json_path = Path(config["paths"]["data_synthetic"]) / "synthetic_qa.json"
        csv_path = Path(config["paths"]["data_synthetic"]) / "synthetic_qa.csv"
        assert json_path.exists(), "synthetic_qa.json should be written"
        assert csv_path.exists(), "synthetic_qa.csv should be written"

    def test_run_deduplicates_identical_questions(self, tmp_path, sample_chunks):
        config = self._make_config(tmp_path)
        gen = QAGenerator(config)

        # Return the same pair twice
        duplicate_response = json.dumps([
            {"question": "What is the transformer model?", "answer": "A deep learning model."},
            {"question": "What is the transformer model?", "answer": "Same question again."},
        ])

        with patch("src.generator.qa_generator.QAGenerator._call_ollama",
                   return_value=duplicate_response):
            pairs = gen.run(sample_chunks[:1])

        questions = [p["question"] for p in pairs]
        assert len(questions) == len(set(q.strip().lower() for q in questions)), (
            "Duplicate questions should be removed"
        )

    def test_run_respects_max_qa_pairs(self, tmp_path, sample_chunks):
        """Output should not exceed max_qa_pairs."""
        config = self._make_config(tmp_path)
        config["generator"]["max_qa_pairs"] = 3
        gen = QAGenerator(config)

        long_response = json.dumps([
            {"question": f"What is concept {i}?", "answer": f"Concept {i} is important."}
            for i in range(20)
        ])

        with patch("src.generator.qa_generator.QAGenerator._call_ollama",
                   return_value=long_response):
            pairs = gen.run(sample_chunks)

        assert len(pairs) <= 3

    def test_run_continues_on_ollama_failure(self, tmp_path, sample_chunks):
        """run() should return empty list (not raise) when Ollama is down."""
        config = self._make_config(tmp_path)
        gen = QAGenerator(config)

        with patch("src.generator.qa_generator.QAGenerator._call_ollama", return_value=None):
            pairs = gen.run(sample_chunks[:1])

        assert isinstance(pairs, list)

    def test_saved_csv_has_correct_columns(self, tmp_path, sample_chunks):
        import csv
        config = self._make_config(tmp_path)
        gen = QAGenerator(config)

        with patch("src.generator.qa_generator.QAGenerator._call_ollama",
                   return_value=self._valid_ollama_response()):
            gen.run(sample_chunks[:1])

        csv_path = Path(config["paths"]["data_synthetic"]) / "synthetic_qa.csv"
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        assert "question" in rows[0]
        assert "answer" in rows[0]
        assert "type" in rows[0]

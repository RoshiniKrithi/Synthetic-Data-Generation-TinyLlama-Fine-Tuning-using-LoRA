"""
Unit tests for the Wikipedia scraper.
Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.scraper.wiki_scraper import WikipediaScraper

MINIMAL_CONFIG = {
    "scraper": {
        "wikipedia_url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "max_chunks": 5,
        "retry_attempts": 2,
        "retry_delay": 0,
    },
    "paths": {
        "data_raw": "/tmp/test_raw",
        "data_processed": "/tmp/test_proc",
    },
}

SAMPLE_HTML = """
<html><body>
<div id="mw-content-text">
  <h2><span class="mw-headline">Introduction</span></h2>
  <p>Python is a high-level programming language. It was created by Guido van Rossum.</p>
  <p>Python emphasizes code readability and supports multiple programming paradigms.</p>
  <h2><span class="mw-headline">References</span></h2>
  <p>This section should be skipped by the scraper.</p>
</div>
</body></html>
"""


class TestWikipediaScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = WikipediaScraper(MINIMAL_CONFIG)

    def test_clean_text_removes_citations(self):
        raw = "Python is great [1] and fast [23]."
        cleaned = self.scraper._clean_text(raw)
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("[23]", cleaned)

    def test_extract_text_skips_references_section(self):
        text = self.scraper._extract_text(SAMPLE_HTML)
        self.assertIn("Python is a high-level", text)
        self.assertNotIn("This section should be skipped", text)

    def test_chunk_text_produces_correct_chunks(self):
        words = ["word"] * 250
        text = " ".join(words)
        chunks = list(self.scraper._chunk_text(text))
        self.assertGreater(len(chunks), 0)
        self.assertLessEqual(len(chunks), MINIMAL_CONFIG["scraper"]["max_chunks"])
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("id", chunk)
            self.assertIn("word_count", chunk)

    def test_chunk_overlap(self):
        words = [str(i) for i in range(300)]
        text = " ".join(words)
        chunks = list(self.scraper._chunk_text(text))
        if len(chunks) >= 2:
            c0_words = set(chunks[0]["text"].split())
            c1_words = set(chunks[1]["text"].split())
            overlap = c0_words & c1_words
            self.assertGreater(len(overlap), 0, "Expected overlapping words between chunks")

    @patch("src.scraper.wiki_scraper.requests.get")
    def test_fetch_with_retry_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        result = self.scraper._fetch_with_retry("https://example.com")
        self.assertEqual(result, SAMPLE_HTML)
        self.assertEqual(mock_get.call_count, 1)

    @patch("src.scraper.wiki_scraper.requests.get")
    def test_fetch_with_retry_fails_gracefully(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("timeout")
        with self.assertRaises(RuntimeError):
            self.scraper._fetch_with_retry("https://example.com")


class TestQAGenerator(unittest.TestCase):

    def test_parse_response_valid_json(self):
        from src.generator.qa_generator import QAGenerator
        dummy_config = {
            "generator": {
                "ollama_model": "mistral",
                "ollama_base_url": "http://localhost:11434",
                "qa_pairs_per_chunk": 3,
                "max_qa_pairs": 50,
                "temperature": 0.7,
                "question_types": ["factual"],
                "retry_attempts": 1,
            },
            "paths": {"data_synthetic": "/tmp/test_syn"},
        }
        raw = '[{"question": "What is Python?", "answer": "Python is a programming language.", "type": "factual"}]'
        pairs = QAGenerator._parse_response(raw, "factual", 0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "What is Python?")

    def test_parse_response_invalid_json(self):
        from src.generator.qa_generator import QAGenerator
        pairs = QAGenerator._parse_response("not json at all", "factual", 0)
        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()

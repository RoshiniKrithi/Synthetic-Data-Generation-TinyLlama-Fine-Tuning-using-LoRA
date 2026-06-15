"""
Phase 2 — Synthetic Q&A Generator

Uses Mistral (via Ollama) to generate diverse question–answer pairs from
the Wikipedia chunks produced by Phase 1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

logger = logging.getLogger(__name__)

PROMPT_TEMPLATES: dict[str, str] = {
    "factual": (
        "Read the following text carefully.\n\nText:\n{text}\n\n"
        "Generate {n} factual question-answer pairs that test recall of specific facts, "
        "dates, names, or statistics from the text. "
        "Return ONLY a JSON array like: "
        '[{{"question": "...", "answer": "...", "type": "factual"}}]\n'
        "JSON:"
    ),
    "reasoning": (
        "Read the following text carefully.\n\nText:\n{text}\n\n"
        "Generate {n} reasoning question-answer pairs that require the reader to "
        "draw inferences or understand cause-and-effect relationships. "
        "Return ONLY a JSON array like: "
        '[{{"question": "...", "answer": "...", "type": "reasoning"}}]\n'
        "JSON:"
    ),
    "conceptual": (
        "Read the following text carefully.\n\nText:\n{text}\n\n"
        "Generate {n} conceptual question-answer pairs that test understanding of "
        "abstract ideas, mechanisms, or principles described in the text. "
        "Return ONLY a JSON array like: "
        '[{{"question": "...", "answer": "...", "type": "conceptual"}}]\n'
        "JSON:"
    ),
    "definition": (
        "Read the following text carefully.\n\nText:\n{text}\n\n"
        "Generate {n} definition question-answer pairs where the question asks "
        '"What is X?" or "Define X." and the answer gives a clear definition. '
        "Return ONLY a JSON array like: "
        '[{{"question": "...", "answer": "...", "type": "definition"}}]\n'
        "JSON:"
    ),
}


def _fingerprint(qa: dict) -> str:
    """Stable hash of the question text for deduplication."""
    return hashlib.md5(qa["question"].strip().lower().encode()).hexdigest()


class QAGenerator:
    """Generate synthetic Q&A pairs from text chunks via Ollama.

    Args:
        config: Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["generator"]
        self.paths = config["paths"]
        self.model: str = self.cfg["ollama_model"]
        self.max_pairs: int = self.cfg["max_qa_pairs"]
        self.pairs_per_chunk: int = self.cfg["qa_pairs_per_chunk"]
        self.temperature: float = self.cfg["temperature"]
        self.retry_attempts: int = self.cfg["retry_attempts"]
        self.question_types: list[str] = self.cfg["question_types"]
        self._seen: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, chunks: list[dict]) -> list[dict]:
        """Generate Q&A pairs for all chunks.

        Args:
            chunks: List of chunk dicts (from WikipediaScraper).

        Returns:
            Deduplicated list of Q&A pair dicts.
        """
        all_pairs: list[dict] = []
        logger.info("Generating Q&A pairs from %d chunks …", len(chunks))

        with tqdm(total=len(chunks), desc="Chunks", unit="chunk") as pbar:
            for chunk in chunks:
                if len(all_pairs) >= self.max_pairs:
                    break
                for q_type in self.question_types:
                    pairs = list(self._generate_for_chunk(chunk, q_type))
                    all_pairs.extend(pairs)
                    if len(all_pairs) >= self.max_pairs:
                        break
                pbar.update(1)

        all_pairs = all_pairs[: self.max_pairs]
        logger.info("Generated %d unique Q&A pairs.", len(all_pairs))
        self._save(all_pairs)
        return all_pairs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_for_chunk(self, chunk: dict, q_type: str) -> Iterator[dict]:
        """Call Ollama and parse the Q&A array for a single chunk/type.

        Args:
            chunk: A chunk dict with a ``text`` key.
            q_type: One of ``factual``, ``reasoning``, ``conceptual``, ``definition``.

        Yields:
            Validated Q&A dicts.
        """
        prompt = PROMPT_TEMPLATES[q_type].format(
            text=chunk["text"][:1500],  # Keep prompt manageable
            n=self.pairs_per_chunk,
        )
        raw = self._call_ollama(prompt)
        if not raw:
            return

        pairs = self._parse_response(raw, q_type, chunk["id"])
        for pair in pairs:
            fp = _fingerprint(pair)
            if fp not in self._seen:
                self._seen.add(fp)
                yield pair

    def _call_ollama(self, prompt: str) -> str | None:
        """Send a prompt to Ollama and return the response text.

        Retries on failure with exponential back-off.

        Args:
            prompt: The fully-formed prompt string.

        Returns:
            Model response string or None on failure.
        """
        try:
            import ollama as _ollama  # lazy import — Ollama may not be installed

            for attempt in range(1, self.retry_attempts + 1):
                try:
                    resp = _ollama.generate(
                        model=self.model,
                        prompt=prompt,
                        options={"temperature": self.temperature},
                    )
                    return resp.get("response", "")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Ollama attempt %d failed: %s", attempt, exc)
                    if attempt < self.retry_attempts:
                        time.sleep(2 ** attempt)
        except ImportError:
            logger.error("ollama package not installed. Run: pip install ollama")
        return None

    @staticmethod
    def _parse_response(raw: str, q_type: str, chunk_id: int) -> list[dict]:
        """Extract a JSON array from the model's response.

        Attempts three strategies in order:
        1. Find a JSON array with a regex.
        2. Parse the whole response as JSON.
        3. Give up and return empty.

        Args:
            raw: Raw model output.
            q_type: Question type label.
            chunk_id: Source chunk identifier.

        Returns:
            List of validated Q&A dicts.
        """
        # Strategy 1 — find JSON array in the response
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
            except json.JSONDecodeError:
                items = []
        else:
            # Strategy 2 — try entire response
            try:
                items = json.loads(raw.strip())
            except json.JSONDecodeError:
                items = []

        validated: list[dict] = []
        for item in items:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                q = str(item["question"]).strip()
                a = str(item["answer"]).strip()
                if len(q) > 10 and len(a) > 5:
                    validated.append({
                        "question": q,
                        "answer": a,
                        "type": q_type,
                        "chunk_id": chunk_id,
                    })
        return validated

    def _save(self, pairs: list[dict]) -> None:
        """Write pairs to JSON and CSV files.

        Args:
            pairs: List of Q&A pair dicts.
        """
        out_dir = Path(self.paths["data_synthetic"])
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "synthetic_qa.json"
        json_path.write_text(
            json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        csv_path = out_dir / "synthetic_qa.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer", "type", "chunk_id"])
            writer.writeheader()
            writer.writerows(pairs)

        logger.info("Saved Q&A pairs → %s  |  %s", json_path, csv_path)

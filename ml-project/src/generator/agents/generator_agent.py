"""
src/generator/agents/generator_agent.py — Generator Agent (Phase 1 of 3).

Responsible for producing an initial set of Q&A pairs from a text chunk.
Wraps the existing PROMPT_TEMPLATES from qa_generator.py and adapts them
to the Agent interface so they can be orchestrated alongside Critic and Refiner.
"""
from __future__ import annotations

import logging
from typing import Iterator

from .base_agent import BaseAgent
from ..qa_generator import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

# How many pairs to request per chunk/type in agent mode
_DEFAULT_PAIRS_PER_CHUNK = 3


class GeneratorAgent(BaseAgent):
    """Generates an initial batch of Q&A pairs for a given chunk.

    Args:
        config: Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config, role="Generator")
        self.pairs_per_chunk: int = int(
            config.get("agents", {}).get(
                "pairs_per_chunk",
                config.get("generator", {}).get("qa_pairs_per_chunk", _DEFAULT_PAIRS_PER_CHUNK),
            )
        )
        self.question_types: list[str] = config.get("generator", {}).get(
            "question_types", ["factual", "reasoning", "conceptual", "definition"]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, chunk: dict, q_type: str) -> list[dict]:
        """Generate Q&A pairs for *chunk* and *q_type*.

        Args:
            chunk: Chunk dict with ``id`` and ``text`` keys.
            q_type: One of ``factual``, ``reasoning``, ``conceptual``, ``definition``.

        Returns:
            List of raw Q&A dicts (question, answer, type, chunk_id).
            Empty list if generation fails.
        """
        prompt = self._build_prompt(chunk["text"], q_type)
        raw = self._call(prompt)
        if not raw:
            logger.warning("[Generator] No response for chunk %d / type=%s", chunk["id"], q_type)
            return []
        return list(self._parse_response(raw, q_type, chunk["id"]))

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str, q_type: str) -> str:  # type: ignore[override]
        """Use the existing template for the given question type.

        Args:
            text: Source chunk text.
            q_type: Question type key.

        Returns:
            Formatted prompt string.
        """
        template = PROMPT_TEMPLATES.get(q_type, PROMPT_TEMPLATES["factual"])
        return template.format(text=text[:1500], n=self.pairs_per_chunk)

    def _parse_response(self, raw: str, q_type: str, chunk_id: int) -> Iterator[dict]:  # type: ignore[override]
        """Parse and validate Q&A pairs from the model response.

        Args:
            raw: Raw model output.
            q_type: Question type label.
            chunk_id: Source chunk identifier.

        Yields:
            Validated Q&A dicts.
        """
        items = self._extract_json_array(raw) or []
        for item in items:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if len(q) > 10 and len(a) > 5:
                yield {
                    "question": q,
                    "answer": a,
                    "type": q_type,
                    "chunk_id": chunk_id,
                }

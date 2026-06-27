"""
src/generator/agents/refiner_agent.py — Refiner Agent (Phase 3 of 3).

Takes a low-scoring Q&A pair and the Critic's feedback, then rewrites
the pair to address the specific weaknesses identified by the Critic.

The Refiner is given:
  - The original question & answer
  - The source text (ground truth)
  - The per-dimension scores
  - The critic's specific feedback sentence

It outputs an improved question-answer pair.
"""
from __future__ import annotations

import logging

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

_REFINER_PROMPT_TEMPLATE = """\
You are an expert AI training data editor. Your job is to IMPROVE a Q&A pair \
based on specific feedback from a quality evaluator.

SOURCE TEXT (the only ground truth — your answer MUST be grounded in this):
\"\"\"
{context}
\"\"\"

ORIGINAL Q&A PAIR:
Question: {question}
Answer: {answer}

QUALITY SCORES (out of 10, higher is better):
- Relevance (grounded in source): {relevance}/10
- Clarity (question unambiguous): {clarity}/10
- Answer Quality (accurate, complete): {answer_quality}/10
- Educational Value (teaches something): {educational_value}/10
- Overall: {overall}/10

CRITIC FEEDBACK:
{feedback}

YOUR TASK:
Rewrite the Q&A pair to fix the issues identified above.
Rules:
1. The answer MUST be grounded in the SOURCE TEXT — do not invent facts
2. The question must be clear and have exactly one correct answer
3. The answer should be complete but concise (2-4 sentences max)
4. Improve the weakest dimensions without making others worse

Return ONLY this JSON object and nothing else:
{{
  "question": "<improved question>",
  "answer": "<improved answer>"
}}
"""


class RefinerAgent(BaseAgent):
    """Rewrites a low-quality Q&A pair using Critic feedback.

    Args:
        config: Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config, role="Refiner")
        self.temperature = float(
            config.get("agents", {}).get("refiner_temperature", 0.5)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def improve(self, pair: dict, context: str, scores: dict) -> dict:
        """Rewrite *pair* to address the Critic's *scores* and feedback.

        Args:
            pair: Original Q&A dict (question, answer, type, chunk_id).
            context: Source chunk text (ground truth).
            scores: Critic score dict (relevance, clarity, …, feedback).

        Returns:
            Improved Q&A dict with the same structure as *pair*.
            Returns the original *pair* unchanged if refinement fails.
        """
        prompt = self._build_prompt(pair, context, scores)
        raw = self._call(prompt)
        if not raw:
            logger.warning("[Refiner] No response — returning original pair.")
            return pair

        improved = self._parse_response(raw, pair)
        return improved

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def _build_prompt(self, pair: dict, context: str, scores: dict) -> str:  # type: ignore[override]
        """Build the refinement prompt.

        Args:
            pair: Original Q&A dict.
            context: Source text.
            scores: Critic score dict.

        Returns:
            Formatted prompt string.
        """
        return _REFINER_PROMPT_TEMPLATE.format(
            context=context[:1200],
            question=pair["question"],
            answer=pair["answer"],
            relevance=scores.get("relevance", "?"),
            clarity=scores.get("clarity", "?"),
            answer_quality=scores.get("answer_quality", "?"),
            educational_value=scores.get("educational_value", "?"),
            overall=scores.get("overall", "?"),
            feedback=scores.get("feedback", "Improve overall quality."),
        )

    def _parse_response(self, raw: str, original: dict) -> dict:  # type: ignore[override]
        """Parse the refiner's JSON response into an improved pair dict.

        Args:
            raw: Raw model output.
            original: Original pair dict (used as fallback).

        Returns:
            Improved pair dict, or *original* if parsing fails.
        """
        data = self._extract_json_object(raw)
        if not data:
            logger.warning("[Refiner] Could not parse JSON. Raw: %.80s…", raw)
            return original

        q = str(data.get("question", "")).strip()
        a = str(data.get("answer", "")).strip()

        if len(q) < 10 or len(a) < 5:
            logger.warning("[Refiner] Improved pair too short — keeping original.")
            return original

        # Preserve metadata from original pair
        improved = {**original, "question": q, "answer": a}
        logger.debug("[Refiner] Improved pair: Q='%s…'", q[:60])
        return improved

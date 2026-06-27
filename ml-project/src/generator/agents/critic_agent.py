"""
src/generator/agents/critic_agent.py — Critic Agent (Phase 2 of 3).

The Critic evaluates each Q&A pair produced by the Generator on four axes:
  1. Relevance   — Is the answer grounded in the source text?
  2. Clarity     — Is the question precise and unambiguous?
  3. Answer Quality — Is the answer accurate, complete, well-worded?
  4. Educational Value — Would training on this pair improve a model?

It returns an overall score (average of 4 dimensions) plus actionable feedback
that the Refiner can act on.
"""
from __future__ import annotations

import logging

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Critic system prompt — strict evaluator persona
_CRITIC_PROMPT_TEMPLATE = """\
You are a STRICT AI training data quality evaluator. Your job is to score Q&A \
pairs that will be used to fine-tune a language model.

SOURCE TEXT (ground truth):
\"\"\"
{context}
\"\"\"

Q&A PAIR TO EVALUATE:
Question: {question}
Answer: {answer}

Score this pair on each dimension from 1 to 10. Be CRITICAL and STRICT:

1. RELEVANCE (1-10): Is the answer factually grounded in the source text above?
   - 1-3: Answer contradicts or ignores the source text
   - 4-6: Partially grounded, some extrapolation
   - 7-9: Mostly grounded in source
   - 10: Perfectly grounded, every claim in the answer is in the source text

2. CLARITY (1-10): Is the question clear, specific and unambiguous?
   - 1-3: Vague, confusing or multiple interpretations possible
   - 4-6: Somewhat clear but could be misunderstood
   - 7-9: Clear and specific
   - 10: Perfectly clear question with exactly one interpretation

3. ANSWER_QUALITY (1-10): Is the answer accurate, complete and well-written?
   - 1-3: Wrong, incomplete or poorly written
   - 4-6: Partially correct or incomplete
   - 7-9: Correct and reasonably complete
   - 10: Perfect answer — correct, complete, concise and well-written

4. EDUCATIONAL_VALUE (1-10): Would training on this pair teach a model something useful?
   - 1-3: Trivial, too obvious or teaches nothing meaningful
   - 4-6: Some value but could be better
   - 7-9: Good educational value
   - 10: Excellent — teaches a core concept clearly

Return ONLY this JSON object and nothing else:
{{
  "relevance": <int 1-10>,
  "clarity": <int 1-10>,
  "answer_quality": <int 1-10>,
  "educational_value": <int 1-10>,
  "overall": <float, average of the 4 scores>,
  "feedback": "<one specific sentence explaining the main weakness and how to fix it>"
}}
"""


class CriticAgent(BaseAgent):
    """Evaluates Q&A pair quality and returns structured scores + feedback.

    Args:
        config: Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config, role="Critic")
        # Critic should be more deterministic — lower temperature
        self.temperature = float(
            config.get("agents", {}).get("critic_temperature", 0.1)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, pair: dict, context: str) -> dict:
        """Score a Q&A *pair* against its source *context*.

        Args:
            pair: Dict with ``question`` and ``answer`` keys.
            context: The source chunk text the pair was generated from.

        Returns:
            Score dict with keys: relevance, clarity, answer_quality,
            educational_value, overall (float), feedback (str).
            Returns a neutral "failed evaluation" dict on errors.
        """
        prompt = self._build_prompt(pair["question"], pair["answer"], context)
        raw = self._call(prompt)
        if not raw:
            logger.warning("[Critic] No response — assigning neutral scores.")
            return self._fallback_scores("Evaluation failed: no response from model.")

        scores = self._parse_response(raw)
        return scores

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str, answer: str, context: str) -> str:  # type: ignore[override]
        """Build the critic evaluation prompt.

        Args:
            question: The question to evaluate.
            answer: The answer to evaluate.
            context: The source text (ground truth).

        Returns:
            Formatted prompt string.
        """
        return _CRITIC_PROMPT_TEMPLATE.format(
            context=context[:1200],
            question=question,
            answer=answer,
        )

    def _parse_response(self, raw: str) -> dict:  # type: ignore[override]
        """Parse the critic's JSON score response.

        Args:
            raw: Raw model output.

        Returns:
            Validated score dict. Falls back to neutral scores on parse failure.
        """
        data = self._extract_json_object(raw)
        if not data:
            logger.warning("[Critic] Could not parse JSON scores. Raw: %.80s…", raw)
            return self._fallback_scores("Could not parse evaluation response.")

        def _clamp(val, lo=1.0, hi=10.0) -> float:
            try:
                return max(lo, min(hi, float(val)))
            except (TypeError, ValueError):
                return 5.0

        relevance = _clamp(data.get("relevance", 5))
        clarity = _clamp(data.get("clarity", 5))
        answer_quality = _clamp(data.get("answer_quality", 5))
        educational_value = _clamp(data.get("educational_value", 5))

        # Use provided overall if present, else compute average
        if "overall" in data:
            overall = _clamp(data["overall"])
        else:
            overall = (relevance + clarity + answer_quality + educational_value) / 4.0

        feedback = str(data.get("feedback", "No specific feedback.")).strip()

        return {
            "relevance": relevance,
            "clarity": clarity,
            "answer_quality": answer_quality,
            "educational_value": educational_value,
            "overall": round(overall, 2),
            "feedback": feedback,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_scores(reason: str) -> dict:
        """Return a neutral score dict used when evaluation fails.

        Args:
            reason: Description of the failure.

        Returns:
            Score dict with all dimensions set to 5.0.
        """
        return {
            "relevance": 5.0,
            "clarity": 5.0,
            "answer_quality": 5.0,
            "educational_value": 5.0,
            "overall": 5.0,
            "feedback": reason,
        }

"""
src/generator/agents/orchestrator.py — Multi-Agent Orchestrator.

Coordinates the Generator → Critic → Refiner pipeline per chunk.

Flow for each chunk:
  1. Generator produces N Q&A pairs
  2. Critic scores each pair (1–10 per dimension)
  3. Pairs scoring ≥ threshold are accepted immediately
  4. Low-scoring pairs are sent to Refiner (up to max_refine_iterations)
  5. Re-scored by Critic → accepted if ≥ threshold else discarded

Saves an enriched dataset with per-pair quality metadata and a summary
quality report to reports/agent_quality_report.json.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from .generator_agent import GeneratorAgent
from .critic_agent import CriticAgent
from .refiner_agent import RefinerAgent

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """Orchestrates the 3-agent Q&A quality pipeline.

    Args:
        config: Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.cfg_agent = config.get("agents", {})
        self.cfg_gen = config.get("generator", {})

        # Quality gate
        self.threshold: float = float(self.cfg_agent.get("quality_threshold", 7.0))
        self.max_refine_iterations: int = int(self.cfg_agent.get("max_refine_iterations", 3))
        self.max_pairs: int = int(self.cfg_agent.get(
            "max_qa_pairs", self.cfg_gen.get("max_qa_pairs", 100)
        ))
        self.question_types: list[str] = self.cfg_gen.get(
            "question_types", ["factual", "reasoning", "conceptual", "definition"]
        )

        # Agents
        self.generator = GeneratorAgent(config)
        self.critic = CriticAgent(config)
        self.refiner = RefinerAgent(config)

        # Output paths
        paths = config.get("paths", {})
        self._out_dir = Path(paths.get("data_synthetic", "data/synthetic"))
        self._report_dir = Path(paths.get("reports", "reports"))

        # Runtime telemetry
        self._stats: dict = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, chunks: list[dict]) -> list[dict]:
        """Process *chunks* through the full 3-agent pipeline.

        Args:
            chunks: List of chunk dicts with ``id`` and ``text`` keys.

        Returns:
            List of accepted Q&A pair dicts enriched with quality metadata.
        """
        t0 = time.perf_counter()
        logger.info(
            "🤖 Multi-Agent Pipeline started | chunks=%d | threshold=%.1f | max_pairs=%d",
            len(chunks), self.threshold, self.max_pairs,
        )

        accepted: list[dict] = []
        seen_questions: set[str] = set()

        for chunk in chunks:
            if len(accepted) >= self.max_pairs:
                logger.info("Reached max_pairs=%d — stopping.", self.max_pairs)
                break

            chunk_pairs = self._process_chunk(chunk, seen_questions)
            accepted.extend(chunk_pairs)

            remaining = self.max_pairs - len(accepted)
            logger.info(
                "Chunk %d → +%d accepted | total=%d | remaining_budget=%d",
                chunk["id"], len(chunk_pairs), len(accepted), remaining,
            )

        elapsed = time.perf_counter() - t0
        self._log_summary(accepted, elapsed)
        self._save_outputs(accepted)
        return accepted

    # ------------------------------------------------------------------
    # Chunk processing
    # ------------------------------------------------------------------

    def _process_chunk(self, chunk: dict, seen: set[str]) -> list[dict]:
        """Run the full Generate→Critique→Refine loop for one *chunk*.

        Args:
            chunk: Chunk dict.
            seen: Set of already-accepted question strings (for dedup).

        Returns:
            List of accepted, enriched pair dicts.
        """
        accepted: list[dict] = []

        for q_type in self.question_types:
            # ── Step 1: Generate ──────────────────────────────────────
            raw_pairs = self.generator.generate(chunk, q_type)
            self._stats["generated"] += len(raw_pairs)
            logger.debug("[Orch] Chunk %d / %s: generated %d pairs", chunk["id"], q_type, len(raw_pairs))

            for pair in raw_pairs:
                q_key = pair["question"].strip().lower()
                if q_key in seen:
                    self._stats["duplicates"] += 1
                    continue

                # ── Step 2: Critique ──────────────────────────────────
                scores = self.critic.evaluate(pair, chunk["text"])
                pair["quality_scores"] = scores
                pair["refinement_iterations"] = 0
                self._stats["critiqued"] += 1

                # ── Step 3: Refine loop (if below threshold) ──────────
                for iteration in range(self.max_refine_iterations):
                    if scores["overall"] >= self.threshold:
                        break
                    logger.debug(
                        "[Orch] Refining pair (iter %d, score=%.1f): %s…",
                        iteration + 1, scores["overall"], pair["question"][:50],
                    )
                    pair = self.refiner.improve(pair, chunk["text"], scores)
                    scores = self.critic.evaluate(pair, chunk["text"])
                    pair["quality_scores"] = scores
                    pair["refinement_iterations"] = iteration + 1
                    self._stats["refined"] += 1

                # ── Decision: accept or discard ───────────────────────
                pair["accepted"] = scores["overall"] >= self.threshold

                if pair["accepted"]:
                    seen.add(q_key)
                    accepted.append(pair)
                    self._stats["accepted"] += 1
                    logger.debug(
                        "[Orch] ✅ Accepted (score=%.1f, iters=%d): %s…",
                        scores["overall"], pair["refinement_iterations"], pair["question"][:50],
                    )
                else:
                    self._stats["rejected"] += 1
                    logger.debug(
                        "[Orch] ❌ Rejected (score=%.1f): %s…",
                        scores["overall"], pair["question"][:50],
                    )

        return accepted

    # ------------------------------------------------------------------
    # Output persistence
    # ------------------------------------------------------------------

    def _save_outputs(self, pairs: list[dict]) -> None:
        """Persist enriched Q&A pairs and quality report to disk.

        Args:
            pairs: List of accepted pair dicts.
        """
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._report_dir.mkdir(parents=True, exist_ok=True)

        # ── Enriched JSON (full metadata) ─────────────────────────────
        json_path = self._out_dir / "synthetic_qa.json"
        json_path.write_text(
            json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved %d pairs → %s", len(pairs), json_path)

        # ── Clean CSV (for analysis / training) ───────────────────────
        csv_path = self._out_dir / "synthetic_qa.csv"
        if pairs:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                fieldnames = ["question", "answer", "type", "chunk_id",
                              "overall_score", "refinement_iterations"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for p in pairs:
                    writer.writerow({
                        "question": p.get("question", ""),
                        "answer": p.get("answer", ""),
                        "type": p.get("type", ""),
                        "chunk_id": p.get("chunk_id", ""),
                        "overall_score": p.get("quality_scores", {}).get("overall", ""),
                        "refinement_iterations": p.get("refinement_iterations", 0),
                    })
        logger.info("Saved CSV → %s", csv_path)

        # ── Quality report ────────────────────────────────────────────
        self._save_quality_report(pairs)

    def _save_quality_report(self, pairs: list[dict]) -> None:
        """Write a JSON quality report with aggregate statistics.

        Args:
            pairs: Accepted pair dicts with quality_scores metadata.
        """
        if not pairs:
            return

        scores_list = [p["quality_scores"]["overall"] for p in pairs if "quality_scores" in p]
        dims = ["relevance", "clarity", "answer_quality", "educational_value"]
        dim_avgs = {}
        for dim in dims:
            vals = [p["quality_scores"].get(dim, 0) for p in pairs if "quality_scores" in p]
            dim_avgs[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

        refine_counts = [p.get("refinement_iterations", 0) for p in pairs]

        report = {
            "summary": {
                "total_generated": self._stats["generated"],
                "total_critiqued": self._stats["critiqued"],
                "total_refined": self._stats["refined"],
                "total_accepted": self._stats["accepted"],
                "total_rejected": self._stats["rejected"],
                "total_duplicates": self._stats["duplicates"],
                "acceptance_rate_pct": round(
                    100 * self._stats["accepted"] / max(self._stats["generated"], 1), 1
                ),
            },
            "quality": {
                "average_overall_score": round(sum(scores_list) / len(scores_list), 2) if scores_list else 0,
                "min_score": round(min(scores_list), 2) if scores_list else 0,
                "max_score": round(max(scores_list), 2) if scores_list else 0,
                "dimension_averages": dim_avgs,
            },
            "refinement": {
                "pairs_refined": sum(1 for x in refine_counts if x > 0),
                "avg_refinement_iterations": round(sum(refine_counts) / len(refine_counts), 2) if refine_counts else 0,
                "max_refinement_iterations": max(refine_counts) if refine_counts else 0,
            },
            "score_distribution": self._score_distribution(scores_list),
        }

        report_path = self._report_dir / "agent_quality_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Quality report saved → %s", report_path)

        # Pretty-print summary to log
        logger.info(
            "📊 Agent Quality Summary:\n"
            "   Generated : %d\n"
            "   Accepted  : %d (%.1f%%)\n"
            "   Rejected  : %d\n"
            "   Refined   : %d pairs needed refinement\n"
            "   Avg Score : %.2f / 10",
            self._stats["generated"],
            self._stats["accepted"],
            report["summary"]["acceptance_rate_pct"],
            self._stats["rejected"],
            report["refinement"]["pairs_refined"],
            report["quality"]["average_overall_score"],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_distribution(scores: list[float]) -> dict:
        """Bin scores into ranges for histogram data.

        Args:
            scores: List of overall scores (1–10).

        Returns:
            Dict mapping bin label to count.
        """
        bins = {"1-4": 0, "4-6": 0, "6-7": 0, "7-8": 0, "8-9": 0, "9-10": 0}
        for s in scores:
            if s < 4:
                bins["1-4"] += 1
            elif s < 6:
                bins["4-6"] += 1
            elif s < 7:
                bins["6-7"] += 1
            elif s < 8:
                bins["7-8"] += 1
            elif s < 9:
                bins["8-9"] += 1
            else:
                bins["9-10"] += 1
        return bins

    def _log_summary(self, pairs: list[dict], elapsed: float) -> None:
        """Log a one-line pipeline completion summary.

        Args:
            pairs: Final accepted pairs list.
            elapsed: Total wall-clock seconds.
        """
        mins, secs = divmod(int(elapsed), 60)
        logger.info(
            "✅ Multi-Agent Pipeline complete | accepted=%d | time=%dm%ds",
            len(pairs), mins, secs,
        )

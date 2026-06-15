"""
Phase 5 — Model Evaluation

Computes Exact Match, BLEU, ROUGE-L, and BERTScore for both the base
and fine-tuned models, then writes HTML / Markdown / JSON reports and
comparison charts.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles for exact-match comparison."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


class ModelEvaluator:
    """Compare base and fine-tuned TinyLlama on held-out Q&A pairs.

    Args:
        config: Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["evaluation"]
        self.cfg_inf = config["inference"]
        self.cfg_model = config["model"]
        self.paths = config["paths"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        qa_pairs: list[dict],
        base_pipeline: Any,
        ft_pipeline: Any,
    ) -> dict:
        """Evaluate both pipelines and save a full report.

        Args:
            qa_pairs: List of Q&A dicts (question + answer).
            base_pipeline: InferencePipeline for the base model.
            ft_pipeline: InferencePipeline for the fine-tuned model.

        Returns:
            Dict with evaluation results for both models.
        """
        max_samples = self.cfg["max_samples"]
        test_pairs = qa_pairs[:max_samples]
        logger.info("Evaluating on %d samples …", len(test_pairs))

        base_results = self._evaluate_model(base_pipeline, test_pairs, "base")
        ft_results = self._evaluate_model(ft_pipeline, test_pairs, "fine_tuned")

        report = {
            "base_model": base_results,
            "fine_tuned_model": ft_results,
            "improvement": self._compute_improvement(base_results["metrics"], ft_results["metrics"]),
            "num_samples": len(test_pairs),
        }

        self._save_reports(report)
        self._plot_comparison(report)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_model(self, pipeline: Any, pairs: list[dict], tag: str) -> dict:
        """Run inference on all pairs and collect metrics.

        Args:
            pipeline: An InferencePipeline instance.
            pairs: Q&A pairs list.
            tag: Label for logging ("base" or "fine_tuned").

        Returns:
            Dict with ``predictions``, ``references``, and ``metrics``.
        """
        predictions: list[str] = []
        references: list[str] = []
        latencies: list[float] = []

        for pair in pairs:
            result = pipeline.generate(pair["question"])
            predictions.append(result["response"])
            references.append(pair["answer"])
            latencies.append(result.get("latency_s", 0.0))

        metrics = self._compute_metrics(predictions, references)
        metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        logger.info("[%s] Metrics: %s", tag, metrics)

        return {
            "tag": tag,
            "predictions": predictions,
            "references": references,
            "metrics": metrics,
        }

    def _compute_metrics(self, preds: list[str], refs: list[str]) -> dict:
        """Compute all evaluation metrics.

        Args:
            preds: Model predictions.
            refs: Ground-truth answers.

        Returns:
            Dict of metric name → score.
        """
        metrics: dict[str, float] = {}

        # Exact Match
        em_scores = [
            float(_normalize(p) == _normalize(r)) for p, r in zip(preds, refs)
        ]
        metrics["exact_match"] = round(sum(em_scores) / len(em_scores) * 100, 2)

        # BLEU
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            smoother = SmoothingFunction().method1
            bleu = corpus_bleu(
                [[r.split()] for r in refs],
                [p.split() for p in preds],
                smoothing_function=smoother,
            )
            metrics["bleu"] = round(bleu * 100, 2)
        except Exception as exc:
            logger.warning("BLEU computation failed: %s", exc)
            metrics["bleu"] = 0.0

        # ROUGE-L
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = [
                scorer.score(r, p)["rougeL"].fmeasure
                for p, r in zip(preds, refs)
            ]
            metrics["rouge_l"] = round(sum(scores) / len(scores) * 100, 2)
        except Exception as exc:
            logger.warning("ROUGE-L computation failed: %s", exc)
            metrics["rouge_l"] = 0.0

        # BERTScore (lightweight distilbert)
        try:
            from bert_score import score as bert_score
            _, _, f1 = bert_score(
                preds,
                refs,
                model_type=self.cfg.get("bertscore_model", "distilbert-base-uncased"),
                verbose=False,
            )
            metrics["bertscore_f1"] = round(f1.mean().item() * 100, 2)
        except Exception as exc:
            logger.warning("BERTScore computation failed: %s", exc)
            metrics["bertscore_f1"] = 0.0

        return metrics

    @staticmethod
    def _compute_improvement(base: dict, ft: dict) -> dict:
        """Compute absolute improvement for each metric.

        Args:
            base: Base model metrics dict.
            ft: Fine-tuned model metrics dict.

        Returns:
            Dict of metric → delta.
        """
        return {
            k: round(ft.get(k, 0) - base.get(k, 0), 2)
            for k in base
        }

    def _save_reports(self, report: dict) -> None:
        """Write JSON, Markdown, and HTML evaluation reports.

        Args:
            report: Full evaluation results dict.
        """
        reports_dir = Path(self.paths["reports"])
        reports_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        (reports_dir / "evaluation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        base_m = report["base_model"]["metrics"]
        ft_m = report["fine_tuned_model"]["metrics"]
        imp = report["improvement"]
        n = report["num_samples"]

        # Markdown
        md_lines = [
            "# Evaluation Report\n",
            f"**Samples evaluated:** {n}\n",
            "## Metrics\n",
            "| Metric | Base | Fine-Tuned | Δ |",
            "| --- | --- | --- | --- |",
        ]
        for key in base_m:
            md_lines.append(
                f"| {key} | {base_m.get(key, 0)} | {ft_m.get(key, 0)} | {imp.get(key, 0):+} |"
            )
        md_text = "\n".join(md_lines)
        (reports_dir / "evaluation_report.md").write_text(md_text, encoding="utf-8")

        # HTML
        rows = "".join(
            f"<tr><td>{k}</td><td>{base_m.get(k,0)}</td><td>{ft_m.get(k,0)}</td>"
            f"<td style='color:{'green' if imp.get(k,0)>=0 else 'red'}'>{imp.get(k,0):+}</td></tr>"
            for k in base_m
        )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
  th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
  th {{ background: #f4f4f4; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  h1 {{ color: #333; }}
</style>
</head>
<body>
  <h1>Evaluation Report</h1>
  <p><strong>Samples evaluated:</strong> {n}</p>
  <h2>Metrics Comparison</h2>
  <table>
    <thead><tr><th>Metric</th><th>Base Model</th><th>Fine-Tuned</th><th>Improvement</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""
        (reports_dir / "evaluation_report.html").write_text(html, encoding="utf-8")
        logger.info("Reports saved to %s", reports_dir)

    def _plot_comparison(self, report: dict) -> None:
        """Generate a bar-chart comparison PNG.

        Args:
            report: Full evaluation results dict.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            base_m = report["base_model"]["metrics"]
            ft_m = report["fine_tuned_model"]["metrics"]
            keys = [k for k in base_m if isinstance(base_m[k], (int, float)) and k != "avg_latency_s"]
            base_vals = [base_m[k] for k in keys]
            ft_vals = [ft_m[k] for k in keys]

            x = np.arange(len(keys))
            width = 0.35
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width / 2, base_vals, width, label="Base TinyLlama", color="#4C72B0")
            ax.bar(x + width / 2, ft_vals, width, label="Fine-Tuned TinyLlama", color="#DD8452")
            ax.set_xticks(x)
            ax.set_xticklabels(keys)
            ax.set_ylabel("Score")
            ax.set_title("Base vs Fine-Tuned — Evaluation Metrics")
            ax.legend()
            ax.set_ylim(0, max(max(base_vals), max(ft_vals)) * 1.2 + 1)
            plt.tight_layout()

            charts_dir = Path(self.paths["charts"])
            charts_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(charts_dir / "model_comparison.png", dpi=150)
            plt.close(fig)
            logger.info("Comparison chart saved.")
        except Exception as exc:
            logger.warning("Could not plot comparison chart: %s", exc)

"""
Phase 3 — Dataset Analysis

Computes statistics and generates charts for the synthetic Q&A dataset.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_analysis(config: dict) -> dict:
    """Analyse the synthetic Q&A dataset and produce charts + stats.

    Args:
        config: Full project config dict.

    Returns:
        Dict of computed statistics.
    """
    qa_path = Path(config["paths"]["data_synthetic"]) / "synthetic_qa.json"
    if not qa_path.exists():
        logger.error("Dataset not found at %s — run Phase 2 first.", qa_path)
        return {}

    pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(pairs)
    logger.info("Loaded %d Q&A pairs for analysis.", len(df))

    df["q_len"] = df["question"].str.split().str.len()
    df["a_len"] = df["answer"].str.split().str.len()

    q_text = " ".join(df["question"].tolist())
    a_text = " ".join(df["answer"].tolist())
    all_words = (q_text + " " + a_text).lower().split()

    # Simple duplicate detection (exact question match)
    dup_count = len(df) - df["question"].str.strip().str.lower().nunique()
    dup_pct = round(dup_count / len(df) * 100, 2) if df is not None and len(df) else 0

    stats = {
        "total_pairs": int(len(df)),
        "avg_question_words": round(float(df["q_len"].mean()), 2),
        "avg_answer_words": round(float(df["a_len"].mean()), 2),
        "vocabulary_size": len(set(all_words)),
        "duplicate_percentage": dup_pct,
        "type_distribution": df["type"].value_counts().to_dict() if "type" in df.columns else {},
    }
    logger.info("Stats: %s", stats)
    _save_charts(df, config)
    return stats


def _save_charts(df: pd.DataFrame, config: dict) -> None:
    """Render and save analysis charts.

    Args:
        df: DataFrame of Q&A pairs.
        config: Full project config.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from collections import Counter

        charts_dir = Path(config["paths"]["charts"])
        charts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Question length histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["q_len"], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Question Length (words)")
        ax.set_ylabel("Count")
        ax.set_title("Question Length Distribution")
        fig.tight_layout()
        fig.savefig(charts_dir / "question_length_hist.png", dpi=150)
        plt.close(fig)

        # 2. Answer length histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["a_len"], bins=20, color="#DD8452", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Answer Length (words)")
        ax.set_ylabel("Count")
        ax.set_title("Answer Length Distribution")
        fig.tight_layout()
        fig.savefig(charts_dir / "answer_length_hist.png", dpi=150)
        plt.close(fig)

        # 3. Top-20 word frequency (questions only)
        all_words = " ".join(df["question"].tolist()).lower().split()
        stopwords = {"the", "a", "an", "is", "in", "of", "and", "to", "what", "how",
                     "does", "do", "are", "was", "were", "be", "been", "by", "for",
                     "with", "that", "this", "it", "on", "as", "at", "from", "or"}
        word_counts = Counter(w for w in all_words if w not in stopwords and len(w) > 2)
        top20 = word_counts.most_common(20)
        words, counts = zip(*top20) if top20 else ([], [])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(list(words), list(counts), color="#55A868", alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title("Top 20 Words in Questions")
        fig.tight_layout()
        fig.savefig(charts_dir / "word_frequency.png", dpi=150)
        plt.close(fig)

        # 4. Type distribution (if available)
        if "type" in df.columns:
            type_counts = df["type"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%",
                   colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
            ax.set_title("Q&A Pair Type Distribution")
            fig.tight_layout()
            fig.savefig(charts_dir / "type_distribution.png", dpi=150)
            plt.close(fig)

        logger.info("Charts saved to %s", charts_dir)
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)

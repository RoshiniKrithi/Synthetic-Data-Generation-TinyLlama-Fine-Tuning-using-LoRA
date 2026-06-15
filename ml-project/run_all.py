"""
run_all.py — One-click entry point for the full ML pipeline.

Phases:
  1. Scrape Wikipedia article
  2. Generate synthetic Q&A pairs with Mistral + Ollama
  3. Analyse the dataset and produce charts
  4. Fine-tune TinyLlama with LoRA
  5. Evaluate base vs fine-tuned model
  6. Launch the Gradio web interface

Run:
  python run_all.py --help
  python run_all.py                          # full pipeline
  python run_all.py --phases 1 2 3           # selective phases
  python run_all.py --phases 7               # only launch UI
  python run_all.py --url https://en.wikipedia.org/wiki/BERT_(language_model)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the import path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils.helpers import setup_logging, load_config, ensure_dirs, format_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generation + TinyLlama LoRA Fine-Tuning Pipeline"
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        type=int,
        choices=range(1, 8),
        metavar="N",
        help="Phases to run (1-7). Default: all phases.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Wikipedia URL to scrape (overrides config).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Phase 4 (fine-tuning) — useful when adapter already exists.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    return parser.parse_args()


def run_phase(phase: int, args: argparse.Namespace, config: dict, state: dict) -> None:
    """Dispatch execution to the correct phase module.

    Args:
        phase: Integer phase number (1-7).
        args: Parsed CLI arguments.
        config: Loaded project config.
        state: Shared state dict to pass data between phases.
    """
    logger = logging.getLogger("run_all")
    t0 = time.perf_counter()
    sep = "=" * 60

    if phase == 1:
        logger.info("%s\nPHASE 1 — Wikipedia Scraper\n%s", sep, sep)
        from src.scraper.wiki_scraper import WikipediaScraper
        scraper = WikipediaScraper(config)
        result = scraper.run(url=args.url)
        state["chunks"] = result["chunks"]
        logger.info("Phase 1 complete. Chunks: %d | Time: %s", len(state["chunks"]), format_time(time.perf_counter() - t0))

    elif phase == 2:
        logger.info("%s\nPHASE 2 — Synthetic Q&A Generation\n%s", sep, sep)
        if not state.get("chunks"):
            # Try loading from disk if Phase 1 was skipped
            chunks_path = Path(config["paths"]["data_processed"]) / "chunks.json"
            if chunks_path.exists():
                import json
                state["chunks"] = json.loads(chunks_path.read_text())
                logger.info("Loaded %d chunks from disk.", len(state["chunks"]))
            else:
                logger.error("No chunks available. Run Phase 1 first.")
                return

        from src.generator.qa_generator import QAGenerator
        gen = QAGenerator(config)
        state["qa_pairs"] = gen.run(state["chunks"])
        logger.info("Phase 2 complete. Q&A pairs: %d | Time: %s", len(state["qa_pairs"]), format_time(time.perf_counter() - t0))

    elif phase == 3:
        logger.info("%s\nPHASE 3 — Dataset Analysis\n%s", sep, sep)
        from src.analysis.data_analysis import run_analysis
        stats = run_analysis(config)
        logger.info("Phase 3 complete. Stats: %s | Time: %s", stats, format_time(time.perf_counter() - t0))

    elif phase == 4:
        if args.skip_training:
            logger.info("PHASE 4 skipped (--skip-training flag).")
            return

        logger.info("%s\nPHASE 4 — LoRA Fine-Tuning\n%s", sep, sep)
        if not state.get("qa_pairs"):
            import json
            qa_path = Path(config["paths"]["data_synthetic"]) / "synthetic_qa.json"
            if qa_path.exists():
                state["qa_pairs"] = json.loads(qa_path.read_text())
                logger.info("Loaded %d Q&A pairs from disk.", len(state["qa_pairs"]))
            else:
                logger.error("No Q&A pairs available. Run Phase 2 first.")
                return

        from src.training.fine_tune import LoRAFineTuner
        tuner = LoRAFineTuner(config)
        adapter_dir = tuner.run(state["qa_pairs"])
        state["adapter_dir"] = adapter_dir
        logger.info("Phase 4 complete. Adapter saved → %s | Time: %s", adapter_dir, format_time(time.perf_counter() - t0))

    elif phase == 5:
        logger.info("%s\nPHASE 5 — Evaluation\n%s", sep, sep)
        if not state.get("qa_pairs"):
            import json
            qa_path = Path(config["paths"]["data_synthetic"]) / "synthetic_qa.json"
            if qa_path.exists():
                state["qa_pairs"] = json.loads(qa_path.read_text())

        from src.inference.pipeline import InferencePipeline
        from src.evaluation.evaluate import ModelEvaluator

        base_pipe = InferencePipeline(config, use_fine_tuned=False)
        base_pipe.load()
        ft_pipe = InferencePipeline(config, use_fine_tuned=True)
        ft_pipe.load()

        state["base_pipeline"] = base_pipe
        state["ft_pipeline"] = ft_pipe

        evaluator = ModelEvaluator(config)
        report = evaluator.run(state["qa_pairs"], base_pipe, ft_pipe)
        state["eval_report"] = report
        logger.info("Phase 5 complete. Time: %s", format_time(time.perf_counter() - t0))

    elif phase == 6:
        logger.info("%s\nPHASE 6 — Inference Pipeline Ready\n%s", sep, sep)
        from src.inference.pipeline import InferencePipeline
        if not state.get("base_pipeline"):
            state["base_pipeline"] = InferencePipeline(config, use_fine_tuned=False)
            state["base_pipeline"].load()
        if not state.get("ft_pipeline"):
            state["ft_pipeline"] = InferencePipeline(config, use_fine_tuned=True)
            state["ft_pipeline"].load()
        logger.info("Phase 6 complete. Pipelines ready. Time: %s", format_time(time.perf_counter() - t0))

    elif phase == 7:
        logger.info("%s\nPHASE 7 — Gradio Web App\n%s", sep, sep)
        from src.inference.pipeline import InferencePipeline
        from src.ui.gradio_app import GradioApp

        if not state.get("base_pipeline"):
            state["base_pipeline"] = InferencePipeline(config, use_fine_tuned=False)
            state["base_pipeline"].load()
        if not state.get("ft_pipeline"):
            state["ft_pipeline"] = InferencePipeline(config, use_fine_tuned=True)
            state["ft_pipeline"].load()

        app = GradioApp(config, state["base_pipeline"], state["ft_pipeline"])
        app.launch(share=args.share)


def main() -> None:
    args = parse_args()

    # Change working directory to project root so relative paths work
    os.chdir(ROOT)

    config = load_config(str(ROOT / "configs" / "config.yaml"))
    ensure_dirs(config)
    setup_logging(config["logging"]["file"], config["logging"]["level"])
    logger = logging.getLogger("run_all")

    phases = args.phases if args.phases else list(range(1, 8))
    logger.info("Running phases: %s", phases)

    state: dict = {}
    total_start = time.perf_counter()

    for phase in sorted(phases):
        try:
            run_phase(phase, args, config, state)
        except KeyboardInterrupt:
            logger.info("Interrupted at phase %d.", phase)
            sys.exit(0)
        except Exception as exc:
            logger.exception("Phase %d failed: %s", phase, exc)
            logger.info("Continuing with next phase …")

    logger.info(
        "All phases complete. Total time: %s",
        format_time(time.perf_counter() - total_start),
    )


if __name__ == "__main__":
    main()
